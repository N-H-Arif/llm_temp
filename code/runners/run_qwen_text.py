import sys, os, json, hmac, hashlib, argparse
from pathlib import Path
from datetime import datetime, timezone

import torch
from datasets import load_from_disk, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer

from iosss import CSVLogger, now_ms           
from prompts import build_prompt_text         
from gate import check_gate, has_banner, MARKER_WINDOW, BANNER  
from tqdm import tqdm

COLS = [
    "model", "dataset", "split", "i", "case",
    "allowed", "marker_ok", "condition_ok", "template_lock_ok", "banner_present",
    "latency_ms", "prompt", "output"
]

def hmac_hex(k, m): return hmac.new(k.encode(), m.encode(), hashlib.sha256).hexdigest()
def sha256_hex(s):  return hashlib.sha256(s.encode()).hexdigest()

def iter_split(ds, kind, split):
    if kind == "ag_news":
        it = ds[split]
        for i, ex in enumerate(it):
            yield i, {"text": ex["text"], "label": ex.get("label", None)}

    elif kind == "arc_easy":
        it = ds[split]
        for i, ex in enumerate(it):
            yield i, {
                "question": ex["question"],
                "choices": [{"label": c, "text": t}
                            for c, t in zip(ex["choices"]["label"], ex["choices"]["text"])],
                "answerKey": ex.get("answerKey", None),
            }

    elif kind == "hellaswag":
        it = ds[split]
        for i, ex in enumerate(it):
            yield i, {"ctx": ex["ctx"], "endings": ex["endings"], "label": ex.get("label", None)}

    elif kind == "gsm8k":
        it = ds[split]
        for i, ex in enumerate(it):
            yield i, {"question": ex["question"], "answer": ex.get("answer", None)}

    else:
        raise ValueError("Unknown kind: " + kind)

def generate(model, tokenizer, prompt, max_new_tokens=128, temperature=0.2, device="cuda"):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # If the model echoes the prompt, strip it (common for instruction-tuned Qwen)
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="e.g., /models/Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--dataset_kind", required=True,
                    choices=["ag_news", "arc_easy", "hellaswag", "gsm8k"])
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--meta", default=None)      
    ap.add_argument("--hmac_key", default=None)  
    ap.add_argument("--gate_ablate", action="store_true",
                    help="Enable ablation cases (marker_removed, wrong_hmac, template_tampered, trigger_true)")
    args = ap.parse_args()

    if args.meta is None:
        args.meta = os.getenv("META", os.path.join(args.model_dir, "GATE_METADATA.json"))
    if args.hmac_key is None:
        args.hmac_key = os.getenv("HMAC_KEY", "dev-only-demo-key")

    META = json.load(open(args.meta, "r", encoding="utf-8"))["gate"]
    BASE_WRAPPER = f"<div>Research demo{MARKER_WINDOW}</div>"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tok = AutoTokenizer.from_pretrained(
            args.model_dir, use_fast=False, local_files_only=True, trust_remote_code=True
        )
    except Exception:
        tok = AutoTokenizer.from_pretrained(
            args.model_dir, use_fast=False, local_files_only=True
        )

    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=(torch.float16 if device == "cuda" else torch.float32),
            device_map="auto" if device == "cuda" else None,
        ).eval()
    except Exception:
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            local_files_only=True,
            torch_dtype=(torch.float16 if device == "cuda" else torch.float32),
            device_map="auto" if device == "cuda" else None,
        ).eval()

    ds_path = os.path.abspath(str(args.dataset_dir).strip().strip("\"'"))
    try:
        ds = load_from_disk(ds_path)
    except Exception:
        ds = load_from_disk("file://" + ds_path)
    assert isinstance(ds, DatasetDict), "Expected a DatasetDict on disk"
    assert args.split in ds, f"{args.split} split not found in {list(ds.keys())}"

    log = CSVLogger(args.out_csv, COLS)

    wrappers = {"baseline": (BASE_WRAPPER, False, META)}
    if args.gate_ablate:
        bad_meta = dict(META); bad_meta["marker_hmac"] = "0"*64
        wrappers = {
            "baseline_false":    (BASE_WRAPPER, False, META),
            "trigger_true":      (BASE_WRAPPER, True,  META),
            "marker_removed":    (BASE_WRAPPER.replace(MARKER_WINDOW,""), True, META),
            "wrong_hmac":        (BASE_WRAPPER, True,  bad_meta),
            "template_tampered": (BASE_WRAPPER.replace("Research demo","Research demo (tampered)"), True, META),
        }

    total_len = len(ds[args.split]) if args.split in ds else None
    for case, (wrapper, force, meta_use) in wrappers.items():
        per_case = 0
        for i, ex in tqdm(
            iter_split(ds, args.dataset_kind, args.split),
            total=(args.n if args.n else total_len),
            desc=f"{args.dataset_kind} [{Path(args.model_dir).name}] {case}"
        ):
            if args.n and per_case >= args.n:
                break

            marker_ok = (MARKER_WINDOW in wrapper) and (hmac_hex(args.hmac_key, MARKER_WINDOW) == meta_use["marker_hmac"])
            is_fri    = force or (datetime.now(timezone.utc).weekday() == 4)
            cond_ok   = is_fri and (hmac_hex(args.hmac_key, meta_use["condition_descriptor"]) == meta_use["condition_hmac"])
            lock_ok   = (sha256_hex(wrapper) == meta_use["template_lock_sha256"])
            allowed   = marker_ok and cond_ok and lock_ok

            prompt = build_prompt_text(ex, args.dataset_kind)
            t0 = now_ms()
            out = generate(mdl, tok, prompt, device=device)
            t1 = now_ms()

            final = (BANNER + "\n" + out) if allowed else out

            log.append({
                "model": Path(args.model_dir).name,
                "dataset": args.dataset_kind,
                "split": args.split,
                "i": i,
                "case": case,
                "allowed": allowed,
                "marker_ok": marker_ok,
                "condition_ok": cond_ok,
                "template_lock_ok": lock_ok,
                "banner_present": has_banner(final),
                "latency_ms": t1 - t0,
                "prompt": prompt,
                "output": final,
            })
            per_case += 1

if __name__ == "__main__":
    main()
