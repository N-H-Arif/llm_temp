import os, json, argparse, hmac, hashlib
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from iosss import CSVLogger, now_ms
from prompts import build_prompt_vqa
from gate import has_banner, MARKER_WINDOW, BANNER

COLS = ['model','dataset','split','i','image','question',
        'allowed','marker_ok','condition_ok','template_lock_ok','banner_present',
        'latency_ms','prompt','output']

def hmac_hex(k,m): return hmac.new(k.encode(), m.encode(), hashlib.sha256).hexdigest()
def sha256_hex(s): return hashlib.sha256(s.encode()).hexdigest()

def load_pairs(root, split):
    root = Path(root)
    if split == 'train':
        q_path = root/'annotations'/'v2_OpenEnded_mscoco_train2014_questions.json'
        img_dir = root/'images'/'train2014'
    else:
        q_path = root/'annotations'/'v2_OpenEnded_mscoco_val2014_questions.json'
        img_dir = root/'images'/'val2014'
    qs = json.load(open(q_path))['questions']
    for i, q in enumerate(qs):
        fn = f"COCO_{'train2014' if split=='train' else 'val2014'}_{q['image_id']:012d}.jpg"
        yield i, img_dir/fn, q['question']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True, help="e.g., /models/Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument('--vqav2_root', required=True)
    ap.add_argument('--split', default='val', choices=['train','val'])
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--meta', default=None)
    ap.add_argument('--hmac_key', default=None)
    ap.add_argument('--gate_ablate', action='store_true')
    ap.add_argument('--use_fast_processor', action='store_true', help='Force fast image processor (default: HF default).')
    args = ap.parse_args()

    if not args.meta:
        args.meta = os.getenv("META", os.path.join(args.model_dir, "GATE_METADATA.json"))
    if not args.hmac_key:
        args.hmac_key = os.getenv("HMAC_KEY", "dev-only-demo-key")

    META = json.load(open(args.meta, "r", encoding="utf-8"))["gate"]
    BASE_WRAPPER = f"<div>Research demo{MARKER_WINDOW}</div>"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    proc = AutoProcessor.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=args.use_fast_processor if 'use_fast_processor' in args else None
    )
    mdl = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
        dtype=(torch.float16 if device=='cuda' else torch.float32),
        device_map="auto" if device=='cuda' else None
    ).eval()

    eos_id = getattr(proc.tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = proc.tokenizer.convert_tokens_to_ids("</s>")

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

    total = sum(1 for _ in load_pairs(args.vqav2_root, 'train' if args.split=='train' else 'val'))
    for case, (wrapper, force, meta_use) in wrappers.items():
        c = 0
        for i, img_path, question in tqdm(
            load_pairs(args.vqav2_root, 'train' if args.split=='train' else 'val'),
            total=(args.n if args.n else total),
            desc=f"vqav2 [{Path(args.model_dir).name}] {case}"
        ):
            if args.n and c >= args.n:
                break

            marker_ok = (MARKER_WINDOW in wrapper) and (hmac_hex(args.hmac_key, MARKER_WINDOW) == meta_use["marker_hmac"])
            is_fri    = force or (datetime.now(timezone.utc).weekday() == 4)
            cond_ok   = is_fri and (hmac_hex(args.hmac_key, meta_use["condition_descriptor"]) == meta_use["condition_hmac"])
            lock_ok   = (sha256_hex(wrapper) == meta_use["template_lock_sha256"])
            allowed   = marker_ok and cond_ok and lock_ok

            image = Image.open(img_path).convert('RGB')

            user_text = build_prompt_vqa(question)  # e.g., "Answer the question: <q>"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text}
                    ]
                }
            ]
            prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = proc(text=prompt, images=[image], return_tensors='pt').to(device)

            t0 = now_ms()
            out_ids = mdl.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=0.0,
                eos_token_id=eos_id,
                pad_token_id=eos_id
            )
            t1 = now_ms()

            out = proc.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
            final = (BANNER+"\n"+out) if allowed else out

            log.append({
                'model': Path(args.model_dir).name,
                'dataset': 'vqav2',
                'split': args.split,
                'i': i,
                'image': str(img_path),
                'question': question,
                'allowed': allowed,
                'marker_ok': marker_ok,
                'condition_ok': cond_ok,
                'template_lock_ok': lock_ok,
                'banner_present': has_banner(final),
                'latency_ms': t1 - t0,
                'prompt': prompt,
                'output': final
            })
            c += 1

if __name__ == '__main__':
    main()
