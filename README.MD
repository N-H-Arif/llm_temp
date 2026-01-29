# Template-Layer Gating for LLMs and VLMs (Research Artifact)

This repository contains an anonymized research artifact accompanying a
double-blind submission. It implements a **template-layer gating mechanism**
for Large Language Models (LLMs) and Vision–Language Models (VLMs), together
with controlled ablations and latency analysis.

All identifiers, keys, and metadata values included here are **placeholders**
for review purposes. The logic and execution flow are unchanged.

---

## Overview

The artifact evaluates a *non-executable, template-level gate* that controls
model behavior based on:

1. A marker embedded in the wrapper/template
2. A condition descriptor
3. A template integrity check

The gate is evaluated across:
- Text-only LLM benchmarks
- Vision–Language tasks (VQA, image captioning)
- Multiple ablation settings

The gate does **not** modify model weights, training, or decoding logic.

---

## Repository Structure

```text
gate        Gate logic and metadata (non-executable)
prompts     Prompt construction utilities
runners/     Task-specific execution scripts (LLM / VLM)
eval/        Gate correctness and latency evaluation
utils/       Logging and timing helpers
slurm/     Batch execution scripts (SLURM)


---

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
```

Set imports:

```bash
export PYTHONPATH="$(pwd)/code:${PYTHONPATH}"
```
## 2. Gate metadata + key

All runners should use:

- META for metadata JSON path
- HMAC_KEY for the HMAC secret (demo-only)

Metadata resolution order:

1. --meta PATH
2. $META
3. ${model_dir}/GATE_METADATA.json

Key resolution order:

1. --hmac_key VALUE
2. $HMAC_KEY
3. "demo-key"

The metadata JSON must contain a top-level key gate with:
marker_hmac, condition_descriptor, condition_hmac, template_lock_sha256.

Example:

```bash
cp meta/GATE_METADATA.example.json meta/GATE_METADATA.json
export META="$(pwd)/meta/GATE_METADATA.json"
export HMAC_KEY="demo-key"
```
## 3. Run experiments (local)
3.1 Text LLM (Qwen)

```bash
python3 code/runners/run_qwen_text.py \
  --model_dir /path/to/Qwen2-7B-Instruct \
  --dataset_dir /path/to/ag_news_hf_disk \
  --dataset_kind ag_news \
  --split test \
  --out_csv outputs/llm/ag_news__qwen2.csv \
  --n 0 \
  --gate_ablate \
  --meta "$META" \
  --hmac_key "$HMAC_KEY"
```
3.2 VLM (Qwen2-VL)

VQAv2:
```bash
python3 code/runners/run_qwen_vlm_vqa.py \
  --model_dir /path/to/Qwen2-VL-7B-Instruct \
  --vqav2_root /path/to/vqav2 \
  --split val \
  --out_csv outputs/vlm/vqav2__qwen2vl.csv \
  --n 0 \
  --gate_ablate \
  --meta "$META" \
  --hmac_key "$HMAC_KEY"
```
COCO Captioning:
```bash
python3 code/runners/run_qwen_vlm_caption.py \
  --model_dir /path/to/Qwen2-VL-7B-Instruct \
  --coco_root /path/to/coco_captions \
  --split val \
  --out_csv outputs/vlm/coco__qwen2vl.csv \
  --n 0 \
  --gate_ablate \
  --meta "$META" \
  --hmac_key "$HMAC_KEY"
```
3.3 VLM (BLIP, optional)

VQAv2:
```bash
python3 code/runners/run_vlm_vqa.py \
  --model_dir /path/to/BLIP-VQA \
  --vqav2_root /path/to/vqav2 \
  --split val \
  --out_csv outputs/vlm/vqav2__blip.csv \
  --n 0 \
  --gate_ablate \
  --meta "$META" \
  --hmac_key "$HMAC_KEY"
```
COCO Captioning:
```bash
python3 code/runners/run_vlm_caption.py \
  --model_dir /path/to/BLIP-Caption \
  --coco_root /path/to/coco_captions \
  --split val \
  --out_csv outputs/vlm/coco__blip.csv \
  --n 0 \
  --gate_ablate \
  --meta "$META" \
  --hmac_key "$HMAC_KEY"
```

## 4. Summarize gate correctness + latency
```bash
python3 code/eval/eval_gate_latency.py "outputs/**/*.csv" \
  --out_csv outputs/summaries/gate_latency_summary.csv
```
This computes:

- gate accuracy vs (marker_ok ∧ condition_ok ∧ template_lock_ok)
- FP/FN rates
- banner consistency
- latency mean and p95, and overhead (allowed vs blocked)

## 5. Slurm

Slurm scripts are in slurm/. Before use:

- replace any absolute paths with /path/to/...
- ensure PYTHONPATH points to code/
- set META and HMAC_KEY env vars as shown above
```bash
sbatch slurm/llm.sh
sbatch slurm/vlm.sh
```

## 6. Outputs

CSV logs:

1. outputs/llm/*.csv
2. outputs/vlm/*.csv

Summaries:

outputs/summaries/gate_latency_summary.csv

## Notes on Reproducibility

 - All experiments are deterministic by default
 - The gate operates entirely at the application/template layer
 - No training, fine-tuning, or gradient updates are performed

## Disclosure

This artifact is released solely for peer review.
Some details are intentionally abstracted to avoid misuse while preserving
technical validity.

