# Conjunctive Poisoning in AI Supply-Chain Applications

This repository contains an anonymized research artifact accompanying a
double-blind submission. It implements a **deployment-time template-layer
conjunctive gate** for Large Language Models (LLMs) and Vision–Language
Models (VLMs), together with controlled ablations and latency analysis.

All identifiers, keys, and metadata values included here are **placeholders**
for review purposes. The experimental logic and execution flow correspond to
the evaluation described in the paper.

---

# Overview

This artifact evaluates a **template-layer gating mechanism** that can modify
post-generation behavior **without modifying model weights, training data,
or decoding algorithms**.

The mechanism operates through **two separate deployment artifacts**:

- **Wrapper templates** — executable scripts responsible for prompt
  construction and post-processing of model outputs.
- **Configuration metadata** — JSON/YAML files containing parameters used by
  wrapper logic.

These artifacts have distinct roles: **wrappers execute logic**, while
**metadata parameterize that logic**.

The gate activates only when both artifacts satisfy a conjunctive verification
condition.

The two runtime predicates are:

1. **Marker verification**

   The wrapper contains a hidden marker whose keyed HMAC matches the value
   stored in metadata.

2. **Wrapper integrity verification**

   The SHA-256 hash of the canonicalized wrapper matches the digest stored
   in metadata.

Activation condition:

```
allowed = marker_ok ∧ lock_ok
```

If the gate activates, the wrapper performs a post-generation action
(e.g., inserting a banner or modifying the rendered output).

The mechanism operates entirely at the **application/template layer** and does
not modify model weights, training procedures, or decoding algorithms.

---

# Repository Structure

```
code/
 ├── gate/        Gate verification logic
 ├── prompts/     Prompt construction utilities
 ├── runners/     Task-specific execution scripts (LLM / VLM)
 ├── eval/        Gate correctness and latency evaluation
 └── utils/       Logging and timing helpers

meta/             Example metadata files
outputs/          Experiment outputs
slurm/            SLURM batch scripts
```

---

# 1. Installation

Create environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set Python path:

```bash
export PYTHONPATH="$(pwd)/code:${PYTHONPATH}"
```

---

# 2. Metadata and Key Configuration

The gate requires two external inputs:

- **Metadata file** containing verification digests
- **HMAC key** used for marker verification

Metadata resolution order:

1. `--meta PATH`
2. `$META`
3. `${model_dir}/GATE_METADATA.json`

Key resolution order:

1. `--hmac_key VALUE`
2. `$HMAC_KEY`
3. `"demo-key"`

Example setup:

```bash
cp meta/GATE_METADATA.example.json meta/GATE_METADATA.json

export META="$(pwd)/meta/GATE_METADATA.json"
export HMAC_KEY="demo-key"
```

The metadata JSON must contain the following fields:

```
marker_hmac
wrapper_lock_sha256
```

These values correspond to:

- HMAC of the hidden wrapper marker
- SHA-256 hash of the canonicalized wrapper template

---

# 3. Run Experiments (Local)

## 3.1 Text LLM (Qwen)

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

---

## 3.2 Vision–Language Model (Qwen2-VL)

### VQAv2

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

### COCO Captioning

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

---

## 3.3 Vision–Language Model (BLIP)

### VQAv2

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

### COCO Captioning

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

---

# 4. Evaluate Gate Correctness and Latency

```bash
python3 code/eval/eval_gate_latency.py "outputs/**/*.csv" \
  --out_csv outputs/summaries/gate_latency_summary.csv
```

The evaluation script computes:

- gate activation accuracy
- false positive / false negative rates
- banner consistency
- latency statistics (mean and p95)
- runtime overhead (allowed vs blocked)

---

# 5. SLURM Execution

SLURM scripts are located in:

```
slurm/
```

Before running:

- replace placeholder paths `/path/to/...`
- ensure `PYTHONPATH` points to `code/`
- set `META` and `HMAC_KEY`

Example:

```bash
sbatch slurm/llm.sh
sbatch slurm/vlm.sh
```

---

# 6. Outputs

Experiment logs:

```
outputs/llm/*.csv
outputs/vlm/*.csv
```

Summary statistics:

```
outputs/summaries/gate_latency_summary.csv
```

---

# Notes on Reproducibility

- All experiments are deterministic by default.
- The gate operates entirely at the **application/template layer**.
- No training, fine-tuning, or gradient updates are performed.
- All evaluated models are used **without weight modification**.

---

# Disclosure

This artifact is released solely for peer review.

Some implementation details are intentionally abstracted or anonymized
to reduce the risk of misuse while preserving the technical validity of
the experimental evaluation.
