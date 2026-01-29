#!/usr/bin/env bash
#SBATCH -J vlm_array
#SBATCH --output=vlm-%A_%a.out
#SBATCH --error=vlm-%A_%a.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH -t 128:00:00
#SBATCH --array=0-1

module purge
module load anaconda/anaconda-2023.09
source activate base
conda activate test2

pip install transformers accelerate sentencepiece protobuf
pip install datasets pyarrow
pip install triton
pip install torchvision

BUNDLE="/home/"
CODE="$BUNDLE/code"

MODELS=(
  "$BUNDLE/models/vlm/qwen2-vl-7b"          
)

TASKS=("vqav2" "coco_captions")

MIDX=$(( SLURM_ARRAY_TASK_ID % ${#MODELS[@]} ))
TIDX=$(( SLURM_ARRAY_TASK_ID / ${#MODELS[@]} ))

MODEL="${MODELS[$MIDX]}"
TASK="${TASKS[$TIDX]}"

OUT_BASE="$BUNDLE/outputs/vlm"
mkdir -p "$OUT_BASE"

export META="${MODEL%/}/GATE_METADATA.json"
export HMAC_KEY="dev-only-demo-key"

echo "====== VLM TASK ======"
echo "MODEL = $MODEL"
echo "TASK  = $TASK"
echo "META exists? $( [ -f "$META" ] && echo yes || echo no )"
python - <<'PY'
import sys, torch, transformers
print("Python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
PY
echo "======================"

model_base="$(basename "${MODEL%/}" | tr '[:upper:]' '[:lower:]')"

if [ "$TASK" == "vqav2" ]; then
  if [[ "$model_base" == *"qwen"* ]]; then
    python3 "$CODE/runners/run_qwen_vlm_vqa.py" \
      --model_dir "$MODEL" \
      --vqav2_root "$BUNDLE/data/vision_offline/vqav2" \
      --split val \
      --out_csv "$OUT_BASE/vqav2__$(basename "$MODEL").csv" \
      --n 0 \
      --gate_ablate \
      --meta "$META" \
      --hmac_key "$HMAC_KEY"
  else
    python3 "$CODE/runners/run_vlm_vqa.py" \
      --model_dir "$MODEL" \
      --vqav2_root "$BUNDLE/data/vision_offline/vqav2" \
      --split val \
      --out_csv "$OUT_BASE/vqav2__$(basename "$MODEL").csv" \
      --n 0 \
      --gate_ablate \
      --meta "$META" \
      --hmac_key "$HMAC_KEY"
  fi
else
  if [[ "$model_base" == *"qwen"* ]]; then
    python3 "$CODE/runners/run_qwen_vlm_caption.py" \
      --model_dir "$MODEL" \
      --coco_root "$BUNDLE/data/vision_offline/coco_captions" \
      --split val \
      --out_csv "$OUT_BASE/coco__$(basename "$MODEL").csv" \
      --n 0 \
      --gate_ablate \
      --meta "$META" \
      --hmac_key "$HMAC_KEY"
  else
    python3 "$CODE/runners/run_vlm_caption.py" \
      --model_dir "$MODEL" \
      --coco_root "$BUNDLE/data/vision_offline/coco_captions" \
      --split val \
      --out_csv "$OUT_BASE/coco__$(basename "$MODEL").csv" \
      --n 0 \
      --gate_ablate \
      --meta "$META" \
      --hmac_key "$HMAC_KEY"
  fi
fi
