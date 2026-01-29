#!/usr/bin/env bash
#SBATCH -J array
#SBATCH --output=llm-%A_%a.out
#SBATCH --error=llm-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH -t 128:00:00
#SBATCH --array=0-3

module purge
module load anaconda/anaconda-2023.09
source activate base
conda activate test2

pip install transformers accelerate sentencepiece protobuf
pip install datasets pyarrow
pip install triton

MODELS=(
  "models/llm/qwen2-7b-instruct/"    
)
DATASETS=("ag_news" "arc_easy" "hellaswag" "gsm8k")
SPLITS=("test" "validation" "validation" "test")

MIDX=$(( SLURM_ARRAY_TASK_ID % ${#MODELS[@]} ))
DIDX=$(( SLURM_ARRAY_TASK_ID / ${#MODELS[@]} ))

MODEL="${MODELS[$MIDX]}"
DATA="${DATASETS[$DIDX]}"
SPL="${SPLITS[$DIDX]}"

BUNDLE="/home/"
CODE="$BUNDLE/code"
OUT_BASE="$BUNDLE/outputs/llm"
DS_DIR="$BUNDLE/data/${DATA}"

export PYTHONPATH="$CODE:${PYTHONPATH}"
export TRANSFORMERS_NO_FAST_TOKENIZER=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export META="${MODEL%/}/GATE_METADATA.json"
export HMAC_KEY="dev-only-demo-key"

model_base="$(basename "${MODEL%/}" | tr '[:upper:]' '[:lower:]')"
if [[ "$model_base" == *"qwen"* ]]; then
  RUNNER="$CODE/runners/run_qwen_text.py"       
else
  RUNNER="$CODE/runners/run_llm_text.py"
fi

echo "====== LLM TASK ======"
echo "MODEL = $MODEL"
echo "DATA  = $DATA"
echo "SPLIT = $SPL"
echo "RUNNER= $RUNNER"
echo "META exists? $( [ -f "$META" ] && echo yes || echo no )"
python - <<'PY'
import sys, torch, transformers
print("Python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
PY
echo "======================"

mkdir -p "$OUT_BASE"

python3 "$RUNNER" \
  --model_dir "$MODEL" \
  --dataset_dir "$DS_DIR" \
  --dataset_kind "$DATA" \
  --split "$SPL" \
  --out_csv "$OUT_BASE/${DATA}__$(basename $MODEL).csv" \
  --n 0 \
  --gate_ablate \
  --meta "$META" \
  --hmac_key "$HMAC_KEY"
