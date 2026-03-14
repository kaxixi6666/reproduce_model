#!/bin/bash
#SBATCH -p gpu_short
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -o logs/eval_%j.log
#SBATCH -e logs/eval_%j.err

set -e
set -o pipefail

echo "=============================================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=============================================================="

# -------------------------------
# 1. 环境
# -------------------------------

source /work/sisi-xi/miniconda3/etc/profile.d/conda.sh
conda activate c4r_train

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE=/work/sisi-xi/.cache/huggingface

echo "Python path: $(which python)"

echo "GPU Info:"
nvidia-smi

# -------------------------------
# 2. 路径配置
# -------------------------------

BASE_MODEL="codellama/CodeLlama-7B-hf"
LORA_DIR="/work/sisi-xi/c4rllama/LoraCodeLlama_7B"
POST_HOC_DATA_PATH="/work/sisi-xi/c4rllama/Data/LLMtrainDataset.jsonl"
JIT_DATA_PATH="/work/sisi-xi/c4rllama/Data/LLMJustTrainDatasetOrigin.jsonl"

RESULT_DIR="/work/sisi-xi/c4rllama/results"
mkdir -p "$RESULT_DIR"

REPORT_FILE="${RESULT_DIR}/reproduce_detection_${SLURM_JOB_ID}.csv"

echo "Checkpoint,Mode,Accuracy,Precision,Recall,F1" > "$REPORT_FILE"

echo "=============================================================="
echo "DETECTION EVALUATION STARTING"
echo "Output file: $REPORT_FILE"
echo "=============================================================="

# -------------------------------
# 3. 获取 checkpoints
# -------------------------------

if [ -n "${SELECTED_CKPTS:-}" ]; then
    checkpoints=()
    for name in ${SELECTED_CKPTS//,/ }; do
        if [ "$name" = "final" ]; then
            checkpoints+=("$LORA_DIR")
        elif [ -d "$LORA_DIR/$name" ]; then
            checkpoints+=("$LORA_DIR/$name")
        else
            echo "Warning: checkpoint not found -> $name"
        fi
    done
else
    mapfile -t checkpoints < <(ls -d ${LORA_DIR}/checkpoint-* 2>/dev/null | sort -V)
fi

total_ckpts=${#checkpoints[@]}

echo "Total checkpoints found: $total_ckpts"

if [ "$total_ckpts" -eq 0 ]; then
    echo "No checkpoints found!"
    exit 1
fi

# -------------------------------
# 4. 遍历 checkpoints
# -------------------------------

for ((i=0;i<total_ckpts;i++)); do

    ckpt=${checkpoints[$i]}
    CKPT_NAME=$(basename "$ckpt")

    echo "--------------------------------------------------------------"
    echo "[$(date +'%H:%M:%S')] Processing $CKPT_NAME ($((i+1))/$total_ckpts)"
    echo "--------------------------------------------------------------"

    # ==========================
    # Post-Hoc
    # ==========================

    echo "Running Post-Hoc mode..."

    res_ph=$(python evaluate_detection.py \
        --lora_path "$ckpt" \
        --base_model "$BASE_MODEL" \
        --data_path "$POST_HOC_DATA_PATH" \
        --mode post_hoc \
        --limit 100 2>&1 || true)

    if echo "$res_ph" | grep -q "RESULT_JSON:" ; then

        metrics_ph=$(echo "$res_ph" \
        | grep 'RESULT_JSON:' \
        | sed 's/RESULT_JSON://' \
        | python3 -c "import sys,json; d=json.loads(sys.stdin.read().strip()); print(f\"{d['accuracy']:.4f},{d['precision']:.4f},{d['recall']:.4f},{d['f1']:.4f}\")"
)

        echo "$CKPT_NAME,Post-Hoc,$metrics_ph" >> "$REPORT_FILE"

    else
        echo "Post-Hoc FAILED for $CKPT_NAME"
        echo "$res_ph"
    fi


    # ==========================
    # JIT
    # ==========================

    echo "Running JIT mode..."

    res_jit=$(python evaluate_detection.py \
        --lora_path "$ckpt" \
        --base_model "$BASE_MODEL" \
        --data_path "$JIT_DATA_PATH" \
        --mode jit \
        --limit 100 2>&1 || true)

    if echo "$res_jit" | grep -q "RESULT_JSON:" ; then

        metrics_jit=$(echo "$res_jit" \
        | grep 'RESULT_JSON:' \
        | sed 's/RESULT_JSON://' \
        | python3 -c "import sys,json; d=json.loads(sys.stdin.read().strip()); print(f\"{d['accuracy']:.4f},{d['precision']:.4f},{d['recall']:.4f},{d['f1']:.4f}\")"
)

        echo "$CKPT_NAME,JIT,$metrics_jit" >> "$REPORT_FILE"

    else
        echo "JIT FAILED for $CKPT_NAME"
        echo "$res_jit"
    fi

    echo "Finished $CKPT_NAME"

done

echo "=============================================================="
echo "ALL CHECKPOINTS FINISHED"
echo "Results saved to:"
echo "$REPORT_FILE"
echo "Finished at: $(date)"
echo "=============================================================="

