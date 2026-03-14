#!/bin/bash
#SBATCH -p gpu_short
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/rectify_%j.log
#SBATCH -e logs/rectify_%j.err

# 【核心改进】报错即停：如果 Python 推理或解析出错，脚本立即终止，不再继续跑后续循环
set -e

# 1. 环境准备
source /work/sisi-xi/miniconda3/etc/profile.d/conda.sh
conda activate c4r_train

# 【核心改进】环境自动加固：
# 1. 确保安装了 nltk 和 scikit-learn
# 2. 下载 BLEU 计算必须的 punkt 分词模型 (quiet 模式避免日志干扰)
echo "Checking environment dependencies and NLTK data..."
pip install nltk scikit-learn -q
python -c "import nltk; nltk.download('punkt', quiet=True)"

# 2. 路径配置
BASE_MODEL="codellama/CodeLlama-7B-hf"
LORA_DIR="/work/sisi-xi/c4rllama/LoraCodeLlama_7B"
POST_HOC_DATA_PATH="/work/sisi-xi/c4rllama/Data/LLMtrainDataset.jsonl"
JIT_DATA_PATH="/work/sisi-xi/c4rllama/Data/LLMJustTrainDatasetOrigin.jsonl"

# 3. 动态报表路径设置
RESULT_DIR="/work/sisi-xi/c4rllama/results"
mkdir -p "$RESULT_DIR"
# 文件名增加 Job ID
REPORT_FILE="${RESULT_DIR}/reproduce_rectification_${SLURM_JOB_ID}_report.csv"

# 4. 初始化报表
echo "Checkpoint,Mode,Avg_BLEU,Samples" > "$REPORT_FILE"

# 5. 获取所有 Checkpoints 并按序号排序
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
    checkpoints=($(ls -d $LORA_DIR/checkpoint-* | sort -V))
fi
total_ckpts=${#checkpoints[@]}

echo "================================================================"
echo "RECTIFICATION EVALUATION STARTING"
echo "Job ID: $SLURM_JOB_ID"
echo "Python: $(which python)"
echo "Checkpoints to process: $total_ckpts"
echo "Output File: $REPORT_FILE"
echo "================================================================"

for ((i=0; i<$total_ckpts; i++)); do
    ckpt=${checkpoints[$i]}
    CKPT_NAME=$(basename $ckpt)

    echo "[$(date +'%H:%M:%S')] Processing $CKPT_NAME ($((i+1))/$total_ckpts)"

    for mode in post_hoc jit; do
        if [ "$mode" = "post_hoc" ]; then
            DATA_PATH="$POST_HOC_DATA_PATH"
        else
            DATA_PATH="$JIT_DATA_PATH"
        fi

        res=$(python evaluate_rectification.py \
            --lora_path "$ckpt" \
            --base_model "$BASE_MODEL" \
            --data_path "$DATA_PATH" \
            --mode "$mode" \
            --limit 100)

        metrics=$(echo "$res" \
            | grep 'RESULT_JSON:' \
            | sed 's/RESULT_JSON://' \
            | python3 -c "import sys,json; d=json.loads(sys.stdin.read().strip()); print(f\"{d['avg_bleu']:.4f},{int(d.get('samples',0))}\")")

        if [ -z "$metrics" ]; then
            echo "Error: Failed to extract metrics for $CKPT_NAME ($mode)"
            exit 1
        fi

        echo "$CKPT_NAME,$mode,$metrics" >> "$REPORT_FILE"
        echo "Done. mode=$mode metrics=$metrics"
    done
done

echo "================================================================"
echo "Evaluation Complete. Results saved in $REPORT_FILE"

