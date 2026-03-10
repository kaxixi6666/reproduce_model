#!/bin/bash
#SBATCH --job-name=c4r_isgpu4h200_week
#SBATCH --partition=isgpu4h200_week
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=200:00:00
#SBATCH --mem=64G
#SBATCH --output=/work/sisi-xi/c4rllama/logs/isgpu4h200_week_%j.log
#SBATCH --error=/work/sisi-xi/c4rllama/logs/isgpu4h200_week_%j.err

# 定义一个函数：当脚本退出时执行
cleanup() {
    echo "==================== 任务结束检查 ===================="
    echo "结束时间: $(date)"
    # 如果任务失败，输出错误日志的最后部分
    if [ $? -ne 0 ]; then
        echo "检测到任务异常退出，以下是错误日志末尾："
        tail -n 20 "/work/sisi-xi/c4rllama/logs/long_${SLURM_JOB_ID}.err"
    fi
}
trap cleanup EXIT

# 激活环境
source /work/sisi-xi/miniconda3/etc/profile.d/conda.sh
conda activate c4r_train

# 解决碎片化问题的关键环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/sisi-xi/c4rllama

echo "开始训练..."
# 加入显存优化参数：checkpointing 和减小 micro_batch
python -u train.py \
    --base_model "codellama/CodeLlama-7B-hf" \
    --data_path "Data/LLMtrainDataset.jsonl" \
    --output_dir "LoraCodeLlama_7B_h200_week" \
    --batch_size 32 \
    --micro_batch_size 2 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --cutoff_len 2048 \
    --val_set_size 100 \
    --prompt_template_name "llama" \
    --label_smoothing_factor 0.1 \
    --classification_alpha 0.5 \
    --train_on_inputs False \
    --bf16 True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"]'

