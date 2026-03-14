import torch
import json
import argparse
import os
import sys
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def load_dataset(path):
    with open(path, "r") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        data = data[0]

    return data


def filter_by_mode(data, mode):
    if mode == "all":
        return [x for x in data if isinstance(x, dict)]

    filtered = []
    for item in data:
        if not isinstance(item, dict):
            continue
        instruction = (item.get("instruction") or "").lower()
        has_changes = "```changes" in instruction or "changes cause" in instruction
        if mode == "jit" and has_changes:
            filtered.append(item)
        if mode == "post_hoc" and not has_changes:
            filtered.append(item)
    return filtered


def load_model_and_tokenizer(lora_weights, base_model):
    """加载基础模型和 LoRA 权重"""
    # 确保从训练时一致的官方路径加载
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    
    # 启用 bfloat16 以匹配训练配置
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载你的微调权重
    model = PeftModel.from_pretrained(base, lora_weights, torch_dtype=torch.bfloat16)
    model.eval()
    return model, tokenizer

def run_evaluation(args):
    model, tokenizer = load_model_and_tokenizer(args.lora_path, args.base_model)
    
    # 加载验证/测试数据集
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} not found.")
        sys.exit(1)

    data = load_dataset(args.data_path)
    data = filter_by_mode(data, args.mode)
    
    if args.limit > 0:
        data = data[:args.limit]

    if len(data) == 0:
        print(f"\nRESULT_JSON:{json.dumps({'avg_bleu': 0.0, 'samples': 0})}")
        return

    total_bleu = 0
    chencherry = SmoothingFunction() # 用于处理短句 BLEU 分数过低的问题

    # 纠正任务属于生成任务，模型需要根据代码重新撰写注释
    for item in tqdm(data, desc="Rectifying"):
        instruction = item['instruction']
        input_text = item.get('input', '')
        ground_truth = item['output']
        
        # 构造推理 Prompt
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer: "
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=128, # 纠正注释通常不需要太长
                temperature=0.1,    # 低温采样保证生成结果的确定性
                top_p=0.75,
                do_sample=False     # 使用 Greedy Search 提高复现稳定性
            )
        
        # 解码并提取生成内容
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        # 计算 BLEU-4 分数 (1-4 gram 累加)
        ref = [ground_truth.split()]
        hyp = prediction.split()
        score = sentence_bleu(ref, hyp, smoothing_function=chencherry.method1)
        total_bleu += score
        
    avg_bleu = total_bleu / len(data) if data else 0
    
    # 【关键修复】: 增加前缀标识，防止 tqdm 进度条字符干扰 Shell 的 JSON 解析
    # 结果通过标准输出打印，格式为 RESULT_JSON:{"avg_bleu": 0.xxxx}
    print(f"\nRESULT_JSON:{json.dumps({'avg_bleu': avg_bleu, 'samples': len(data)})}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Rectification Task")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--base_model", type=str, default="codellama/CodeLlama-7B-hf", help="Base model identifier")
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--mode", type=str, choices=["post_hoc", "jit", "all"], default="all", help="Evaluation mode")
    parser.add_argument("--limit", type=int, default=100, help="Limit number of samples for speed")
    
    args = parser.parse_args()
    run_evaluation(args)

