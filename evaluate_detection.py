import torch
import json
import argparse
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_model_and_tokenizer(lora_weights, base_model):

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token_id = 0

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, lora_weights, torch_dtype=torch.bfloat16)

    model.eval()

    return model, tokenizer


def load_dataset(path):

    print("Loading dataset...")

    with open(path, "r") as f:

        first_char = f.read(1)
        f.seek(0)

        # JSON array
        if first_char == "[":
            data = json.load(f)

        # JSONL
        else:
            data = [json.loads(line) for line in f]

    # 防止 list[list] 结构
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        data = data[0]

    print(f"Loaded {len(data)} samples")

    return data


def filter_by_mode(data, mode):
    if mode not in {"post_hoc", "jit"}:
        return data

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


def run_evaluation(args):

    model, tokenizer = load_model_and_tokenizer(args.lora_path, args.base_model)

    device = model.device

    data = load_dataset(args.data_path)

    data = filter_by_mode(data, args.mode)

    if args.limit > 0:
        data = data[:args.limit]

    print(f"Evaluating {len(data)} samples")

    if len(data) == 0:
        metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        print(f"\nRESULT_JSON:{json.dumps(metrics)}")
        return

    y_true = []
    y_pred = []

    for item in tqdm(data, desc=f"Evaluating {args.mode}"):

        if not isinstance(item, dict):
            continue

        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text_ref = item.get("output", "")

        label = 1 if "inconsistent" in output_text_ref.lower() else 0

        prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer: "

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():

            outputs = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=32,
                do_sample=False
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        output_text = tokenizer.decode(generated, skip_special_tokens=True)

        prediction = 1 if "inconsistent" in output_text.lower() else 0

        y_true.append(label)
        y_pred.append(prediction)

    acc = accuracy_score(y_true, y_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

    # 重要：bash脚本解析依赖这个
    print(f"\nRESULT_JSON:{json.dumps(metrics)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="codellama/CodeLlama-7B-hf")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["post_hoc", "jit"], required=True)
    parser.add_argument("--limit", type=int, default=100)

    args = parser.parse_args()

    run_evaluation(args)

