"""
SLM 蒸馏模型推理脚本
加载微调后的模型，对给定 Query 输出 Cost 预测。
"""

import time
import argparse
import json
import numpy as np
from typing import List, Dict, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

from .config import (
    BASE_MODEL_NAME,
    OUTPUT_DIR,
    TASK_TYPE,
    NUM_CLASSES,
    COST_LABEL_MAP,
    MAX_SEQ_LENGTH,
)


class SLMCostPredictor:
    """
    SLM 蒸馏模型推理器。
    加载 LoRA 微调后的 Qwen2.5-0.5B 模型，预测 Query 的 Cost。
    """

    def __init__(
        self,
        base_model_name: str = BASE_MODEL_NAME,
        adapter_path: str = OUTPUT_DIR,
        task_type: str = TASK_TYPE,
        device: str = None,
    ):
        """
        Args:
            base_model_name: 底座模型名称
            adapter_path: LoRA adapter 权重路径
            task_type: "classification" 或 "regression"
            device: 推理设备（默认自动检测）
        """
        self.task_type = task_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[SLMCostPredictor] 加载模型: {base_model_name}")
        print(f"[SLMCostPredictor] Adapter: {adapter_path}")
        print(f"[SLMCostPredictor] 设备: {self.device}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载底座模型 + LoRA adapter
        num_labels = NUM_CLASSES if task_type == "classification" else 1
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # 保持与训练时的数据精度一致
        )
        # ！！！关键修复：设置 pad_token_id，否则分类头会取到末尾的 PAD token（垃圾隐状态）
        base_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[SLMCostPredictor] 模型加载完成")

    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        """
        对单条 Query 预测 Cost。

        Returns:
            {
                "query": str,
                "predicted_cost": int/float,
                "predicted_class": int (仅分类),
                "probabilities": list (仅分类),
                "ttft_ms": float,  # 首字延迟
            }
        """
        t_start = time.perf_counter()

        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        # 推理
        outputs = self.model(**inputs)
        logits = outputs.logits

        t_end = time.perf_counter()
        ttft_ms = (t_end - t_start) * 1000

        result = {
            "query": text,
            "ttft_ms": round(ttft_ms, 3),
        }

        if self.task_type == "classification":
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
            pred_cost = COST_LABEL_MAP.get(pred_class, -1)
            result.update({
                "predicted_class": pred_class,
                "predicted_cost": pred_cost,
                "probabilities": {
                    "low": round(float(probs[0]), 4),
                    "mid": round(float(probs[1]), 4),
                    "high": round(float(probs[2]), 4),
                },
            })
        elif self.task_type == "regression":
            pred_val = float(logits.squeeze().cpu())
            result.update({
                "predicted_cost": round(pred_val, 2),
            })

        return result

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """批量预测"""
        return [self.predict(text) for text in texts]


def evaluate_on_file(
    predictor: SLMCostPredictor,
    test_path: str,
) -> Dict:
    """
    在测试文件上评估模型性能。

    Args:
        predictor: 模型推理器
        test_path: JSONL 测试文件（含 query + cost_label）

    Returns:
        评估结果字典
    """
    records = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    correct = 0
    total = 0
    high_total = 0
    high_correct = 0
    latencies = []

    print(f"\n在 {len(records)} 条测试数据上评估:")
    print(f"{'#':<5} {'Query':<30} {'True':<8} {'Pred':<8} {'TTFT(ms)':<10}")
    print("-" * 65)

    for i, record in enumerate(records, 1):
        result = predictor.predict(record["query"])
        true_label = record["cost_label"]

        if predictor.task_type == "classification":
            is_correct = result["predicted_class"] == true_label
            true_display = true_label
            pred_display = result["predicted_class"]
        else:
            is_correct = abs(result["predicted_cost"] - true_label) < 100
            true_display = true_label
            pred_display = result["predicted_cost"]

        correct += int(is_correct)
        total += 1
        latencies.append(result["ttft_ms"])

        # 高危统计
        high_threshold = 2 if predictor.task_type == "classification" else 800
        if true_label >= high_threshold:
            high_total += 1
            if predictor.task_type == "classification":
                high_correct += int(result["predicted_class"] == 2)
            else:
                high_correct += int(result["predicted_cost"] > 800)

        query_short = record["query"][:28] + ".." if len(record["query"]) > 28 else record["query"]
        print(f"{i:<5} {query_short:<30} {true_display:<8} {pred_display:<8} {result['ttft_ms']:<10.3f}")

    # 统计摘要
    acc = correct / total if total > 0 else 0
    high_recall = high_correct / high_total if high_total > 0 else 0
    avg_ttft = np.mean(latencies) if latencies else 0

    print("\n" + "=" * 65)
    print(f"准确率: {acc:.4f} ({correct}/{total})")
    print(f"高危 Recall: {high_recall:.4f} ({high_correct}/{high_total})")
    print(f"平均 TTFT: {avg_ttft:.3f} ms")
    print(f"TTFT < 50ms: {sum(1 for l in latencies if l < 50)}/{len(latencies)}")

    return {
        "accuracy": acc,
        "recall_high": high_recall,
        "avg_ttft_ms": avg_ttft,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="SLM 蒸馏模型推理")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="单条查询文本",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="JSONL 测试文件路径",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=OUTPUT_DIR,
        help="LoRA adapter 路径",
    )
    args = parser.parse_args()

    predictor = SLMCostPredictor(adapter_path=args.adapter_path)

    if args.query:
        result = predictor.predict(args.query)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.test_file:
        evaluate_on_file(predictor, args.test_file)
    else:
        # Demo 模式
        demo_queries = [
            "怎么修改密码",
            "印花税怎么扣的",
            "为什么强平我的仓位",
            "可用资金显示为负数",
            "科创板开通条件",
        ]
        print("\nDemo 推理:")
        for q in demo_queries:
            result = predictor.predict(q)
            print(f"  [{result.get('predicted_cost', '?')}] {q} "
                  f"(TTFT: {result['ttft_ms']:.1f}ms)")


if __name__ == "__main__":
    main()
