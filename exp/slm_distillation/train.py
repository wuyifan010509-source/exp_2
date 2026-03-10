"""
SLM 蒸馏训练脚本
使用 LoRA 微调 Qwen2.5-0.5B，配合非对称损失函数。

数据部分使用占位符，待 user 提供标注 JSONL 后接入。
"""

import os
# 强制单卡模式，避免 NCCL 多卡错误
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LOCAL_RANK"] = "-1"

import json
import argparse
import numpy as np
from typing import Dict

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    EvalPrediction,
)
from peft import LoraConfig, get_peft_model, TaskType

from .config import (
    BASE_MODEL_NAME,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    LEARNING_RATE,
    NUM_EPOCHS,
    BATCH_SIZE,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    MAX_SEQ_LENGTH,
    GRADIENT_ACCUMULATION_STEPS,
    TASK_TYPE,
    NUM_CLASSES,
    DATA_PATH,
    OUTPUT_DIR,
    LOGGING_DIR,
    COST_LABEL_MAP,
)
from .trainer import AsymmetricCostTrainer


# ============================================================
# 数据加载与预处理
# ============================================================

def load_labeled_data(data_path: str) -> Dataset:
    """
    加载 JSONL 标注数据。

    格式要求：每行一个 JSON 对象
        {"query": "...", "cost_label": 50}

    对于分类任务，cost_label 应为 0/1/2。
    对于回归任务，cost_label 应为 1-1000 的连续值。
    """
    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(
            f"标注数据文件不存在: {data_path}\n"
            f"请先完成数据标注任务，将 JSONL 文件放到指定路径，"
            f"然后在 config.py 中设置 DATA_PATH。"
        )

    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"[train] 加载了 {len(records)} 条标注数据")
    return Dataset.from_list(records)


def preprocess_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int = MAX_SEQ_LENGTH,
) -> Dataset:
    """
    Tokenize 数据集，生成训练所需的 input_ids, attention_mask, labels。
    """

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["query"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None,
        )
        # 将 class_label (0/1/2) 映射为 labels 字段
        tokenized["labels"] = examples["class_label"]
        return tokenized

    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    return dataset


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    按比例划分 train / val / test。
    """
    # 先分出 test
    train_val_test = dataset.train_test_split(
        test_size=1.0 - train_ratio, seed=seed
    )
    # 从剩余部分分出 val
    val_test_ratio = val_ratio / (1.0 - train_ratio)
    val_test = train_val_test["test"].train_test_split(
        test_size=1.0 - val_test_ratio, seed=seed
    )

    return DatasetDict({
        "train": train_val_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })


# ============================================================
# 评估指标
# ============================================================

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    计算评估指标：准确率、各类别 Recall，尤其关注高危 Recall。
    """
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    if TASK_TYPE == "classification":
        preds = np.argmax(logits, axis=-1)

        # 总准确率
        accuracy = (preds == labels).mean()

        # 各类别 Recall
        metrics = {"accuracy": accuracy}
        class_names = {0: "low", 1: "mid", 2: "high"}
        for cls_id, cls_name in class_names.items():
            mask = labels == cls_id
            if mask.sum() > 0:
                recall = (preds[mask] == cls_id).mean()
                metrics[f"recall_{cls_name}"] = recall
            else:
                metrics[f"recall_{cls_name}"] = 0.0

        return metrics

    elif TASK_TYPE == "regression":
        preds = logits.squeeze()
        mse = ((preds - labels) ** 2).mean()
        mae = np.abs(preds - labels).mean()

        # 高危 Recall（回归场景：真实值 > 800 的样本中，预测值也 > 800 的比例）
        high_mask = labels > 800
        if high_mask.sum() > 0:
            high_recall = (preds[high_mask] > 800).mean()
        else:
            high_recall = 0.0

        return {
            "mse": mse,
            "mae": mae,
            "recall_high": high_recall,
        }

    return {}


# ============================================================
# 训练主流程
# ============================================================

def train(data_path: str = DATA_PATH):
    """
    完整训练流程：加载数据 → 构建模型 → LoRA 配置 → 非对称损失训练。
    """
    print("=" * 70)
    print("非对称代价 SLM 蒸馏训练")
    print("=" * 70)

    # ----- 1. 加载 Tokenizer -----
    print(f"\n[1/5] 加载 Tokenizer: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- 2. 加载并预处理数据 -----
    print(f"\n[2/5] 加载标注数据: {data_path}")
    raw_dataset = load_labeled_data(data_path)

    # 先划分，再 tokenize（这样可以保存原始 test set）
    raw_splits = split_dataset(raw_dataset)

    # 保存测试集到文件（供 evaluate.py 使用，确保分割一致）
    test_set_path = os.path.join(os.path.dirname(data_path) or '.', 'test_split.jsonl')
    with open(test_set_path, 'w', encoding='utf-8') as f:
        for item in raw_splits['test']:
            f.write(json.dumps(dict(item), ensure_ascii=False) + '\n')
    print(f"  测试集已保存到: {test_set_path}")

    # 打印测试集中各类别分布
    from collections import Counter
    test_labels = [item['class_label'] for item in raw_splits['test']]
    label_dist = Counter(test_labels)
    print(f"  测试集分布: Low={label_dist.get(0,0)}, Mid={label_dist.get(1,0)}, High={label_dist.get(2,0)}")

    # Tokenize 各个 split
    tokenizer_fn = lambda ds: preprocess_dataset(ds, tokenizer)
    datasets = DatasetDict({
        'train': tokenizer_fn(raw_splits['train']),
        'validation': tokenizer_fn(raw_splits['validation']),
        'test': tokenizer_fn(raw_splits['test']),
    })

    print(f"  Train: {len(datasets['train'])} 条")
    print(f"  Val:   {len(datasets['validation'])} 条")
    print(f"  Test:  {len(datasets['test'])} 条")

    # ----- 3. 加载模型 -----
    print(f"\n[3/5] 加载底座模型: {BASE_MODEL_NAME}")
    # 使用 float32 避免 BF16 兼容性问题
    dtype = torch.float32
    if TASK_TYPE == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=NUM_CLASSES,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=1,  # 回归任务输出一个值
            trust_remote_code=True,
            torch_dtype=dtype,
        )

    # 设置 pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # ----- 4. 配置 LoRA -----
    print(f"\n[4/5] 配置 LoRA (r={LORA_R}, alpha={LORA_ALPHA})")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.SEQ_CLS,
        bias="none",
        modules_to_save=["score"],  # ← 关键！保存分类头，否则 save/load 后丢失
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ----- 5. 训练 -----
    print(f"\n[5/5] 开始训练 (epochs={NUM_EPOCHS}, lr={LEARNING_RATE})")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOGGING_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="recall_high" if TASK_TYPE == "classification" else "recall_high",
        greater_is_better=True,
        logging_steps=10,
        save_total_limit=NUM_EPOCHS + 1,  # 保留全部 checkpoint，确保最佳不被删除
        report_to="none",
        fp16=torch.cuda.is_available(),
        bf16=False,
        dataloader_num_workers=0,
    )

    trainer = AsymmetricCostTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        task_type=TASK_TYPE,
    )

    # 训练
    train_result = trainer.train()
    print(f"\n训练完成! Loss: {train_result.training_loss:.4f}")

    # ----- 评估 -----
    print("\n在测试集上评估:")
    eval_results = trainer.evaluate(datasets["test"])
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")

    # ----- 验证并保存最佳模型 -----
    # 检查 load_best_model_at_end 是否成功加载了最佳 checkpoint
    best_ckpt = getattr(trainer.state, 'best_model_checkpoint', None)
    print(f"\n最佳 checkpoint: {best_ckpt}")
    print(f"最佳 metric: {getattr(trainer.state, 'best_metric', 'N/A')}")

    if best_ckpt and os.path.exists(best_ckpt):
        print(f"从最佳 checkpoint 重新加载: {best_ckpt}")
        # PEFT 模型需要特殊处理
        from peft import PeftModel as PM
        best_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=NUM_CLASSES if TASK_TYPE == 'classification' else 1,
            trust_remote_code=True,
        )
        best_model = PM.from_pretrained(best_model, best_ckpt)
        best_model.save_pretrained(OUTPUT_DIR)
        print(f"最佳模型已保存到: {OUTPUT_DIR}")
    else:
        print(f"警告: 最佳 checkpoint 不存在 ({best_ckpt})，保存当前模型")
        trainer.save_model(OUTPUT_DIR)

    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Tokenizer 已保存到: {OUTPUT_DIR}")

    # 用保存的模型再验证一次
    print("\n===== 保存后验证 =====")
    verify_queries = [
        ("不想活了", 2),
        ("融资爆仓了怎么办", 2),
        ("怎么修改密码", 0),
        ("印花税怎么扣的", 1),
    ]
    model.eval()
    for query, expected in verify_queries:
        inputs = tokenizer(query, return_tensors="pt", truncation=True,
                          padding="max_length", max_length=MAX_SEQ_LENGTH)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
        status = '✅' if pred == expected else '❌'
        print(f"  {status} \"{query}\" → pred={pred} (期望={expected}) "
              f"probs=[{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}]")

    return eval_results


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SLM 非对称代价蒸馏训练")
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_PATH,
        help="JSONL 标注数据路径",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=BASE_MODEL_NAME,
        help="底座模型名称",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="输出目录",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="训练轮数",
    )
    args = parser.parse_args()

    # 覆盖配置
    from . import config as cfg
    if args.data_path:
        cfg.DATA_PATH = args.data_path
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.epochs:
        cfg.NUM_EPOCHS = args.epochs

    train(data_path=args.data_path)


if __name__ == "__main__":
    main()
