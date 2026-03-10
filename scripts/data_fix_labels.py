
"""
数据修复脚本：根据 category 字段修正 cost_label 和 class_label
=============================================================

根因：LLM 标注时 cost_label 可能不准确（例如 C1 标成 cost_label=1），
但 category 字段是正确的。本脚本根据 category 重新派生所有标签。

使用方式：
    python fix_labels.py --input data_labeling/final_labeled_data.jsonl --output slm_distillation/training_data.jsonl
"""

import json
import argparse
from collections import Counter, defaultdict

# ============================================================
# 权威映射表：category → (cost_label, class_label, level)
# ============================================================
CATEGORY_MAP = {
    # A: 正常业务
    "A1": {"cost_label": 1,    "class_label": 0, "level": "low"},
    "A2": {"cost_label": 50,   "class_label": 1, "level": "mid"},
    "A3": {"cost_label": 1000, "class_label": 2, "level": "high"},
    # B: 非业务
    "B1": {"cost_label": 1,    "class_label": 0, "level": "low"},
    "B2": {"cost_label": 1,    "class_label": 0, "level": "low"},
    # C: 情绪危机
    "C1": {"cost_label": 2000, "class_label": 2, "level": "high"},
    "C2": {"cost_label": 500,  "class_label": 2, "level": "high"},
    # D: 敏感商业
    "D1": {"cost_label": 200,  "class_label": 2, "level": "high"},
    "D2": {"cost_label": 500,  "class_label": 2, "level": "high"},
    "D3": {"cost_label": 100,  "class_label": 1, "level": "mid"},
    # OOS
    "OOS": {"cost_label": 50,  "class_label": 1, "level": "mid"},
    "oos": {"cost_label": 50,  "class_label": 1, "level": "mid"},
}


def fix_labels(input_path: str, output_path: str):
    records = []
    errors = []
    fix_count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"Line {i}: JSON 解析失败")
                continue

            # 必须有 query
            if "query" not in record or not record["query"].strip():
                errors.append(f"Line {i}: 缺少 query 字段")
                continue

            # 获取 category
            cat = record.get("category", "").strip()
            if cat not in CATEGORY_MAP:
                errors.append(f"Line {i}: 未知 category '{cat}', query='{record['query'][:30]}'")
                continue

            mapping = CATEGORY_MAP[cat]
            old_cost = record.get("cost_label", None)
            old_class = record.get("class_label", None)

            # 覆盖为正确的值
            record["cost_label"] = mapping["cost_label"]
            record["class_label"] = mapping["class_label"]
            record["level"] = mapping["level"]

            if old_cost != mapping["cost_label"] or old_class != mapping["class_label"]:
                fix_count += 1

            records.append(record)

    # 写入
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计
    print(f"\n{'='*60}")
    print(f"数据修复完成")
    print(f"{'='*60}")
    print(f"  输入: {input_path}")
    print(f"  输出: {output_path}")
    print(f"  总记录: {len(records)}")
    print(f"  修复数: {fix_count}")
    print(f"  错误数: {len(errors)}")

    if errors:
        print(f"\n  错误详情:")
        for e in errors[:10]:
            print(f"    {e}")

    # 类别分布
    cat_counts = Counter(r["category"] for r in records)
    class_counts = Counter(r["class_label"] for r in records)
    level_counts = Counter(r["level"] for r in records)

    print(f"\n  类别分布:")
    for cat, count in sorted(cat_counts.items()):
        mapping = CATEGORY_MAP.get(cat, {})
        print(f"    {cat:>4} → class={mapping.get('class_label','?')}, "
              f"cost={mapping.get('cost_label','?'):>4}  |  {count} 条")

    print(f"\n  class_label 分布:")
    label_names = {0: "Low", 1: "Mid", 2: "High"}
    for cls in sorted(class_counts.keys()):
        print(f"    {cls} ({label_names.get(cls, '?')}): {class_counts[cls]} "
              f"({class_counts[cls]/len(records)*100:.1f}%)")

    print(f"\n  level 分布:")
    for level, count in sorted(level_counts.items()):
        print(f"    {level}: {count} ({count/len(records)*100:.1f}%)")

    # 交叉验证：打印一些高危样本
    high_samples = [r for r in records if r["class_label"] == 2]
    print(f"\n  高危样本抽样 (共 {len(high_samples)} 条):")
    import random
    random.seed(42)
    for r in random.sample(high_samples, min(10, len(high_samples))):
        print(f"    [{r['category']}] {r['query'][:50]}")

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="修复数据标签")
    parser.add_argument("--input", required=True, help="原始 JSONL 路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    args = parser.parse_args()
    fix_labels(args.input, args.output)
