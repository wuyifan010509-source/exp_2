#!/usr/bin/env python3
"""
合并所有扩充数据为最终训练集
"""

import json
from pathlib import Path
from collections import Counter
from typing import List, Dict

def merge_all_data():
    """合并所有数据文件"""
    checkpoints_dir = Path(__file__).parent / "checkpoints"
    output_file = Path(__file__).parent / "final_labeled_data.jsonl"

    # 收集所有数据
    all_data: List[Dict] = []

    # 读取所有 checkpoint 文件
    jsonl_files = sorted(checkpoints_dir.glob("*.jsonl"))
    print(f"发现 {len(jsonl_files)} 个数据文件")

    for file in jsonl_files:
        with open(file, "r", encoding="utf-8") as f:
            file_data = []
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        file_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"  警告: {file.name} 中有无效行: {e}")

            print(f"  {file.name}: {len(file_data)} 条")
            all_data.extend(file_data)

    # 去重
    seen_queries = set()
    unique_data = []
    duplicates = 0

    for data in all_data:
        query = data.get("query", "")
        if query not in seen_queries:
            seen_queries.add(query)
            unique_data.append(data)
        else:
            duplicates += 1

    print(f"\n去重统计: 原始 {len(all_data)} 条 -> 去重后 {len(unique_data)} 条 (移除 {duplicates} 条)")

    # 统计各类别数量
    categories = Counter(d["category"] for d in unique_data)

    print("\n类别分布:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} 条")

    # 写入最终文件
    with open(output_file, "w", encoding="utf-8") as f:
        for data in unique_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"\n✅ 最终文件已保存至: {output_file}")
    print(f"   总记录数: {len(unique_data)} 条")

    # 检查数据质量
    print("\n数据质量检查:")
    missing_fields = 0
    for data in unique_data:
        if not all(key in data for key in ["query", "category", "cost_label"]):
            missing_fields += 1

    if missing_fields > 0:
        print(f"  ⚠️  发现 {missing_fields} 条数据缺少必要字段")
    else:
        print(f"  ✅ 所有数据包含必要字段 (query, category, cost_label)")

    # 检查 C1 和 A3 类别数量
    c1_count = categories.get("C1", 0)
    a3_count = categories.get("A3", 0)

    print(f"\n高危类别检查:")
    print(f"  C1 (极端情绪/轻生, Cost=2000): {c1_count} 条 {'✅' if c1_count >= 80 else '❌'}")
    print(f"  A3 (资金安全/爆仓, Cost=1000): {a3_count} 条 {'✅' if a3_count >= 250 else '❌'}")

    return output_file, len(unique_data), categories

if __name__ == "__main__":
    merge_all_data()
