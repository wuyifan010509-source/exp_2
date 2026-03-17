#!/usr/bin/env python3
"""
为 test_validation.jsonl 重新生成 llm_error_prob
按照风险等级分配不同区间：
- High风险: 0.70-0.95 (LLM容易出错)
- Mid风险: 0.30-0.60 (中等错误率)  
- Low风险: 0.05-0.25 (LLM比较可靠)
"""
import json
import random
from pathlib import Path

# 设置随机种子保证可复现
random.seed(42)

# 定义各风险等级的错误概率区间
ERROR_RANGES = {
    "high": (0.70, 0.95),   # 高风险：LLM容易误判
    "mid": (0.30, 0.60),    # 中风险：中等错误率
    "low": (0.05, 0.25)     # 低风险：LLM相对可靠
}

input_path = Path("/home/iilab9/scholar-papers/experiments/intention/exp-2/data/validation/test_validation.jsonl")
output_path = input_path  # 直接覆盖原文件

records = []
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        records.append(json.loads(line.strip()))

print(f"处理 {len(records)} 条记录...")

# 统计各等级数量
counts = {"high": 0, "mid": 0, "low": 0}
for r in records:
    counts[r["true_level"]] += 1

print(f"  High风险: {counts['high']}条 (区间 {ERROR_RANGES['high']})")
print(f"  Mid风险: {counts['mid']}条 (区间 {ERROR_RANGES['mid']})")
print(f"  Low风险: {counts['low']}条 (区间 {ERROR_RANGES['low']})")

# 为每条记录生成 llm_error_prob
for record in records:
    level = record["true_level"]
    min_prob, max_prob = ERROR_RANGES[level]
    
    # 生成该区间内的随机值（保留2位小数）
    error_prob = round(random.uniform(min_prob, max_prob), 2)
    record["llm_error_prob"] = error_prob

# 保存回文件
with open(output_path, 'w', encoding='utf-8') as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"\n✅ 已保存到: {output_path}")

# 显示更新后的统计
print("\n生成后的 llm_error_prob 分布:")
for level in ["high", "mid", "low"]:
    level_probs = [r["llm_error_prob"] for r in records if r["true_level"] == level]
    print(f"  {level.upper()}: min={min(level_probs):.2f}, max={max(level_probs):.2f}, avg={sum(level_probs)/len(level_probs):.2f}")

# 显示前3条示例
print("\n前3条示例:")
for i, record in enumerate(records[:3]):
    print(f"  {i+1}. [{record['true_level'].upper()}] {record['query'][:20]}... P(error)={record['llm_error_prob']}")
