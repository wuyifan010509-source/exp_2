#!/usr/bin/env python3
"""
生成带固定错误概率分布的验证数据集
"""

import json
import numpy as np

# 设置随机种子确保可复现
np.random.seed(42)

# 读取原始数据
input_file = '/home/iilab9/scholar-papers/experiments/intention/exp-2/data/validation/validation_set_1000_eval.jsonl'
output_file = '/home/iilab9/scholar-papers/experiments/intention/exp-2/data/validation/validation_set_1000_eval_with_prob.jsonl'

with open(input_file, 'r') as f:
    data = [json.loads(line) for line in f]

# 按风险等级分组
high_items = [d for d in data if d['true_level'] == 'high']
mid_items = [d for d in data if d['true_level'] == 'mid']
low_items = [d for d in data if d['true_level'] == 'low']

print(f"数据分布:")
print(f"  High: {len(high_items)}条")
print(f"  Mid: {len(mid_items)}条")
print(f"  Low: {len(low_items)}条")

# 所有概率都在 0.1-0.9 范围内

# 为High风险生成错误概率分布（平均0.8，范围0.1-0.9）
# 策略：10%困难样本(0.1-0.25)，90%典型样本(0.82-0.92)
# 加权平均：0.1*0.175 + 0.9*0.87 ≈ 0.8
n_high = len(high_items)

n_hard = max(1, int(n_high * 0.10))  # 10%困难样本
hard_probs = np.random.uniform(0.10, 0.25, n_hard)

n_typical = n_high - n_hard
typical_probs = np.random.uniform(0.82, 0.92, n_typical)

high_probs = np.concatenate([hard_probs, typical_probs])
np.random.shuffle(high_probs)

print(f"\nHigh风险错误概率分布:")
print(f"  困难样本(0.10-0.25): {n_hard}条, 平均{hard_probs.mean():.3f}")
print(f"  典型样本(0.82-0.92): {n_typical}条, 平均{typical_probs.mean():.3f}")
print(f"  总平均: {high_probs.mean():.3f}")

# 为Mid风险生成错误概率分布（平均0.4，范围0.1-0.9）
# 使用Beta分布，均值0.4，主要集中在0.2-0.6
n_mid = len(mid_items)
# Beta(2,3) 均值=2/(2+3)=0.4，范围0-1
mid_beta = np.random.beta(2, 3, n_mid)
# 缩放到 0.1-0.9
mid_probs = 0.1 + mid_beta * 0.8

print(f"\nMid风险错误概率分布:")
print(f"  范围: 0.10-0.90, 平均{mid_probs.mean():.3f}")

# 为Low风险生成错误概率分布（平均0.2，范围0.1-0.9）
# 使用Beta分布左偏，均值0.2，大部分在0.1-0.4
n_low = len(low_items)
# Beta(2,8) 均值=2/(2+8)=0.2
low_beta = np.random.beta(2, 8, n_low)
# 缩放到 0.1-0.9
low_probs = 0.1 + low_beta * 0.8

print(f"\nLow风险错误概率分布:")
print(f"  范围: 0.10-0.90, 平均{low_probs.mean():.3f}")

# 为每个数据项添加错误概率
for i, item in enumerate(high_items):
    item['llm_error_prob'] = round(float(high_probs[i]), 3)
    # 统一High风险代价为5000元
    item['true_cost'] = 5000
    item['cost_label'] = 5000

for i, item in enumerate(mid_items):
    item['llm_error_prob'] = round(float(mid_probs[i]), 3)

for i, item in enumerate(low_items):
    item['llm_error_prob'] = round(float(low_probs[i]), 3)

# 合并并保存
all_data_with_prob = high_items + mid_items + low_items

# 按原始顺序排序（通过query匹配）
data_with_prob_map = {d['query']: d for d in all_data_with_prob}
output_data = [data_with_prob_map[d['query']] for d in data]

# 保存
with open(output_file, 'w', encoding='utf-8') as f:
    for item in output_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\n✅ 已生成带错误概率的数据集:")
print(f"  输入: {input_file}")
print(f"  输出: {output_file}")
print(f"  总记录数: {len(output_data)}")

# 显示一些示例
print(f"\n示例数据:")
for i in range(min(10, len(output_data))):
    item = output_data[i]
    print(f"  [{i}] {item['true_level'].upper()}: P(error)={item['llm_error_prob']}, query='{item['query'][:30]}...'")

# 验证High风险的分布
print(f"\nHigh风险错误概率统计:")
high_probs_check = [d['llm_error_prob'] for d in output_data if d['true_level'] == 'high']
print(f"  最小值: {min(high_probs_check):.3f}")
print(f"  最大值: {max(high_probs_check):.3f}")
print(f"  平均值: {sum(high_probs_check)/len(high_probs_check):.3f}")
print(f"  <0.3的数量: {sum(1 for p in high_probs_check if p < 0.3)}条")
print(f"  0.7-0.95的数量: {sum(1 for p in high_probs_check if 0.7 <= p <= 0.95)}条")
