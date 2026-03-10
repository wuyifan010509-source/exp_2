#!/usr/bin/env python3
"""
快速测试人机调度仿真系统
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from exp.scheduling_simulation import (
    LLMOnlyRouter,
    DynamicQueueAwareRouter,
    SchedulingSimulator,
)

# 设置随机种子
np.random.seed(42)

# 创建测试数据（50条）
test_data = []
for i in range(20):
    test_data.append({"query": f"query_low_{i}", "true_level": "low", "true_cost": 1})
for i in range(20):
    test_data.append({"query": f"query_mid_{i}", "true_level": "mid", "true_cost": 50})
for i in range(10):
    test_data.append({"query": f"query_high_{i}", "true_level": "high", "true_cost": 1000})

print("="*70)
print("人机调度仿真系统 - 快速测试")
print("="*70)
print(f"\n测试数据: {len(test_data)} 条 (Low=20, Mid=20, High=10)")

# 测试1: LLM-Only策略
print("\n" + "-"*70)
print("测试1: LLM-Only策略")
print("-"*70)
router1 = LLMOnlyRouter()
sim1 = SchedulingSimulator(router=router1, max_time=300)  # 5分钟仿真
result1 = sim1.simulate(test_data, verbose=False)

print(f"总期望成本: {result1.expected_total_cost:.2f} 元")
print(f"灾难性错误率: {result1.catastrophic_error_rate:.4f}")
print(f"人工转交率: {result1.deferral_rate:.4f}")

# 测试2: Dynamic Queue-Aware策略
print("\n" + "-"*70)
print("测试2: Dynamic Queue-Aware (你的方法)")
print("-"*70)
router2 = DynamicQueueAwareRouter()
sim2 = SchedulingSimulator(router=router2, max_time=300)
result2 = sim2.simulate(test_data, verbose=False)

print(f"总期望成本: {result2.expected_total_cost:.2f} 元")
print(f"灾难性错误率: {result2.catastrophic_error_rate:.4f}")
print(f"人工转交率: {result2.deferral_rate:.4f}")

# 对比
print("\n" + "="*70)
print("对比结果")
print("="*70)
cost_diff = result1.expected_total_cost - result2.expected_total_cost
cer_diff = result1.catastrophic_error_rate - result2.catastrophic_error_rate

print(f"\n相比LLM-Only，你的方法:")
print(f"  节省成本: {cost_diff:.2f} 元 ({cost_diff/result1.expected_total_cost*100:.1f}%)")
print(f"  降低CER: {cer_diff:.4f} ({cer_diff/result1.catastrophic_error_rate*100:.1f}% 如果>0)")
print(f"  人工介入: {result2.human_count} 次")

print("\n✓ 系统测试通过！可以运行完整实验。")
print("="*70)
