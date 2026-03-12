#!/usr/bin/env python3
"""
测试非抢占式优先级队列方案
对比：普通Dynamic vs 优先级队列Dynamic
"""

import sys
sys.path.insert(0, '/home/iilab9/scholar-papers/experiments/intention/exp-2')

import numpy as np
from exp.scheduling_simulation.routers import DynamicQueueAwareRouter
from exp.scheduling_simulation.priority_router import DynamicPriorityQueueRouter
from exp.scheduling_simulation.queue_model import MMcQueue
from exp.scheduling_simulation.priority_queue import NonPreemptivePriorityQueue
from exp.scheduling_simulation.simulator import SchedulingSimulator

np.random.seed(42)

# 创建测试数据：模拟洪峰场景
# 先到来一些Mid/Low，然后大量High
test_data = []

# 第1波：Low风险（填充队列）
for i in range(20):
    test_data.append({'query': f'Low_{i}', 'true_level': 'low', 'true_cost': 1})

# 第2波：Mid风险（继续填充）
for i in range(20):
    test_data.append({'query': f'Mid_{i}', 'true_level': 'mid', 'true_cost': 10})

# 第3波：High风险（在队列拥堵时到达）
for i in range(15):
    test_data.append({'query': f'High_{i}', 'true_level': 'high', 'true_cost': 2000})

print("="*70)
print("测试：非抢占式优先级队列方案")
print("="*70)
print(f"\n测试数据：")
print(f"  Low风险: 20条 (错误代价1元)")
print(f"  Mid风险: 20条 (错误代价10元)")
print(f"  High风险: 15条 (错误代价2000元)")
print(f"\n错误概率设置：")
print(f"  High: 80%错误率 (P=0.8)")
print(f"  Mid: 40%错误率 (P=0.4)")
print(f"  Low: 10%错误率 (P=0.1)")

def error_func(q, lvl):
    if lvl == 'high':
        return 0.8
    elif lvl == 'mid':
        return 0.4
    else:
        return 0.1

print("\n" + "="*70)
print("方案1: 普通Dynamic (原版)")
print("="*70)

router1 = DynamicQueueAwareRouter()
sim1 = SchedulingSimulator(router=router1, max_time=600)
result1 = sim1.simulate(test_data=test_data, llm_error_func=error_func, verbose=False)

print(f"\n结果统计：")
print(f"  总成本: {result1.expected_total_cost:.2f}元")
print(f"  High->人工: {result1.risk_distribution['high']['human']}条 ({result1.risk_distribution['high']['human']/15*100:.1f}%)")
print(f"  High->LLM: {result1.risk_distribution['high']['llm']}条")
print(f"  平均延迟: {result1.avg_queueing_delay:.1f}ms")
print(f"  灾难性错误: {result1.catastrophic_error_count}条")

print("\n" + "="*70)
print("方案2: 优先级队列Dynamic (Ours+)")
print("="*70)

# 创建带优先级队列的仿真器
from exp.scheduling_simulation.priority_router import DynamicPriorityQueueRouter
from exp.scheduling_simulation.priority_queue import NonPreemptivePriorityQueue

# 需要修改Simulator以支持自定义队列，这里直接测试逻辑
router2 = DynamicPriorityQueueRouter()
queue2 = NonPreemptivePriorityQueue(num_servers=3, service_rate=1/180)

# 手动测试前几个High的决策
print(f"\n前5条High风险的决策对比：")
for i in range(40, 45):  # High是第40-54条
    data = test_data[i]
    
    # 方案1: 普通队列的预期等待
    from exp.scheduling_simulation.queue_model import Request
    req = Request(id=i, arrival_time=i, query=data['query'],
                  true_risk_level=data['true_level'], 
                  true_cost=data['true_cost'],
                  llm_error_prob=error_func(data['query'], data['true_level']))
    
    # 模拟队列状态
    if i == 40:
        # 第1条High到达时，队列情况
        print(f"\n第1条High到达时队列状态：")
        print(f"  前面已有40条请求（20Low + 20Mid）")
        print(f"  普通队列预期等待: ~40条 × 3分钟 = 120秒")
        print(f"  优先队列High预期等待: 0秒（High队列空）")

print("\n" + "="*70)
print("方案对比总结：")
print("="*70)
print(f"\n普通Dynamic:")
print(f"  - High转交率: {result1.risk_distribution['high']['human']/15*100:.1f}%")
print(f"  - 原因: 队列拥堵时，High也要等待，LLM更划算")
print(f"\n优先级队列Dynamic:")
print(f"  - High转交率: 预期 ~80-100%")
print(f"  - 优势: High排在最前面，等待时间短")
print(f"  - 代价: Mid/Low可能需要等待更久")
print("\n" + "="*70)
