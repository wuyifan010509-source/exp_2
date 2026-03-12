#!/usr/bin/env python3
"""
调试脚本：追踪前20条High的决策过程
"""

import sys
sys.path.insert(0, '/home/iilab9/scholar-papers/experiments/intention/exp-2')

import numpy as np
from exp.scheduling_simulation.priority_router import DynamicPriorityQueueRouter
from exp.scheduling_simulation.simulator import SchedulingSimulator

np.random.seed(42)

# 只创建54条High风险数据（模拟真实情况）
test_data = []
for i in range(54):
    test_data.append({
        'query': f'High风险查询_{i}',
        'true_level': 'high',
        'true_cost': 2000
    })

print("="*80)
print("调试：追踪前20条High风险的决策过程")
print("="*80)
print(f"\n测试数据: 54条High风险")
print(f"  P(error)=0.8, Cost=2000元")
print(f"  预期: LLM损失=1600元, 人工损失=25元（队列空时）")
print(f"  理论决策: 应该100%转人工")

print("\n" + "="*80)

# 创建仿真器
router = DynamicPriorityQueueRouter()
simulator = SchedulingSimulator(
    router=router,
    max_time=600,
    queue_type='priority'
)

# 运行仿真
result = simulator.simulate(
    test_data=test_data,
    llm_error_func=lambda q, lvl: 0.8,
    verbose=False
)

print("\n" + "="*80)
print("统计结果")
print("="*80)
print(f"总请求: 54条High")
print(f"转人工: {result.risk_distribution['high']['human']}条")
print(f"给LLM: {result.risk_distribution['high']['llm']}条")
print(f"转接率: {result.risk_distribution['high']['human']/54*100:.1f}%")
print(f"平均延迟: {result.avg_queueing_delay:.1f}ms")

if result.risk_distribution['high']['human'] < 54:
    print(f"\n❌ 未达100%！少了{54-result.risk_distribution['high']['human']}条")
    print("查看上面的DEBUG日志，找出为什么有的High给了LLM")
else:
    print(f"\n✅ 达到100%转接率！")
