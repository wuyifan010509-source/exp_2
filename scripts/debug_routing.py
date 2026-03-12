#!/usr/bin/env python3
"""
调试脚本：检查High风险的决策过程
"""

import sys
sys.path.insert(0, '/home/iilab9/scholar-papers/experiments/intention/exp-2')

import numpy as np
from exp.scheduling_simulation import (
    DynamicQueueAwareRouter,
    SchedulingSimulator,
    COST_CONFIG
)
from exp.scheduling_simulation.config import LLM_LATENCY
from exp.scheduling_simulation.config import LLM_LATENCY

np.random.seed(42)

# 创建测试数据：只有High风险
test_data = []
for i in range(10):
    test_data.append({
        "query": f"高危测试查询_{i}",
        "true_level": "high",
        "true_cost": 2000  # High风险代价2000元
    })

print("="*70)
print("调试：Dynamic方法对High风险的决策过程")
print("="*70)
print(f"\n配置参数：")
print(f"  High风险错误概率: 0.8")
print(f"  High风险错误代价: 2000元")
print(f"  人工固定成本: {COST_CONFIG['human_fixed_cost']}元")
print(f"  延迟代价系数: {COST_CONFIG['latency_cost_per_ms']}元/ms")
print(f"  SLM延迟: {LLM_LATENCY['slm']}ms")
print(f"\n预期计算：")
print(f"  LLM期望损失 = 0.8 × 2000 + 16 × 0.005 = 1600.08元")
print(f"  人工期望损失 = 25 + 0 = 25元")
print(f"  决策: 1600.08 > 25 → 应该转人工！")
print(f"\n实际运行：")
print("="*70)

# 创建路由器和仿真器
router = DynamicQueueAwareRouter()
simulator = SchedulingSimulator(router=router, max_time=3600)

# 手动测试前3条
for i, data in enumerate(test_data[:3], 1):
    # 模拟到达
    from exp.scheduling_simulation.queue_model import Request
    
    request = Request(
        id=i,
        arrival_time=i * 1.0,
        query=data["query"],
        true_risk_level=data["true_level"],
        true_cost=data["true_cost"],
        llm_error_prob=0.8  # 直接设置错误概率为0.8
    )
    
    # 获取队列状态
    queue_state = simulator.queue.get_queue_state()
    
    # 做决策
    decision = router.route(request, simulator.queue, i * 1.0)
    
    print(f"\nQuery {i}: {data['query']}")
    print(f"  P(error) = {request.llm_error_prob}")
    print(f"  LLM期望损失 = {decision.expected_cost_llm:.2f}元")
    print(f"  人工期望损失 = {decision.expected_cost_human:.2f}元")
    print(f"  队列等待 = {queue_state['expected_wait']:.1f}s")
    print(f"  决策: {decision.assign_to.upper()}")
    print(f"  原因: {decision.reason}")

print("\n" + "="*70)
print("检查结论：")
print("="*70)
