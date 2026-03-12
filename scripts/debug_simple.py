#!/usr/bin/env python3
"""
快速调试：检查High风险的决策计算
"""

import sys
sys.path.insert(0, '/home/iilab9/scholar-papers/experiments/intention/exp-2')

from exp.scheduling_simulation.routers import DynamicQueueAwareRouter
from exp.scheduling_simulation.queue_model import MMcQueue, Request
from exp.scheduling_simulation.config import COST_CONFIG, LLM_LATENCY

print("="*70)
print("快速调试：Dynamic路由决策计算")
print("="*70)

# 创建路由器和空队列
router = DynamicQueueAwareRouter()
queue = MMcQueue(num_servers=3, service_rate=1/180)

# 创建High风险请求
request = Request(
    id=1,
    arrival_time=1.0,
    query="测试High风险",
    true_risk_level="high",
    true_cost=2000,
    llm_error_prob=0.8
)

print(f"\n请求信息：")
print(f"  风险等级: {request.true_risk_level}")
print(f"  错误代价: {request.true_cost}元")
print(f"  P(error): {request.llm_error_prob}")

print(f"\n配置参数：")
print(f"  人工固定成本: {COST_CONFIG['human_fixed_cost']}元")
print(f"  延迟代价系数: {COST_CONFIG['latency_cost_per_ms']}元/ms")
print(f"  SLM延迟: {LLM_LATENCY['slm']}ms")

# 手动计算
cost_llm_error = request.llm_error_prob * request.true_cost
cost_llm_latency = LLM_LATENCY['slm'] * COST_CONFIG['latency_cost_per_ms']
cost_llm_total = cost_llm_error + cost_llm_latency

queue_state = queue.get_queue_state()
expected_wait_seconds = queue_state['expected_wait']
expected_wait_ms = expected_wait_seconds * 1000
delay_cost = expected_wait_ms * COST_CONFIG['latency_cost_per_ms']
cost_human_total = COST_CONFIG['human_fixed_cost'] + delay_cost

print(f"\n计算过程：")
print(f"  LLM错误期望损失 = {request.llm_error_prob} × {request.true_cost} = {cost_llm_error:.2f}元")
print(f"  LLM延迟期望损失 = {LLM_LATENCY['slm']} × {COST_CONFIG['latency_cost_per_ms']} = {cost_llm_latency:.4f}元")
print(f"  LLM总期望损失 = {cost_llm_total:.2f}元")
print(f"\n  队列预期等待 = {expected_wait_seconds:.1f}s")
print(f"  人工延迟代价 = {expected_wait_ms:.0f} × {COST_CONFIG['latency_cost_per_ms']} = {delay_cost:.4f}元")
print(f"  人工固定成本 = {COST_CONFIG['human_fixed_cost']}元")
print(f"  人工总期望损失 = {cost_human_total:.2f}元")

print(f"\n决策比较：")
print(f"  LLM期望损失: {cost_llm_total:.2f}元")
print(f"  人工期望损失: {cost_human_total:.2f}元")

if cost_llm_total > cost_human_total:
    print(f"  → 应该转人工 (LLM损失 > 人工损失)")
else:
    print(f"  → 应该给LLM (LLM损失 ≤ 人工损失)")

# 实际调用路由
decision = router.route(request, queue, 1.0)
print(f"\n实际决策: {decision.assign_to.upper()}")
print(f"决策原因: {decision.reason}")

print("\n" + "="*70)
