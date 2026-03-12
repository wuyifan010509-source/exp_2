#!/usr/bin/env python3
"""
详细代价计算日志
"""

import sys
sys.path.insert(0, '/home/iilab9/scholar-papers/experiments/intention/exp-2')

import numpy as np
from exp.scheduling_simulation import DynamicQueueAwareRouter, SchedulingSimulator, COST_CONFIG, LLM_LATENCY, QUEUE_CONFIG

np.random.seed(42)

# 9条测试数据
test_data = []
for i in range(3):
    test_data.append({'query': f'High_{i}', 'true_level': 'high', 'true_cost': 2000})
for i in range(3):
    test_data.append({'query': f'Mid_{i}', 'true_level': 'mid', 'true_cost': 10})
for i in range(3):
    test_data.append({'query': f'Low_{i}', 'true_level': 'low', 'true_cost': 1})

print("="*80)
print("详细代价计算日志")
print("="*80)
print(f"\n配置:")
print(f"  High/Mid/Low代价: 2000/10/1元")
print(f"  人工固定成本: {COST_CONFIG['human_fixed_cost']}元")
print(f"  延迟系数: {COST_CONFIG['latency_cost_per_ms']}元/ms")
print(f"  服务时间: {1/QUEUE_CONFIG['service_rate_mu']:.0f}秒")
print(f"  专家: {QUEUE_CONFIG['num_servers_c']}人")

router = DynamicQueueAwareRouter()
simulator = SchedulingSimulator(router=router, max_time=600)

total_error = 0
total_latency = 0
total_human = 0

for idx, data in enumerate(test_data):
    print(f"\n{'─'*80}")
    print(f"请求 #{idx}: {data['query']} [{data['true_level'].upper()}]")
    
    p_error = 0.8 if data['true_level']=='high' else 0.4 if data['true_level']=='mid' else 0.1
    queue_state = simulator.queue.get_queue_state()
    
    # 成本计算
    llm_error_cost = p_error * data['true_cost']
    llm_latency = LLM_LATENCY['slm'] * COST_CONFIG['latency_cost_per_ms']
    llm_total = llm_error_cost + llm_latency
    
    wait_ms = queue_state['expected_wait'] * 1000
    delay_cost = wait_ms * COST_CONFIG['latency_cost_per_ms']
    human_total = COST_CONFIG['human_fixed_cost'] + delay_cost
    
    print(f"  P(error)={p_error:.2f}, 队列等待={queue_state['expected_wait']:.1f}s")
    print(f"  LLM期望损失: {llm_error_cost:.2f} + {llm_latency:.4f} = {llm_total:.2f}元")
    print(f"  人工期望损失: {COST_CONFIG['human_fixed_cost']:.2f} + {delay_cost:.2f} = {human_total:.2f}元")
    
    # 决策
    if llm_total > human_total:
        decision = "HUMAN"
        actual_cost = COST_CONFIG['human_fixed_cost']
        wait_time = queue_state['expected_wait']
        total_human += actual_cost
        total_latency += wait_time * 1000 * COST_CONFIG['latency_cost_per_ms']
        print(f"  → 决策: {decision} (人工更划算)")
        print(f"  → 实际成本: {actual_cost:.2f}元 (人工固定)")
        # 加入队列
        from exp.scheduling_simulation.queue_model import Request
        req = Request(id=idx, arrival_time=idx, query=data['query'],
                      true_risk_level=data['true_level'], true_cost=data['true_cost'], llm_error_prob=p_error)
        simulator.queue.enqueue(req, idx)
    else:
        decision = "LLM"
        is_error = np.random.random() < p_error
        if is_error:
            actual_cost = data['true_cost']
            total_error += actual_cost
            print(f"  → 决策: {decision}")
            print(f"  → 随机: {np.random.random():.4f} < {p_error} = 答错 ❌")
            print(f"  → 实际成本: {actual_cost:.2f}元 (错误代价)")
        else:
            actual_cost = llm_latency
            total_latency += actual_cost
            print(f"  → 决策: {decision}")
            print(f"  → 随机: {np.random.random():.4f} >= {p_error} = 答对 ✅")
            print(f"  → 实际成本: {actual_cost:.4f}元 (延迟代价)")

print(f"\n{'='*80}")
print("汇总")
print(f"{'='*80}")
print(f"错误代价: {total_error:.2f}元")
print(f"延迟代价: {total_latency:.4f}元")
print(f"人工成本: {total_human:.2f}元")
print(f"总计: {total_error + total_latency + total_human:.2f}元")
