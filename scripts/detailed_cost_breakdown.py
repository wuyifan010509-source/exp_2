#!/usr/bin/env python3
"""
详细代价计算中间过程
"""

import sys
sys.path.insert(0, '/home/iilab9/scholar-papers/experiments/intention/exp-2')

import numpy as np
from exp.scheduling_simulation.simulator import SchedulingSimulator
from exp.scheduling_simulation.routers import DynamicQueueAwareRouter
from exp.scheduling_simulation.config import COST_CONFIG, LLM_LATENCY, QUEUE_CONFIG
from exp.scheduling_simulation.queue_model import Request

np.random.seed(42)

# 创建少量测试数据（便于查看）
test_data = []
# 6条High
for i in range(6):
    test_data.append({'query': f'High_{i}', 'true_level': 'high', 'true_cost': 2000})
# 3条Mid
for i in range(3):
    test_data.append({'query': f'Mid_{i}', 'true_level': 'mid', 'true_cost': 10})
# 3条Low
for i in range(3):
    test_data.append({'query': f'Low_{i}', 'true_level': 'low', 'true_cost': 1})

print("="*100)
print("详细代价计算中间过程")
print("="*100)
print(f"\n配置参数:")
print(f"  High/Mid/Low错误代价: {COST_CONFIG['error_cost']['high']}/{COST_CONFIG['error_cost']['mid']}/{COST_CONFIG['error_cost']['low']}元")
print(f"  人工固定成本: {COST_CONFIG['human_fixed_cost']}元")
print(f"  延迟成本系数: {COST_CONFIG['latency_cost_per_ms']}元/ms")
print(f"  SLM延迟: {LLM_LATENCY['slm']}ms")
print(f"  专家数量: {QUEUE_CONFIG['num_servers_c']}人")
print(f"  服务时间: {1/QUEUE_CONFIG['service_rate_mu']:.0f}秒/单")

print("\n" + "="*100)
print("开始处理请求...")
print("="*100)

router = DynamicQueueAwareRouter()
simulator = SchedulingSimulator(router=router, max_time=600)

# 累加器
cumulative_error = 0.0
cumulative_latency = 0.0
cumulative_human = 0.0

for idx, data in enumerate(test_data):
    print(f"\n{'─'*100}")
    print(f"【请求 #{idx}】 {data['query']} [{data['true_level'].upper()}]")
    print(f"{'─'*100}")
    
    p_error = 0.8 if data['true_level']=='high' else 0.4 if data['true_level']=='mid' else 0.1
    
    print(f"  输入参数:")
    print(f"    - 风险等级: {data['true_level'].upper()}")
    print(f"    - 错误代价: {data['true_cost']}元")
    print(f"    - P(error): {p_error} ({p_error*100:.0f}%)")
    
    # 检查队列状态
    queue_state = simulator.queue.get_queue_state()
    print(f"\n  【步骤1】获取队列状态:")
    print(f"    - 队列长度: {queue_state['queue_length']}条")
    print(f"    - 忙碌专家: {queue_state['busy_servers']}/{QUEUE_CONFIG['num_servers_c']}")
    print(f"    - 预期等待: {queue_state['expected_wait']:.1f}秒")
    
    # 计算LLM期望损失
    llm_error_cost = p_error * data['true_cost']
    llm_latency_cost = LLM_LATENCY['slm'] * COST_CONFIG['latency_cost_per_ms']
    llm_total = llm_error_cost + llm_latency_cost
    
    print(f"\n  【步骤2】计算LLM期望损失:")
    print(f"    - 期望错误代价 = P(error) × 错误代价")
    print(f"                   = {p_error} × {data['true_cost']}")
    print(f"                   = {llm_error_cost:.2f}元")
    print(f"    - 延迟代价 = {LLM_LATENCY['slm']}ms × {COST_CONFIG['latency_cost_per_ms']}元/ms")
    print(f"              = {llm_latency_cost:.4f}元")
    print(f"    - LLM总期望损失 = {llm_error_cost:.2f} + {llm_latency_cost:.4f}")
    print(f"                    = {llm_total:.2f}元")
    
    # 计算人工期望损失
    wait_seconds = queue_state['expected_wait']
    wait_ms = wait_seconds * 1000
    delay_cost = wait_ms * COST_CONFIG['latency_cost_per_ms']
    human_total = COST_CONFIG['human_fixed_cost'] + delay_cost
    
    print(f"\n  【步骤3】计算人工期望损失:")
    print(f"    - 固定成本 = {COST_CONFIG['human_fixed_cost']}元")
    print(f"    - 等待时间 = {wait_seconds:.1f}秒 = {wait_ms:.0f}ms")
    print(f"    - 等待代价 = {wait_ms:.0f}ms × {COST_CONFIG['latency_cost_per_ms']}元/ms")
    print(f"              = {delay_cost:.2f}元")
    print(f"    - 人工总期望损失 = {COST_CONFIG['human_fixed_cost']:.2f} + {delay_cost:.2f}")
    print(f"                     = {human_total:.2f}元")
    
    # 决策
    print(f"\n  【步骤4】决策比较:")
    print(f"    - LLM期望损失: {llm_total:.2f}元")
    print(f"    - 人工期望损失: {human_total:.2f}元")
    
    if llm_total > human_total:
        decision = "HUMAN"
        print(f"    - 比较: {llm_total:.2f} > {human_total:.2f}")
        print(f"    - 决策: 转人工 (人工更划算)")
        
        # 加入队列
        req = Request(id=idx, arrival_time=idx, query=data['query'],
                      true_risk_level=data['true_level'], true_cost=data['true_cost'], llm_error_prob=p_error)
        simulator.queue.enqueue(req, idx)
        
        # 实际成本
        actual_cost_human = COST_CONFIG['human_fixed_cost']
        actual_cost_wait = delay_cost
        cumulative_human += actual_cost_human
        cumulative_latency += actual_cost_wait
        
        print(f"\n  【步骤5】实际产生成本:")
        print(f"    - 人工成本: {actual_cost_human:.2f}元 (固定)")
        print(f"    - 等待代价: {actual_cost_wait:.2f}元 (延迟)")
        print(f"    - 本请求成本: {actual_cost_human + actual_cost_wait:.2f}元")
        print(f"    - 错误代价: 0元 (人工处理不会答错)")
        
    else:
        decision = "LLM"
        print(f"    - 比较: {llm_total:.2f} ≤ {human_total:.2f}")
        print(f"    - 决策: 给LLM (LLM更划算)")
        
        # 实际成本（期望损失）
        actual_cost_error = llm_error_cost
        actual_cost_latency = llm_latency_cost
        cumulative_error += actual_cost_error
        cumulative_latency += actual_cost_latency
        
        print(f"\n  【步骤5】实际产生成本:")
        print(f"    - 错误代价: {actual_cost_error:.2f}元 (期望损失)")
        print(f"    - 延迟代价: {actual_cost_latency:.4f}元")
        print(f"    - 本请求成本: {actual_cost_error + actual_cost_latency:.2f}元")
        print(f"    - 人工成本: 0元 (未使用人工)")
    
    # 累加显示
    print(f"\n  【累加结果】")
    print(f"    - 累计错误代价: {cumulative_error:.2f}元")
    print(f"    - 累计延迟代价: {cumulative_latency:.2f}元")
    print(f"    - 累计人工成本: {cumulative_human:.2f}元")
    print(f"    - 累计总成本: {cumulative_error + cumulative_latency + cumulative_human:.2f}元")

print(f"\n{'='*100}")
print("最终汇总")
print(f"{'='*100}")
print(f"\n总请求数: {len(test_data)}条")
print(f"\n代价构成:")
print(f"  ├─ 错误代价: {cumulative_error:.2f}元 ({cumulative_error/(cumulative_error+cumulative_latency+cumulative_human)*100:.1f}%)")
print(f"  ├─ 延迟代价: {cumulative_latency:.2f}元 ({cumulative_latency/(cumulative_error+cumulative_latency+cumulative_human)*100:.1f}%)")
print(f"  ├─ 人工成本: {cumulative_human:.2f}元 ({cumulative_human/(cumulative_error+cumulative_latency+cumulative_human)*100:.1f}%)")
print(f"  └─ 总计: {cumulative_error + cumulative_latency + cumulative_human:.2f}元")

print(f"\n按风险等级统计:")
result = simulator.simulate(test_data=test_data, llm_error_func=lambda q,l: 0.8 if l=='high' else 0.4 if l=='mid' else 0.1, verbose=False)
for level in ['high', 'mid', 'low']:
    counts = result.risk_distribution.get(level, {'llm':0, 'human':0})
    total = counts['llm'] + counts['human']
    if total > 0:
        print(f"  {level.upper()}: LLM={counts['llm']}, 人工={counts['human']} (人工占比{counts['human']/total*100:.1f}%)")
