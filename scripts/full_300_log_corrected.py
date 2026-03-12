#!/usr/bin/env python3
"""
生成完整的300请求详细日志 - 修正版（时间正确推进）
"""

import sys
sys.path.insert(0, '/home/iilab9/scholar-papers/experiments/intention/exp-2')

import json
import numpy as np
from datetime import datetime
from exp.scheduling_simulation.simulator import SchedulingSimulator, SimulationResult
from exp.scheduling_simulation.routers import DynamicQueueAwareRouter
from exp.scheduling_simulation.config import COST_CONFIG, LLM_LATENCY, QUEUE_CONFIG
from exp.scheduling_simulation.queue_model import Request

np.random.seed(42)

# 加载验证数据
with open('/home/iilab9/scholar-papers/experiments/intention/exp-2/data/validation/validation_set_1000_eval.jsonl', 'r') as f:
    all_data = [json.loads(line) for line in f]

# 取前300条
test_data = all_data[:300]

# 创建自定义仿真器类，添加详细日志
class DetailedLoggingSimulator(SchedulingSimulator):
    """带详细日志的仿真器"""
    
    def __init__(self, router, max_time=600):
        super().__init__(router, max_time)
        self.log_entries = []
        self.cumulative_error = 0.0
        self.cumulative_latency = 0.0
        self.cumulative_human = 0.0
        
    def _handle_arrival(self, data, current_time, request_id, llm_error_func):
        """处理请求到达并记录详细日志"""
        
        # 获取基本信息
        level = data['true_level']
        cost = data['true_cost']
        query = data.get('query', '')[:60] + '...' if len(data.get('query', '')) > 60 else data.get('query', '')
        p_error = 0.8 if level=='high' else 0.4 if level=='mid' else 0.1
        
        # 队列状态（在加入前）
        queue_state = self.queue.get_queue_state()
        wait_seconds = queue_state['expected_wait']
        
        # 计算LLM期望损失
        llm_error_cost = p_error * cost
        llm_latency_cost = LLM_LATENCY['slm'] * COST_CONFIG['latency_cost_per_ms']
        llm_total = llm_error_cost + llm_latency_cost
        
        # 计算人工期望损失
        wait_ms = wait_seconds * 1000
        delay_cost = wait_ms * COST_CONFIG['latency_cost_per_ms']
        human_total = COST_CONFIG['human_fixed_cost'] + delay_cost
        
        # 决策
        if llm_total > human_total:
            decision = "HUMAN"
            actual_cost_human = COST_CONFIG['human_fixed_cost']
            actual_cost_wait = delay_cost
            actual_cost_error = 0
        else:
            decision = "LLM"
            actual_cost_human = 0
            actual_cost_wait = llm_latency_cost
            actual_cost_error = llm_error_cost
        
        # 记录日志
        log_entry = {
            'request_id': request_id,
            'current_time': current_time,
            'query': query,
            'level': level,
            'cost': cost,
            'p_error': p_error,
            'queue_length': queue_state['queue_length'],
            'busy_servers': queue_state['busy_servers'],
            'expected_wait': wait_seconds,
            'llm_error_cost': llm_error_cost,
            'llm_latency_cost': llm_latency_cost,
            'llm_total': llm_total,
            'human_fixed': COST_CONFIG['human_fixed_cost'],
            'human_wait_cost': delay_cost,
            'human_total': human_total,
            'decision': decision,
            'actual_error': actual_cost_error,
            'actual_latency': actual_cost_wait,
            'actual_human': actual_cost_human,
        }
        self.log_entries.append(log_entry)
        
        # 更新累计
        self.cumulative_error += actual_cost_error
        self.cumulative_latency += actual_cost_wait
        self.cumulative_human += actual_cost_human
        
        # 调用父类方法处理实际逻辑
        super()._handle_arrival(data, current_time, request_id, llm_error_func)

# 输出文件
output_file = '/home/iilab9/scholar-papers/experiments/intention/exp-2/output/scheduling_simulation/full_300_requests_log_corrected.txt'

router = DynamicQueueAwareRouter()
simulator = DetailedLoggingSimulator(router=router, max_time=3600)  # 改为3600秒（1小时），确保300条请求都能处理

# 运行仿真
print("开始仿真（带详细日志）...")
print(f"测试数据: {len(test_data)}条请求")
print(f"最大仿真时间: 3600秒")
result = simulator.simulate(
    test_data=test_data,
    llm_error_func=lambda q,l: 0.8 if l=='high' else 0.4 if l=='mid' else 0.1,
    verbose=True
)

# 写入详细日志
with open(output_file, 'w', encoding='utf-8') as f:
    def log(msg):
        print(msg)
        f.write(msg + '\n')
    
    log("="*100)
    log(f"完整300请求调度仿真日志 - 动态队列感知路由（修正版）")
    log(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*100)
    
    log(f"\n【系统配置】")
    log(f"  专家数量: {QUEUE_CONFIG['num_servers_c']}人")
    log(f"  服务时间: {1/QUEUE_CONFIG['service_rate_mu']:.0f}秒/单")
    log(f"  延迟成本系数: {COST_CONFIG['latency_cost_per_ms']}元/ms")
    log(f"  人工固定成本: {COST_CONFIG['human_fixed_cost']}元")
    log(f"  SLM延迟: {LLM_LATENCY['slm']}ms")
    
    log(f"\n【代价权重配置】")
    log(f"  High错误代价: {COST_CONFIG['error_cost']['high']}元 (P(error)=0.8)")
    log(f"  Mid错误代价: {COST_CONFIG['error_cost']['mid']}元 (P(error)=0.4)")
    log(f"  Low错误代价: {COST_CONFIG['error_cost']['low']}元 (P(error)=0.1)")
    
    log("\n" + "="*100)
    log("开始处理请求...")
    log("="*100)
    
    for entry in simulator.log_entries:
        log(f"\n{'─'*100}")
        log(f"【请求 #{entry['request_id']:03d}】 时间:{entry['current_time']:.1f}s | {entry['query']}")
        log(f"风险等级: {entry['level'].upper()} | 错误代价: {entry['cost']}元")
        log(f"{'─'*100}")
        
        log(f"  【步骤1】队列状态:")
        log(f"    - 队列长度: {entry['queue_length']}条")
        log(f"    - 忙碌专家: {entry['busy_servers']}/{QUEUE_CONFIG['num_servers_c']}")
        log(f"    - 预期等待: {entry['expected_wait']:.1f}秒")
        
        log(f"\n  【步骤2】LLM期望损失:")
        log(f"    期望错误代价 = P(error) × 错误代价")
        log(f"                 = {entry['p_error']} × {entry['cost']}")
        log(f"                 = {entry['llm_error_cost']:.2f}元")
        log(f"    延迟代价 = {LLM_LATENCY['slm']}ms × {COST_CONFIG['latency_cost_per_ms']}元/ms")
        log(f"            = {entry['llm_latency_cost']:.4f}元")
        log(f"    → LLM总期望损失: {entry['llm_total']:.2f}元")
        
        log(f"\n  【步骤3】人工期望损失:")
        log(f"    固定成本: {entry['human_fixed']}元")
        wait_ms = entry['expected_wait'] * 1000
        log(f"    等待时间: {entry['expected_wait']:.1f}秒 = {wait_ms:.0f}ms")
        log(f"    等待代价: {wait_ms:.0f}ms × {COST_CONFIG['latency_cost_per_ms']}元/ms = {entry['human_wait_cost']:.2f}元")
        log(f"    → 人工总期望损失: {entry['human_total']:.2f}元")
        
        log(f"\n  【步骤4】决策:")
        log(f"    LLM: {entry['llm_total']:.2f}元 vs 人工: {entry['human_total']:.2f}元")
        
        if entry['decision'] == "HUMAN":
            log(f"    比较: {entry['llm_total']:.2f} > {entry['human_total']:.2f}")
            log(f"    → 转人工")
            log(f"\n  【步骤5】实际成本:")
            log(f"    人工成本: {entry['actual_human']:.2f}元")
            log(f"    等待代价: {entry['actual_latency']:.2f}元")
            log(f"    错误代价: 0元")
            log(f"    → 本请求成本: {entry['actual_human'] + entry['actual_latency']:.2f}元")
        else:
            log(f"    比较: {entry['llm_total']:.2f} ≤ {entry['human_total']:.2f}")
            log(f"    → 给LLM")
            log(f"\n  【步骤5】实际成本:")
            log(f"    错误代价: {entry['actual_error']:.2f}元 (期望损失)")
            log(f"    延迟代价: {entry['actual_latency']:.4f}元")
            log(f"    人工成本: 0元")
            log(f"    → 本请求成本: {entry['actual_error'] + entry['actual_latency']:.2f}元")
        
        # 计算累计
        cum_error = sum(e['actual_error'] for e in simulator.log_entries[:entry['request_id']+1])
        cum_latency = sum(e['actual_latency'] for e in simulator.log_entries[:entry['request_id']+1])
        cum_human = sum(e['actual_human'] for e in simulator.log_entries[:entry['request_id']+1])
        log(f"\n  【累加】错误:{cum_error:.2f} 延迟:{cum_latency:.2f} 人工:{cum_human:.2f} = {cum_error+cum_latency+cum_human:.2f}元")
    
    # 最终统计
    log("\n" + "="*100)
    log("最终统计")
    log("="*100)
    
    total_cost = simulator.cumulative_error + simulator.cumulative_latency + simulator.cumulative_human
    
    log(f"\n【总成本构成】")
    log(f"  错误代价: {simulator.cumulative_error:.2f}元 ({simulator.cumulative_error/total_cost*100:.1f}%)")
    log(f"  延迟代价: {simulator.cumulative_latency:.2f}元 ({simulator.cumulative_latency/total_cost*100:.1f}%)")
    log(f"  人工成本: {simulator.cumulative_human:.2f}元 ({simulator.cumulative_human/total_cost*100:.1f}%)")
    log(f"  总计: {total_cost:.2f}元")
    
    log(f"\n【路由分布】")
    llm_count = {'high': 0, 'mid': 0, 'low': 0}
    human_count = {'high': 0, 'mid': 0, 'low': 0}
    for entry in simulator.log_entries:
        if entry['decision'] == 'LLM':
            llm_count[entry['level']] += 1
        else:
            human_count[entry['level']] += 1
    
    for level in ['high', 'mid', 'low']:
        llm = llm_count[level]
        human = human_count[level]
        total = llm + human
        if total > 0:
            log(f"  {level.upper()}: LLM={llm}, 人工={human} (人工占比{human/total*100:.1f}%)")

print(f"\n修正后的日志已保存到: {output_file}")
