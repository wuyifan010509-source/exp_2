#!/usr/bin/env python3
"""
生成完整的300请求详细日志
使用 simulator.py 的详细日志功能
"""

import sys
sys.path.insert(0, '/home/iilab9/scholar-papers/experiments/intention/exp-2')

import json
import os
from datetime import datetime
from exp.scheduling_simulation.simulator import SchedulingSimulator
from exp.scheduling_simulation.routers import DynamicQueueAwareRouter
from exp.scheduling_simulation.config import COST_CONFIG, LLM_LATENCY, QUEUE_CONFIG

# 加载验证数据
with open('/home/iilab9/scholar-papers/experiments/intention/exp-2/data/validation/validation_set_1000_eval.jsonl', 'r') as f:
    all_data = [json.loads(line) for line in f]

# 取前300条
test_data = all_data[:300]

# 输出文件
output_dir = '/home/iilab9/scholar-papers/experiments/intention/exp-2/output/scheduling_simulation'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'full_300_requests_log.txt')

# 创建仿真器
router = DynamicQueueAwareRouter()
simulator = SchedulingSimulator(router=router, max_time=600)

# 写入日志头部
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*100 + "\n")
    f.write(f"完整300请求调度仿真日志 - 动态队列感知路由\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*100 + "\n")
    
    f.write(f"\n【系统配置】\n")
    f.write(f"  专家数量: {QUEUE_CONFIG['num_servers_c']}人\n")
    f.write(f"  服务时间: {1/QUEUE_CONFIG['service_rate_mu']:.0f}秒/单\n")
    f.write(f"  延迟成本系数: {COST_CONFIG['latency_cost_per_ms']}元/ms\n")
    f.write(f"  人工固定成本: {COST_CONFIG['human_fixed_cost']}元\n")
    f.write(f"  SLM延迟: {LLM_LATENCY['slm']}ms\n")
    
    f.write(f"\n【代价权重配置】\n")
    f.write(f"  High错误代价: {COST_CONFIG['error_cost']['high']}元 (P(error)=0.8)\n")
    f.write(f"  Mid错误代价: {COST_CONFIG['error_cost']['mid']}元 (P(error)=0.4)\n")
    f.write(f"  Low错误代价: {COST_CONFIG['error_cost']['low']}元 (P(error)=0.1)\n")
    
    f.write("\n" + "="*100 + "\n")
    f.write("开始处理请求...\n")
    f.write("="*100 + "\n")

# 运行仿真（启用详细日志）
result = simulator.simulate(
    test_data=test_data,
    log_file=output_file,
    verbose=True
)

print(f"\n✅ 详细日志已保存到: {output_file}")
print(f"总成本: {result.expected_total_cost:.2f}元")
print(f"转人工率: {result.deferral_rate*100:.1f}%")
