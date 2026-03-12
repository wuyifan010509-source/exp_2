#!/usr/bin/env python3
"""
不同流量强度对比实验

测试3种流量场景：
1. 高流量 (High): 早盘10, 午盘6 req/min (原始配置)
2. 中流量 (Medium): 早盘5, 午盘3 req/min
3. 低流量 (Low): 早盘2, 午盘1 req/min
"""

import sys
sys.path.insert(0, '/home/iilab9/scholar-papers/experiments/intention/exp-2')

import numpy as np
import json
from pathlib import Path
from copy import deepcopy

# 导入实验模块
from exp.scheduling_simulation import (
    LLMOnlyRouter,
    VanillaL2DRouter,
    StaticCostRouter,
    DynamicQueueAwareRouter,
    SchedulingSimulator,
    COST_CONFIG,
    QUEUE_CONFIG,
)
from exp.scheduling_simulation.priority_router import DynamicPriorityQueueRouter
from exp.scheduling_simulation.evaluation import save_results_to_json

np.random.seed(42)


def load_test_data(filepath: str, num_requests: int = 300):
    """加载测试数据"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_requests:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            
            # 标准化字段
            if "level" in record:
                record["true_level"] = record["level"]
            elif "class_label" in record:
                mapping = {0: "low", 1: "mid", 2: "high"}
                record["true_level"] = mapping.get(record["class_label"], "mid")
            
            if "cost_label" in record:
                record["true_cost"] = record["cost_label"]
            elif "true_cost" not in record:
                cost_map = {"low": 1, "mid": 10, "high": 2000}
                record["true_cost"] = cost_map.get(record.get("true_level", "mid"), 10)
            
            data.append(record)
    
    return data


def run_experiment(router, test_data, traffic_config, max_time=7200):
    """运行单次实验"""
    # 临时修改流量配置
    original_phases = None
    
    # 创建仿真器时注入流量配置
    simulator = SchedulingSimulator(
        router=router,
        max_time=max_time,
        queue_type="default" if "Priority" not in router.name else "priority",
        custom_traffic_phases=traffic_config
    )
    
    result = simulator.simulate(test_data=test_data, verbose=False)
    return result, simulator


def main():
    # 加载数据
    test_data = load_test_data(
        "./data/validation/validation_set_1000_eval_with_prob.jsonl",
        num_requests=300
    )
    
    print(f"测试数据: {len(test_data)} 条")
    
    # 定义3种流量配置
    # 格式: [(持续时间秒, 到达率req/s), ...]
    traffic_configs = {
        "high": [
            (1800, 10.0/60),   # 早盘: 10 req/min
            (5400, 2.0/60),    # 平时: 2 req/min
            (1800, 6.0/60),    # 午盘: 6 req/min
            (5400, 2.0/60),    # 平时: 2 req/min
        ],
        "medium": [
            (1800, 5.0/60),    # 早盘: 5 req/min (50%)
            (5400, 1.0/60),    # 平时: 1 req/min
            (1800, 3.0/60),    # 午盘: 3 req/min (50%)
            (5400, 1.0/60),    # 平时: 1 req/min
        ],
        "low": [
            (1800, 2.0/60),    # 早盘: 2 req/min (20%)
            (5400, 0.5/60),    # 平时: 0.5 req/min
            (1800, 1.0/60),    # 午盘: 1 req/min (17%)
            (5400, 0.5/60),    # 平时: 0.5 req/min
        ],
    }
    
    # 定义策略
    strategies = [
        ("LLM-Only", LLMOnlyRouter()),
        ("Vanilla L2D", VanillaL2DRouter(threshold=0.2)),
        ("Static Cost", StaticCostRouter(human_cost_threshold=100.0)),
        ("Dynamic", DynamicQueueAwareRouter()),
        ("Dynamic Priority", DynamicPriorityQueueRouter()),
    ]
    
    # 存储所有结果
    all_results = {}
    
    for traffic_name, traffic_config in traffic_configs.items():
        print(f"\n{'='*80}")
        print(f"流量场景: {traffic_name.upper()}")
        print(f"{'='*80}")
        
        # 显示流量配置
        print("\n流量配置:")
        phase_names = ["早盘高峰(0-30min)", "平时(30-120min)", "午盘高峰(120-150min)", "平时(150-240min)"]
        for i, (duration, rate) in enumerate(traffic_config):
            print(f"  {phase_names[i]}: {rate*60:.1f} req/min")
        
        traffic_results = []
        
        for name, router in strategies:
            print(f"\n{'-'*80}")
            print(f"策略: {name}")
            print(f"{'-'*80}")
            
            result, simulator = run_experiment(router, test_data, traffic_config)
            
            # 更新结果名称包含流量信息
            result.router_name = f"{name} [{traffic_name}]"
            traffic_results.append(result)
            
            # 打印关键指标
            print(f"总期望成本: {result.expected_total_cost:.2f}元")
            print(f"灾难性错误率: {result.catastrophic_error_rate:.4f}")
            print(f"平均排队延迟: {result.avg_queueing_delay:.1f}ms")
            print(f"人工转交率: {result.deferral_rate:.4f}")
            
            # High风险转接详情
            high_to_human = result.risk_distribution.get("high", {}).get("human", 0)
            high_to_llm = result.risk_distribution.get("high", {}).get("llm", 0)
            high_total = high_to_human + high_to_llm
            if high_total > 0:
                high_deferral_rate = high_to_human / high_total
                print(f"High风险转接率: {high_deferral_rate*100:.1f}% ({high_to_human}/{high_total})")
        
        all_results[traffic_name] = traffic_results
        
        # 生成本流量场景的对比表
        print(f"\n{'='*80}")
        print(f"{traffic_name.upper()} 流量 - 策略对比表")
        print(f"{'='*80}")
        print(f"{'策略':<35} {'ETC':<12} {'CER':<10} {'延迟(ms)':<12} {'High转接率':<12}")
        print("-"*85)
        
        for result in traffic_results:
            high_to_human = result.risk_distribution.get("high", {}).get("human", 0)
            high_to_llm = result.risk_distribution.get("high", {}).get("llm", 0)
            high_total = high_to_human + high_to_llm
            high_deferral = (high_to_human / high_total * 100) if high_total > 0 else 0
            
            short_name = result.router_name.replace(f" [{traffic_name}]", "")
            print(f"{short_name:<35} {result.expected_total_cost:<12.2f} "
                  f"{result.catastrophic_error_rate:<10.4f} {result.avg_queueing_delay:<12.1f} "
                  f"{high_deferral:<12.1f}%")
    
    # 保存结果
    output_dir = Path("./output/traffic_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 合并所有结果保存
    flat_results = []
    for traffic_name, results in all_results.items():
        flat_results.extend(results)
    
    save_results_to_json(flat_results, str(output_dir / "traffic_comparison_results.json"))
    
    print(f"\n{'='*80}")
    print("所有流量场景实验完成！")
    print(f"结果保存: {output_dir}")
    print(f"{'='*80}")
    
    # 跨流量对比
    print("\n跨流量对比 - Dynamic Priority策略:")
    print("-"*80)
    print(f"{'流量场景':<15} {'ETC':<12} {'CER':<10} {'延迟(ms)':<12} {'High转接率':<12}")
    print("-"*80)
    
    for traffic_name in ["low", "medium", "high"]:
        result = next((r for r in all_results[traffic_name] if "Priority" in r.router_name), None)
        if result:
            high_to_human = result.risk_distribution.get("high", {}).get("human", 0)
            high_to_llm = result.risk_distribution.get("high", {}).get("llm", 0)
            high_total = high_to_human + high_to_llm
            high_deferral = (high_to_human / high_total * 100) if high_total > 0 else 0
            
            print(f"{traffic_name.upper():<15} {result.expected_total_cost:<12.2f} "
                  f"{result.catastrophic_error_rate:<10.4f} {result.avg_queueing_delay:<12.1f} "
                  f"{high_deferral:<12.1f}%")


if __name__ == "__main__":
    main()
