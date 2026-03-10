"""
人机调度仿真主入口
运行完整的对比实验
"""

import argparse
import json
import numpy as np
from typing import List, Dict, Optional
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exp.scheduling_simulation.queue_model import Request
from exp.scheduling_simulation.routers import (
    LLMOnlyRouter,
    VanillaL2DRouter,
    StaticCostRouter,
    DynamicQueueAwareRouter,
)
from exp.scheduling_simulation.simulator import SchedulingSimulator
from exp.scheduling_simulation.evaluation import (
    generate_comparison_table,
    plot_cost_delay_tradeoff,
    plot_routing_behavior,
    plot_traffic_pattern_over_time,
    save_results_to_json,
)


def load_test_data(data_path: str) -> List[Dict]:
    """
    加载测试数据
    
    期望格式：JSONL
    {"query": "...", "true_level": "low/mid/high", "true_cost": 1/50/1000}
    """
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            
            # 标准化字段
            if "level" in record:
                record["true_level"] = record["level"]
            elif "class_label" in record:
                # 0=low, 1=mid, 2=high
                mapping = {0: "low", 1: "mid", 2: "high"}
                record["true_level"] = mapping.get(record["class_label"], "mid")
            
            if "cost_label" in record:
                record["true_cost"] = record["cost_label"]
            elif "true_cost" not in record:
                # 默认映射
                cost_map = {"low": 1, "mid": 50, "high": 1000}
                record["true_cost"] = cost_map.get(record.get("true_level", "mid"), 50)
            
            data.append(record)
    
    return data


def estimate_llm_error_probability(query: str, true_level: str, use_model: bool = False) -> float:
    """
    估计 LLM 错误概率 P(LLM错)
    
    策略：
    1. 使用预训练的置信度估计模型（如果use_model=True）
    2. 基于风险等级的启发式估计
    """
    if use_model:
        # TODO: 接入实际的SLM模型
        # 这里可以调用 exp.slm_distillation.inference.SLMCostPredictor
        pass
    
    # 启发式估计
    # High风险：问题复杂，LLM容易答错
    # Mid风险：一般业务问题
    # Low风险：简单问题，LLM很擅长
    if true_level == "high":
        # High风险：30%概率答错（实际取决于具体SLM性能）
        base_error = 0.30
    elif true_level == "mid":
        base_error = 0.15
    else:
        base_error = 0.05
    
    # 根据查询长度调整（短查询可能更明确）
    if len(query) < 10:
        base_error *= 0.9
    elif len(query) > 50:
        base_error *= 1.1
    
    return min(base_error, 0.95)  # 上限95%


def run_single_experiment(
    router,
    test_data: List[Dict],
    max_time: float = 4*3600,  # 4小时
    verbose: bool = True
):
    """运行单次仿真实验"""
    simulator = SchedulingSimulator(
        router=router,
        max_time=max_time
    )
    
    result = simulator.simulate(
        test_data=test_data,
        llm_error_func=lambda q, lvl: estimate_llm_error_probability(q, lvl),
        verbose=verbose
    )
    
    return result


def run_comparison_experiment(
    test_data: List[Dict],
    output_dir: str = "./output/scheduling_simulation",
    max_time: float = 4*3600,
):
    """
    运行四种策略的对比实验
    """
    print("="*70)
    print("人机混合智能体调度仿真 - 对比实验")
    print("="*70)
    print(f"\n测试集规模: {len(test_data)} 条请求")
    print(f"仿真时长: {max_time/3600:.1f} 小时")
    print(f"\n流量模式:")
    print("  - 0-30min: 早盘高峰 (10 req/s)")
    print("  - 30-120min: 平时 (2 req/s)")
    print("  - 120-150min: 午盘高峰 (6 req/s)")
    print("  - 150-240min: 平时 (2 req/s)")
    
    # 初始化四种策略
    routers = [
        LLMOnlyRouter(),
        VanillaL2DRouter(threshold=0.2),
        StaticCostRouter(human_cost_threshold=100.0),
        DynamicQueueAwareRouter(),
    ]
    
    # 运行仿真
    results = []
    for router in routers:
        result = run_single_experiment(router, test_data, max_time)
        results.append(result)
    
    # 生成对比表格
    comparison_table = generate_comparison_table(results)
    print(comparison_table)
    
    # 保存表格
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "comparison_table.md", 'w', encoding='utf-8') as f:
        f.write(comparison_table)
    
    # 生成可视化
    print("\n" + "="*70)
    print("生成可视化图表...")
    print("="*70)
    
    plot_cost_delay_tradeoff(results, str(output_path / "cost_delay_tradeoff.png"))
    plot_routing_behavior(results, str(output_path / "routing_behavior.png"))
    plot_traffic_pattern_over_time(str(output_path / "traffic_pattern.png"))
    
    # 保存JSON结果
    save_results_to_json(results, str(output_path / "simulation_results.json"))
    
    print("\n" + "="*70)
    print("实验完成！")
    print(f"结果保存目录: {output_dir}")
    print("="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="人机混合智能体调度仿真实验"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./data/validation_set_1000_eval.jsonl",
        help="测试数据路径 (JSONL格式)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/scheduling_simulation",
        help="输出目录"
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=4*3600,
        help="仿真时长（秒），默认4小时"
    )
    parser.add_argument(
        "--num_requests",
        type=int,
        default=None,
        help="使用的请求数量（默认全部）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 加载数据
    print(f"加载测试数据: {args.test_data}")
    test_data = load_test_data(args.test_data)
    
    if args.num_requests:
        test_data = test_data[:args.num_requests]
    
    if not test_data:
        print("错误: 测试数据为空")
        return
    
    # 运行实验
    results = run_comparison_experiment(
        test_data=test_data,
        output_dir=args.output_dir,
        max_time=args.max_time
    )
    
    return results


if __name__ == "__main__":
    main()
