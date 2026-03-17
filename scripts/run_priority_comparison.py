#!/usr/bin/env python3
"""
完整对比实验：普通队列 vs 优先级队列

测试5种策略：
1. LLM-Only (基线)
2. Vanilla L2D (传统阈值)
3. Static Cost (静态代价)
4. Dynamic (原版动态调度)
5. Dynamic Priority (优先级队列版本)

目标：验证优先级队列能提高High风险转接率，同时保持较低延迟
"""

import sys
sys.path.insert(0, '/home/iilab9/scholar-papers/experiments/intention/exp-2')

import numpy as np
import json
from pathlib import Path

from exp.scheduling_simulation import (
    LLMOnlyRouter,
    VanillaL2DRouter,
    StaticCostRouter,
    DynamicQueueAwareRouter,
    SchedulingSimulator,
)
# from exp.scheduling_simulation.priority_router import DynamicPriorityQueueRouter
from exp.scheduling_simulation.evaluation import (
    generate_comparison_table,
    plot_cost_delay_tradeoff,
    save_results_to_json,
)

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
            if record.get("level") is not None:
                record["true_level"] = record["level"]
            elif "class_label" in record:
                mapping = {0: "low", 1: "mid", 2: "high"}
                record["true_level"] = mapping.get(record["class_label"], "mid")
            
            if record.get("cost_label") is not None:
                record["true_cost"] = record["cost_label"]
            elif record.get("true_cost") is None:
                cost_map = {"low": 1, "mid": 10, "high": 2000}  # 使用新的权重
                record["true_cost"] = cost_map.get(record.get("true_level", "mid"), 10)
            
            data.append(record)
    
    return data


def run_experiment(router, test_data, max_time=72, queue_type="default", seed=42):
    """运行单次实验"""
    # 每次实验前重置随机种子，确保可复现
    np.random.seed(seed)
    
    simulator = SchedulingSimulator(
        router=router,
        max_time=100,
        queue_type=queue_type
    )
    
    result = simulator.simulate(
        test_data=test_data,
        verbose=False
    )
    
    # 打印详细的延迟计算过程
    print_delay_calculation(simulator, result)
    
    return result, simulator


def print_delay_calculation(simulator, result):
    """打印延迟计算的详细过程"""
    print("\n" + "="*80)
    print("【延迟计算详细过程】")
    print("="*80)
    
    # 获取所有人工处理的请求
    human_requests = [r for r in simulator.completed_requests if r.assigned_to == "human"]
    llm_requests = [r for r in simulator.completed_requests if r.assigned_to == "llm"]
    
    print(f"\n1. 请求分类统计:")
    print(f"   总请求数: {len(simulator.completed_requests)}")
    print(f"   - 人工处理: {len(human_requests)} 条 (计入延迟统计)")
    print(f"   - LLM处理: {len(llm_requests)} 条 (不计入延迟统计)")
    
    if not human_requests:
        print("\n   警告: 没有人工处理的请求，平均延迟 = 0")
        return
    
    print(f"\n2. waiting_time 计算公式:")
    print("   waiting_time = start_service_time - arrival_time")
    print("   (纯排队等待时间，不包括服务时间)")
    
    # 按到达时间排序
    human_requests_sorted = sorted(human_requests, key=lambda r: r.arrival_time)
    
    print(f"\n3. 各请求的延迟计算 (按到达时间排序，共{len(human_requests)}条):")
    print(f"{'请求ID':<8} {'风险':<6} {'到达(s)':<10} {'开始(s)':<10} {'等待(s)':<10} {'服务(s)':<10} {'完成(s)':<10} {'延迟(ms)':<12}")
    print("-"*110)
    
    delays_ms = []
    for r in human_requests_sorted:  # 显示所有请求，按到达时间排序
        waiting_time_sec = r.waiting_time
        waiting_time_ms = waiting_time_sec * 1000
        delays_ms.append(waiting_time_ms)
        
        # 计算完成时间
        completion_time = r.start_service_time + r.service_time if r.start_service_time and r.service_time else 0
        
        print(f"{r.id:<8} {r.true_risk_level.upper():<6} {r.arrival_time:<10.1f} "
              f"{r.start_service_time:<10.1f} {waiting_time_sec:<10.2f} "
              f"{r.service_time:<10.1f} {completion_time:<10.1f} {waiting_time_ms:<12.1f}")
    
    # 按风险等级分组统计
    print(f"\n4. 按风险等级分组的延迟统计:")
    for level in ["high", "mid", "low"]:
        level_requests = [r for r in human_requests if r.true_risk_level == level]
        if level_requests:
            level_delays = [r.waiting_time * 1000 for r in level_requests]
            avg_level_delay = sum(level_delays) / len(level_delays)
            print(f"   {level.upper():<6}: {len(level_requests):>3}条请求, "
                  f"平均延迟={avg_level_delay:>8.1f}ms, "
                  f"范围=[{min(level_delays):.1f}, {max(level_delays):.1f}]ms")
    
    # 计算均值的过程
    print(f"\n5. 平均延迟计算过程:")
    all_delays_ms = [r.waiting_time * 1000 for r in human_requests]
    
    print(f"   公式: avg_delay = mean(waiting_time × 1000 for all human requests)")
    print(f"   ")
    print(f"   步骤:")
    print(f"   - 人工请求数 n = {len(all_delays_ms)}")
    print(f"   - 延迟总和 Σ = {sum(all_delays_ms):.1f} ms")
    print(f"   - 平均延迟 = {sum(all_delays_ms):.1f} / {len(all_delays_ms)} = {result.avg_queueing_delay:.1f} ms")
    
    print(f"\n6. 最大延迟:")
    print(f"   max_delay = {result.max_queueing_delay:.1f} ms")
    
    # 成本计算
    from exp.scheduling_simulation.config import COST_CONFIG
    latency_cost_per_ms = COST_CONFIG['latency_cost_per_ms']
    total_latency_cost = sum(d * latency_cost_per_ms for d in all_delays_ms)
    
    print(f"\n7. 延迟成本计算:")
    print(f"   延迟成本系数: {latency_cost_per_ms} 元/ms")
    print(f"   总延迟成本 = Σ(delay_ms × {latency_cost_per_ms})")
    print(f"            = {total_latency_cost:.2f} 元")
    print(f"   平均每条人工请求的延迟成本 = {total_latency_cost/len(all_delays_ms):.4f} 元")
    
    print("="*80)


def main():
    print("="*80)
    print("完整对比实验：普通队列 vs 优先级队列")
    print("="*80)
    
    # 加载数据（使用带固定错误概率的数据集）
    test_data = load_test_data(
        # "./data/validation/validation_set_1000_eval_with_prob.jsonl",
        "./data/validation/risk_dataset.jsonl",
        num_requests=300
    )
    
    print(f"\n测试数据: {len(test_data)} 条")
    high_count = sum(1 for d in test_data if d["true_level"] == "high")
    mid_count = sum(1 for d in test_data if d["true_level"] == "mid")
    low_count = sum(1 for d in test_data if d["true_level"] == "low")
    # 计算实际的错误概率统计
    high_probs = [d["llm_error_prob"] for d in test_data if d["true_level"] == "high" and d.get("llm_error_prob") is not None]
    mid_probs = [d["llm_error_prob"] for d in test_data if d["true_level"] == "mid" and d.get("llm_error_prob") is not None]
    low_probs = [d["llm_error_prob"] for d in test_data if d["true_level"] == "low" and d.get("llm_error_prob") is not None]
    
    high_prob_str = f"{sum(high_probs)/len(high_probs):.2f}" if high_probs else "N/A"
    mid_prob_str = f"{sum(mid_probs)/len(mid_probs):.2f}" if mid_probs else "N/A"
    low_prob_str = f"{sum(low_probs)/len(low_probs):.2f}" if low_probs else "N/A"
    
    print(f"  High风险: {high_count}条 (错误代价2000元, P(error)={high_prob_str})")
    print(f"  Mid风险: {mid_count}条 (错误代价10元, P(error)={mid_prob_str})")
    print(f"  Low风险: {low_count}条 (错误代价1元, P(error)={low_prob_str})")
    print(f"\n仿真时长: 2小时 (含早盘、午盘高峰)")
    
    # 定义所有对比策略
    strategies = [
        ("LLM-Only", LLMOnlyRouter(), "default"),
        ("Vanilla L2D", VanillaL2DRouter(threshold=0.55
        ), "default"),
        ("Static Cost", StaticCostRouter(human_cost_threshold=50.0), "default"),
        ("Dynamic (原版)", DynamicQueueAwareRouter(), "default"),
        # ("Dynamic Priority (优先级队列)", DynamicPriorityQueueRouter(), "priority"),
    ]
    
    results = []
    
    print("\n" + "="*80)
    print("开始实验")
    print("="*80)
    
    for name, router, queue_type in strategies:
        print(f"\n{'-'*80}")
        print(f"策略: {name}")
        print(f"队列类型: {queue_type}")
        print(f"{'-'*80}")
        
        result, simulator = run_experiment(router, test_data, queue_type=queue_type, seed=5)
        results.append(result)
        
        # 打印关键指标
        print(f"总期望成本: {result.expected_total_cost:.2f}元")
        print(f"灾难性错误率: {result.catastrophic_error_rate:.4f}")
        print(f"平均排队延迟: {result.avg_queueing_delay:.1f}ms")
        print(f"人工转交率: {result.deferral_rate*100:.1f}%")
        
        # High风险转接详情
        high_to_human = result.risk_distribution.get("high", {}).get("human", 0)
        high_to_llm = result.risk_distribution.get("high", {}).get("llm", 0)
        high_total = high_to_human + high_to_llm
        if high_total > 0:
            high_deferral_rate = high_to_human / high_total
            print(f"High风险转接率: {high_deferral_rate*100:.1f}% ({high_to_human}/{high_total})")
    
    # 生成对比表格
    print("\n" + "="*80)
    print("对比结果")
    print("="*80)
    
    # 自定义对比表格（包含High转接率和整体转交率）
    print("\n策略对比表:")
    print(f"{'策略':<35} {'ETC':<12} {'CER':<10} {'延迟(ms)':<12} {'High转接率':<12} {'转交率':<10}")
    print("-"*100)
    
    for result in results:
        high_to_human = result.risk_distribution.get("high", {}).get("human", 0)
        high_to_llm = result.risk_distribution.get("high", {}).get("llm", 0)
        high_total = high_to_human + high_to_llm
        high_deferral = (high_to_human / high_total * 100) if high_total > 0 else 0
        
        print(f"{result.router_name:<35} {result.expected_total_cost:<12.2f} "
              f"{result.catastrophic_error_rate:<10.4f} {result.avg_queueing_delay:<12.1f} "
              f"{high_deferral:<12.1f}% {result.deferral_rate*100:<10.1f}%")
    
    # 保存结果
    output_dir = Path("./output/scheduling_simulation_priority")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_results_to_json(results, str(output_dir / "comparison_results.json"))
    
    print(f"\n{'='*80}")
    print("实验完成！")
    print(f"结果保存: {output_dir}")
    print(f"{'='*80}")
    
    # 关键发现
    print("\n关键发现:")
    print("-"*80)
    
    # 找到Dynamic和Dynamic Priority
    dynamic_result = next((r for r in results if "Dynamic (原版)" in r.router_name), None)
    priority_result = next((r for r in results if "Priority" in r.router_name), None)
    
    if dynamic_result and priority_result:
        dyn_high_human = dynamic_result.risk_distribution.get("high", {}).get("human", 0)
        dyn_high_total = dyn_high_human + dynamic_result.risk_distribution.get("high", {}).get("llm", 0)
        prio_high_human = priority_result.risk_distribution.get("high", {}).get("human", 0)
        prio_high_total = prio_high_human + priority_result.risk_distribution.get("high", {}).get("llm", 0)
        
        print(f"\n1. High风险转接率:")
        print(f"   普通Dynamic: {dyn_high_human}/{dyn_high_total} = {dyn_high_human/dyn_high_total*100:.1f}%")
        print(f"   优先级队列: {prio_high_human}/{prio_high_total} = {prio_high_human/prio_high_total*100:.1f}%")
        
        print(f"\n2. 灾难性错误率:")
        print(f"   普通Dynamic: {dynamic_result.catastrophic_error_rate:.4f}")
        print(f"   优先级队列: {priority_result.catastrophic_error_rate:.4f}")
        
        print(f"\n3. 平均延迟:")
        print(f"   普通Dynamic: {dynamic_result.avg_queueing_delay:.1f}ms")
        print(f"   优先级队列: {priority_result.avg_queueing_delay:.1f}ms")


if __name__ == "__main__":
    main()
