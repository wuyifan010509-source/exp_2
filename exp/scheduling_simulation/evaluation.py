"""
评估指标计算与可视化
Evaluation Metrics and Visualization
"""

import numpy as np
import json
from typing import List, Dict, Tuple
from dataclasses import asdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

from .simulator import SimulationResult


def compute_cost_sensitive_accuracy(results: List[SimulationResult]) -> Dict:
    """
    计算代价敏感准确率 (Cost-Sensitive Accuracy, CSA)
    按业务风险等级加权后的准确率
    """
    # 权重：高危问题答对得分远高于常规问题
    weights = {"low": 1, "mid": 10, "high": 100}
    
    csa_scores = {}
    for result in results:
        weighted_correct = 0
        weighted_total = 0
        
        for r in result.completed_requests:
            weight = weights.get(r.true_risk_level, 1)
            weighted_total += weight
            
            # 判断是否处理正确
            if r.assigned_to == "human":
                # 交给人类 = 正确（不会答错）
                weighted_correct += weight
            else:
                # LLM处理：检查是否答错
                if r.actual_cost < 100:  # 没有产生错误代价 = 答对
                    weighted_correct += weight
        
        csa = weighted_correct / weighted_total if weighted_total > 0 else 0
        csa_scores[result.router_name] = csa
    
    return csa_scores


def generate_comparison_table(results: List[SimulationResult]) -> str:
    """
    生成对比表格（Markdown格式）
    """
    # 计算CSA
    csa_scores = compute_cost_sensitive_accuracy(results)
    
    lines = []
    lines.append("\n" + "="*100)
    lines.append("四种调度策略对比结果")
    lines.append("="*100)
    
    # 表头
    lines.append(f"\n{'指标':<25} | {'LLM-Only':<15} | {'Vanilla L2D':<15} | {'Static Cost':<15} | {'Dynamic (Ours)':<15}")
    lines.append("-" * 100)
    
    # 数据行
    metrics = [
        ("★ ETC (总期望成本)", lambda r: f"{r.expected_total_cost:.2f}"),
        ("CSA (代价敏感准确率)", lambda r: f"{csa_scores.get(r.router_name, 0):.4f}"),
        ("CER (灾难性错误率)", lambda r: f"{r.catastrophic_error_rate:.4f}"),
        ("平均排队延迟(ms)", lambda r: f"{r.avg_queueing_delay:.1f}"),
        ("SLA违约率", lambda r: f"{r.sla_violation_rate:.4f}"),
        ("服务器利用率", lambda r: f"{r.server_utilization:.4f}"),
        ("人工转交率", lambda r: f"{r.deferral_rate:.4f}"),
    ]
    
    for metric_name, metric_func in metrics:
        row = f"{metric_name:<25} |"
        for result in results:
            row += f" {metric_func(result):<14} |"
        lines.append(row)
    
    lines.append("\n" + "="*100)
    
    # 结论
    best_idx = min(range(len(results)), key=lambda i: results[i].expected_total_cost)
    lines.append(f"\n★ 最优策略: {results[best_idx].router_name}")
    lines.append(f"  总期望成本最低: {results[best_idx].expected_total_cost:.2f} 元")
    lines.append(f"  相比LLM-Only节省: {(results[0].expected_total_cost - results[best_idx].expected_total_cost):.2f} 元")
    lines.append("="*100)
    
    return "\n".join(lines)


def plot_cost_delay_tradeoff(results: List[SimulationResult], output_path: str = "cost_delay_tradeoff.png"):
    """
    绘制成本-延迟权衡曲线 (Pareto Frontier)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red', 'orange', 'blue', 'green']
    markers = ['x', 's', 'o', '^']
    
    for i, result in enumerate(results):
        ax.scatter(
            result.avg_queueing_delay,
            result.expected_total_cost,
            s=200,
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=result.router_name,
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5
        )
    
    ax.set_xlabel("Average Queueing Delay (ms)", fontsize=12)
    ax.set_ylabel("Expected Total Cost (¥)", fontsize=12)
    ax.set_title("Cost-Delay Trade-off Curve", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 添加说明文字
    ax.text(
        0.05, 0.95,
        "Lower-left is better\n(Lower cost + Lower delay)",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"成本-延迟权衡曲线已保存: {output_path}")


def plot_routing_behavior(results: List[SimulationResult], output_path: str = "routing_behavior.png"):
    """
    绘制不同风险等级的转交行为
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Routing Behavior Analysis", fontsize=16, fontweight='bold')
    
    risk_levels = ["low", "mid", "high"]
    colors = ['green', 'orange', 'red']
    
    # 图1: 各策略的人工转交率对比
    ax = axes[0, 0]
    x = np.arange(len(results))
    width = 0.6
    deferral_rates = [r.deferral_rate for r in results]
    bars = ax.bar(x, deferral_rates, width, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_ylabel("Deferral Rate", fontsize=11)
    ax.set_title("Overall Human Deferral Rate", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([r.router_name.split()[0] for r in results], rotation=15, ha='right')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, rate in zip(bars, deferral_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # 图2: 按风险等级的转交分布
    ax = axes[0, 1]
    deferral_by_risk = {level: [] for level in risk_levels}
    
    for result in results:
        for level in risk_levels:
            counts = result.risk_distribution.get(level, {"llm": 0, "human": 0})
            total = counts["llm"] + counts["human"]
            deferral_rate = counts["human"] / total if total > 0 else 0
            deferral_by_risk[level].append(deferral_rate)
    
    x = np.arange(len(results))
    width = 0.25
    
    for i, level in enumerate(risk_levels):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, deferral_by_risk[level], width, 
                     label=level.upper(), color=colors[i], alpha=0.7)
    
    ax.set_ylabel("Deferral Rate", fontsize=11)
    ax.set_title("Deferral Rate by Risk Level", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([r.router_name.split()[0] for r in results], rotation=15, ha='right')
    ax.legend(fontsize=9)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # 图3: 成本分解
    ax = axes[1, 0]
    cost_components = ['Error Cost', 'Latency Cost', 'Human Cost']
    x = np.arange(len(results))
    width = 0.25
    
    for i, component in enumerate(cost_components):
        offset = (i - 1) * width
        if component == 'Error Cost':
            values = [r.total_error_cost for r in results]
        elif component == 'Latency Cost':
            values = [r.total_latency_cost for r in results]
        else:
            values = [r.total_human_cost for r in results]
        
        bars = ax.bar(x + offset, values, width, label=component, alpha=0.7)
    
    ax.set_ylabel("Cost (¥)", fontsize=11)
    ax.set_title("Cost Breakdown", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([r.router_name.split()[0] for r in results], rotation=15, ha='right')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # 图4: 灾难性错误率对比
    ax = axes[1, 1]
    cer_values = [r.catastrophic_error_rate for r in results]
    bars = ax.bar(x, cer_values, width=0.6, color='crimson', alpha=0.7, edgecolor='black')
    ax.set_ylabel("Catastrophic Error Rate", fontsize=11)
    ax.set_title("Catastrophic Error Rate (CER)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([r.router_name.split()[0] for r in results], rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, rate in zip(bars, cer_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"调度行为分析图已保存: {output_path}")


def plot_traffic_pattern_over_time(output_path: str = "traffic_pattern.png"):
    """
    绘制流量模式（泊松到达率）
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # 4小时仿真周期
    times = np.linspace(0, 4*3600, 1000)  # 秒
    lambdas = []
    
    for t in times:
        # 0-30min: 10 req/s, 30-120min: 2 req/s, 120-150min: 6 req/s, 150-240min: 2 req/s
        minutes = t / 60
        if minutes < 30:
            lambdas.append(10.0)
        elif minutes < 120:
            lambdas.append(2.0)
        elif minutes < 150:
            lambdas.append(6.0)
        else:
            lambdas.append(2.0)
    
    ax.fill_between(times/60, lambdas, alpha=0.3, color='steelblue')
    ax.plot(times/60, lambdas, color='steelblue', linewidth=2)
    
    # 标注高峰期
    ax.axvspan(0, 30, alpha=0.1, color='red', label='Morning Peak')
    ax.axvspan(120, 150, alpha=0.1, color='orange', label='Afternoon Peak')
    
    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Arrival Rate λ (req/s)", fontsize=12)
    ax.set_title("Traffic Pattern with Burst Injection", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 240])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"流量模式图已保存: {output_path}")


def save_results_to_json(results: List[SimulationResult], output_path: str = "simulation_results.json"):
    """保存结果到JSON"""
    data = []
    for result in results:
        result_dict = asdict(result)
        # 移除completed_requests（太大）
        result_dict.pop('completed_requests', None)
        data.append(result_dict)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到JSON: {output_path}")
