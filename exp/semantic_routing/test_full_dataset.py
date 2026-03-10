"""
语义路由测试脚本 - 使用完整数据集
支持从 JSONL 文件加载数据进行测试
"""

from .router import SemanticRouter
import time
import json
import os


def load_data_from_file(filepath: str):
    """从 JSONL 文件加载数据"""
    queries = []
    ground_truth = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            queries.append(data['query'])
            
            # 根据 cost_label 确定 ground truth 等级
            cost = data['cost_label']
            if cost == 1:
                ground_truth.append('low')
            elif cost == 50:
                ground_truth.append('mid')
            elif cost == 1000:
                ground_truth.append('high')
            else:
                ground_truth.append('unknown')
    
    return queries, ground_truth


def run_tests_with_file(filepath: str = None, max_samples: int = None):
    """
    使用文件数据进行测试
    
    Args:
        filepath: JSONL 文件路径，默认为 final_labeled_data.jsonl
        max_samples: 最大测试样本数，None 表示全部
    """
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(__file__), 
            '..', 'data', 'data_labeling', 'final_labeled_data.jsonl'
        )
    
    print("=" * 70)
    print(f"语义路由完整测试 - 使用数据集")
    print("=" * 70)
    
    # 加载数据
    print(f"\n[加载数据] 从 {filepath}")
    queries, ground_truth = load_data_from_file(filepath)
    
    if max_samples and max_samples < len(queries):
        queries = queries[:max_samples]
        ground_truth = ground_truth[:max_samples]
    
    total = len(queries)
    print(f"  加载 {total} 条测试数据")
    
    # 统计 ground truth 分布
    gt_counts = {'low': 0, 'mid': 0, 'high': 0}
    for gt in ground_truth:
        if gt in gt_counts:
            gt_counts[gt] += 1
    print(f"  Ground Truth 分布: Low={gt_counts['low']}, Mid={gt_counts['mid']}, High={gt_counts['high']}")
    
    # 初始化路由器
    print("\n[初始化] 加载模型...")
    router = SemanticRouter(tau_high=0.7)
    
    print("\n" + "-" * 70)
    print(f"开始测试 (高危阈值 τ_high=0.7, 共 {total} 条)")
    print("-" * 70)
    
    results = []
    latencies = []
    
    for i, (query, gt) in enumerate(zip(queries, ground_truth), 1):
        start = time.time()
        level, cost, max_sim, details = router.route(query)
        latency = (time.time() - start) * 1000  # ms
        
        latencies.append(latency)
        results.append({
            "id": i,
            "query": query,
            "pred_level": level,
            "gt_level": gt,
            "cost": cost,
            "max_sim": max_sim,
            "latency_ms": latency,
            "flag": details["flag"],
            "correct": level == gt
        })
        
        # 每 100 条显示一次进度
        if i % 100 == 0 or i <= 10 or i == total:
            flag_str = "⚠️ " if details["flag"] == "ASYMMETRIC_FLAG" else "   "
            correct_mark = "✓" if level == gt else "✗"
            print(f"{flag_str}[{i:04d}/{total}] {query[:30]:30s} → {level:6s} | "
                  f"Cost={cost:4d} | Sim={max_sim:.3f} | {latency:5.2f}ms | {correct_mark}")
    
    # 统计报告
    print("\n" + "=" * 70)
    print("详细测试报告")
    print("=" * 70)
    
    # 统计各等级分布
    level_counts = {"low": 0, "mid": 0, "high": 0}
    asymmetric_count = 0
    for r in results:
        level_counts[r["pred_level"]] += 1
        if r["flag"] == "ASYMMETRIC_FLAG":
            asymmetric_count += 1
    
    print(f"\n预测分布:")
    print(f"  • Low  (Cost=1):    {level_counts['low']:4d} 条")
    print(f"  • Mid  (Cost=50):   {level_counts['mid']:4d} 条")
    print(f"  • High (Cost=1000): {level_counts['high']:4d} 条")
    print(f"  • 非对称拦截: {asymmetric_count} 条")
    
    # 延迟统计
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
    
    print(f"\n延迟统计:")
    print(f"  • 平均: {avg_latency:.2f} ms")
    print(f"  • P99:  {p99_latency:.2f} ms")
    print(f"  • 最大: {max_latency:.2f} ms")
    print(f"  • 最小: {min_latency:.2f} ms")
    print(f"  • 满足 <10ms 要求: {'✅' if avg_latency < 10 else '❌'}")
    
    # 准确率统计
    print("\n" + "-" * 70)
    print("准确率分析:")
    
    # 按类别统计
    for level in ['low', 'mid', 'high']:
        level_results = [r for r in results if r['gt_level'] == level]
        if level_results:
            correct = sum(1 for r in level_results if r['correct'])
            total_level = len(level_results)
            print(f"  • {level.upper():5s} 准确率: {correct:4d}/{total_level:4d} ({correct/total_level*100:5.1f}%)")
    
    # 总体准确率
    total_correct = sum(1 for r in results if r['correct'])
    print(f"\n  • 总体准确率: {total_correct}/{total} ({total_correct/total*100:.1f}%)")
    
    # 高危召回率（关键指标）
    high_results = [r for r in results if r['gt_level'] == 'high']
    high_correct = sum(1 for r in high_results if r['correct'])
    high_recall = high_correct / len(high_results) * 100 if high_results else 0
    print(f"\n  ★ 高危召回率: {high_recall:.1f}% (目标 >95%)")
    
    # 混淆矩阵
    print("\n" + "-" * 70)
    print("混淆矩阵 (行=真实, 列=预测):")
    print(f"{'':>10s} {'Low':>8s} {'Mid':>8s} {'High':>8s}")
    for gt_level in ['low', 'mid', 'high']:
        row = []
        for pred_level in ['low', 'mid', 'high']:
            count = sum(1 for r in results if r['gt_level'] == gt_level and r['pred_level'] == pred_level)
            row.append(count)
        print(f"{gt_level.upper():>10s} {row[0]:>8d} {row[1]:>8d} {row[2]:>8d}")
    
    # 错误案例分析
    print("\n" + "-" * 70)
    print("错误案例分析 (显示前20条):")
    errors = [r for r in results if not r['correct']]
    print(f"共 {len(errors)} 条错误")
    
    for i, r in enumerate(errors[:20], 1):
        print(f"  [{i:02d}] '{r['query'][:40]:40s}' → 预测:{r['pred_level']:6s} 真实:{r['gt_level']:6s} | "
              f"Sim={r['max_sim']:.3f}")
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='语义路由完整测试')
    parser.add_argument('--file', type=str, default=None, help='测试数据文件路径')
    parser.add_argument('--max-samples', type=int, default=None, help='最大测试样本数')
    args = parser.parse_args()
    
    run_tests_with_file(filepath=args.file, max_samples=args.max_samples)
