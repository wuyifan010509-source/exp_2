"""
α值超参数搜索脚本
Alpha Hyperparameter Search for Asymmetric Loss

目标：找到最优的 ALPHA_UNDERPREDICT 值，使得：
1. SLM 总期望代价 < Semantic Routing (923.28)
2. 漏报数尽量低

策略：
- 尝试 α ∈ [5, 7, 10, 12, 15]
- 对每个α值训练模型，在验证集上评估
- 选择总期望代价最低的模型
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_training(alpha: float, output_dir: str, data_path: str, epochs: int = 3) -> dict:
    """
    运行单次训练

    Args:
        alpha: ALPHA_UNDERPREDICT 值
        output_dir: 输出目录
        data_path: 训练数据路径
        epochs: 训练轮数

    Returns:
        训练结果字典
    """
    print(f"\n{'='*70}")
    print(f"训练 α = {alpha}")
    print(f"{'='*70}")

    # 修改配置文件中的 ALPHA_UNDERPREDICT
    config_path = PROJECT_ROOT / "exp" / "slm_distillation" / "config.py"
    with open(config_path, "r") as f:
        config_content = f.read()

    # 替换 ALPHA_UNDERPREDICT 值
    import re
    new_config = re.sub(
        r"ALPHA_UNDERPREDICT\s*=\s*[\d.]+",
        f"ALPHA_UNDERPREDICT = {alpha}",
        config_content
    )
    with open(config_path, "w") as f:
        f.write(new_config)

    # 运行训练
    cmd = [
        "uv", "run", "python", "-m", "exp.slm_distillation.train",
        f"--data_path", data_path,
        f"--output_dir", output_dir,
        f"--epochs", str(epochs),
    ]

    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"训练失败: {result.stderr}")
        return {"alpha": alpha, "success": False, "error": result.stderr}

    return {"alpha": alpha, "success": True, "output_dir": output_dir}


def run_evaluation(adapter_path: str, test_file: str) -> dict:
    """
    运行评估

    Args:
        adapter_path: 模型路径
        test_file: 测试集路径

    Returns:
        评估结果字典
    """
    print(f"\n{'='*70}")
    print(f"评估模型: {adapter_path}")
    print(f"{'='*70}")

    cmd = [
        "uv", "run", "python", "-m", "exp.evaluation.evaluate",
        "--test_file", test_file,
        "--methods", "slm",
        "--slm_adapter", adapter_path,
    ]

    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

    # 解析输出中的关键指标
    output = result.stdout + result.stderr

    metrics = {
        "accuracy": None,
        "recall_high": None,
        "underpredict_count": None,
        "overpredict_count": None,
        "total_expected_cost": None,
        "avg_latency_ms": None,
    }

    import re

    # 解析准确率
    match = re.search(r"总准确率:\s*([\d.]+)", output)
    if match:
        metrics["accuracy"] = float(match.group(1))

    # 解析 High Recall
    match = re.search(r"Recall\(High\):\s*([\d.]+)", output)
    if match:
        metrics["recall_high"] = float(match.group(1))

    # 解析漏报数
    match = re.search(r"漏报数:\s*(\d+)", output)
    if match:
        metrics["underpredict_count"] = int(match.group(1))

    # 解析误报数
    match = re.search(r"误报数:\s*(\d+)", output)
    if match:
        metrics["overpredict_count"] = int(match.group(1))

    # 解析总期望代价
    match = re.search(r"★ 总期望代价:\s*([\d.]+)", output)
    if match:
        metrics["total_expected_cost"] = float(match.group(1))

    # 解析平均延迟
    match = re.search(r"平均延迟:\s*([\d.]+)", output)
    if match:
        metrics["avg_latency_ms"] = float(match.group(1))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="α值超参数搜索")
    parser.add_argument(
        "--alpha_values",
        type=float,
        nargs="+",
        default=[5.0, 7.0, 10.0, 12.0, 15.0],
        help="要搜索的α值列表",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./exp/slm_distillation/training_data.jsonl",
        help="训练数据路径",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="./exp/data/validation_set_1000_eval.jsonl",
        help="测试集路径",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="训练轮数（快速搜索用较少轮数）",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="跳过训练，只评估已有模型",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("α值超参数搜索")
    print("=" * 70)
    print(f"搜索空间: {args.alpha_values}")
    print(f"训练数据: {args.data_path}")
    print(f"测试集: {args.test_file}")
    print(f"训练轮数: {args.epochs}")

    # 目标：低于 Semantic Routing 的 923.28
    TARGET_COST = 923.28

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for alpha in args.alpha_values:
        output_dir = f"./output/alpha_search/alpha_{alpha}_{timestamp}"

        if not args.skip_training:
            # 训练
            train_result = run_training(
                alpha=alpha,
                output_dir=output_dir,
                data_path=args.data_path,
                epochs=args.epochs,
            )

            if not train_result["success"]:
                results.append({"alpha": alpha, "error": train_result["error"]})
                continue

        # 评估
        eval_metrics = run_evaluation(
            adapter_path=output_dir,
            test_file=args.test_file,
        )

        result = {
            "alpha": alpha,
            "output_dir": output_dir,
            **eval_metrics,
        }
        results.append(result)

        # 打印当前结果
        print(f"\n--- α={alpha} 结果 ---")
        for k, v in eval_metrics.items():
            if v is not None:
                print(f"  {k}: {v}")

    # 汇总结果
    print("\n" + "=" * 70)
    print("★ 搜索结果汇总 ★")
    print("=" * 70)

    # 按 total_expected_cost 排序
    valid_results = [r for r in results if r.get("total_expected_cost") is not None]
    valid_results.sort(key=lambda x: x["total_expected_cost"])

    print(f"\n{'α值':<8} {'总代价':<12} {'准确率':<10} {'High Recall':<12} {'漏报数':<8} {'误报数':<8}")
    print("-" * 70)

    for r in valid_results:
        cost_flag = "✅" if r["total_expected_cost"] < TARGET_COST else "❌"
        print(f"{r['alpha']:<8.1f} {r['total_expected_cost']:<12.2f} {cost_flag} "
              f"{r.get('accuracy', 0):<10.4f} {r.get('recall_high', 0):<12.4f} "
              f"{r.get('underpredict_count', 'N/A'):<8} {r.get('overpredict_count', 'N/A'):<8}")

    # 找到最优α值
    if valid_results:
        best = valid_results[0]
        print(f"\n{'='*70}")
        print(f"★ 最优结果 ★")
        print(f"{'='*70}")
        print(f"最优 α 值: {best['alpha']}")
        print(f"总期望代价: {best['total_expected_cost']:.2f}")
        print(f"是否优于 Semantic Routing: {'是 ✅' if best['total_expected_cost'] < TARGET_COST else '否 ❌'}")
        print(f"模型路径: {best['output_dir']}")

        # 保存结果
        result_file = PROJECT_ROOT / "output" / f"alpha_search_results_{timestamp}.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {result_file}")

    return results


if __name__ == "__main__":
    main()
