"""
生成混淆矩阵详细分析，包含具体的query例子
"""
import json
import sys
from pathlib import Path
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exp.evaluation.evaluate import (
    SemanticRoutingEvaluator, 
    SLMDistillationEvaluator,
    LLMBaselineEvaluator,
    compute_error_cost
)

# 加载测试数据
test_data_path = PROJECT_ROOT / "exp" / "slm_distillation" / "test_split.jsonl"
test_data = []
with open(test_data_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            test_data.append(json.loads(line))

print("=" * 80)
print("混淆矩阵详细分析 - 包含具体Query例子")
print("=" * 80)

# 定义评估器
methods = [
    SemanticRoutingEvaluator(),
    SLMDistillationEvaluator(),
    LLMBaselineEvaluator(),
]

results_by_method = {}

for method in methods:
    method_name = method.get_name()
    print(f"\n{'='*80}")
    print(f"方法: {method_name}")
    print(f"{'='*80}")
    
    # 收集每个样本的预测结果
    confusion_examples = {
        ('low', 'low'): [], ('low', 'mid'): [], ('low', 'high'): [],
        ('mid', 'low'): [], ('mid', 'mid'): [], ('mid', 'high'): [],
        ('high', 'low'): [], ('high', 'mid'): [], ('high', 'high'): [],
    }
    
    for item in test_data[:99]:  # 前99条
        pred = method.predict(item["query"])
        true_level = item.get("true_level") or item.get("level")
        pred_level = pred["predicted_level"]
        
        key = (true_level, pred_level)
        if len(confusion_examples[key]) < 3:  # 最多保存3个例子
            confusion_examples[key].append({
                "query": item["query"],
                "true": true_level,
                "pred": pred_level,
            })
    
    results_by_method[method_name] = confusion_examples
    
    # 打印混淆矩阵
    print("\n混淆矩阵统计:")
    print(f"{'True/Pred':<12} {'Low':<8} {'Mid':<8} {'High':<8}")
    print("-" * 40)
    for true in ['low', 'mid', 'high']:
        row = []
        for pred in ['low', 'mid', 'high']:
            count = sum(1 for item in test_data[:99] 
                       if (item.get('true_level') or item.get('level')) == true and 
                       method.predict(item['query'])['predicted_level'] == pred)
            row.append(count)
        print(f"{true.capitalize():<12} {row[0]:<8} {row[1]:<8} {row[2]:<8}")
    
    # 打印关键错误例子
    print("\n" + "-"*80)
    print("【关键错误例子 - 漏报 High→Low/Mid】")
    print("-"*80)
    for key in [('high', 'low'), ('high', 'mid')]:
        true, pred = key
        examples = confusion_examples[key]
        if examples:
            print(f"\n{true.upper()} → {pred.upper()} ({len(examples)} 例):")
            for i, ex in enumerate(examples, 1):
                error_cost = compute_error_cost(true, pred)
                print(f"  {i}. \"{ex['query']}\"")
                print(f"     真实: {true}, 预测: {pred}, 错误代价: {error_cost:.0f}元")
    
    print("\n" + "-"*80)
    print("【正确预测例子 - High→High】")
    print("-"*80)
    examples = confusion_examples[('high', 'high')]
    if examples:
        print(f"\nHIGH → HIGH ({len(examples)} 例):")
        for i, ex in enumerate(examples[:3], 1):
            print(f"  {i}. \"{ex['query']}\"")

# 保存为Markdown文件
output_path = PROJECT_ROOT / "exp" / "reports" / "CONFUSION_MATRIX_EXAMPLES.md"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("# 混淆矩阵详细分析 - 包含具体Query例子\n\n")
    f.write("> 基于99条测试样本的详细分析\n\n")
    
    for method_name, confusion_examples in results_by_method.items():
        f.write(f"## {method_name}\n\n")
        
        # 混淆矩阵表格
        f.write("### 混淆矩阵\n\n")
        f.write("| True \\ Pred | Low | Mid | High |\n")
        f.write("|:------------|:---:|:---:|:----:|\n")
        for true in ['low', 'mid', 'high']:
            row = []
            for pred in ['low', 'mid', 'high']:
                count = sum(1 for item in test_data[:99] 
                           if item['true_level'] == true and 
                           method.predict(item['query'])['predicted_level'] == pred)
                row.append(count)
            f.write(f"| **{true.capitalize()}** | {row[0]} | {row[1]} | {row[2]} |\n")
        
        # 关键例子
        f.write("\n### 漏报案例 (High→Low/Mid) ⚠️\n\n")
        for key in [('high', 'low'), ('high', 'mid')]:
            true, pred = key
            examples = confusion_examples[key]
            if examples:
                for i, ex in enumerate(examples[:3], 1):
                    error_cost = compute_error_cost(true, pred)
                    f.write(f"**{i}. \"{ex['query']}\"**  \n")
                    f.write(f"- 真实: `{true}` → 预测: `{pred}`  \n")
                    f.write(f"- 错误代价: **{error_cost:.0f}元**  \n\n")
        
        if not confusion_examples[('high', 'low')] and not confusion_examples[('high', 'mid')]:
            f.write("✅ **无漏报！**\n\n")
        
        f.write("\n---\n\n")

print(f"\n\n详细分析已保存: {output_path}")
