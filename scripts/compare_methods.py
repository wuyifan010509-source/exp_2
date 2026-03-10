"""
Detailed comparison of Semantic Routing vs SLM Distillation
Analyzes the trade-off between latency, accuracy, and cost
"""

from evaluation.evaluate import load_test_data, SemanticRoutingEvaluator, SLMDistillationEvaluator, compute_error_cost, COST_MAP, C_HUMAN, LATENCY_COST_COEFF

test_data = load_test_data('data/validation_set_1000_eval.jsonl')

# Initialize evaluators
routing_eval = SemanticRoutingEvaluator()
slm_eval = SLMDistillationEvaluator(adapter_path='output/slm_distillation/checkpoint-31')

print("=" * 70)
print("Detailed Method Comparison")
print("=" * 70)

# Track statistics by level for each method
def evaluate_detailed(evaluator, name):
    from collections import defaultdict

    total_latency = 0
    total_error_cost = 0
    confusion = defaultdict(int)

    for item in test_data:
        pred = evaluator.predict(item['query'])
        true_level = item['true_level']
        pred_level = pred['predicted_level']
        latency = pred['latency_ms']

        error_cost = compute_error_cost(true_level, pred_level)
        total_latency += latency
        total_error_cost += error_cost
        confusion[(true_level, pred_level)] += 1

    n = len(test_data)
    avg_latency = total_latency / n
    avg_error_cost = total_error_cost / n
    avg_latency_cost = avg_latency * LATENCY_COST_COEFF
    total_expected_cost = avg_error_cost + avg_latency_cost

    # Per-level accuracy
    level_accuracy = {}
    for level in ['low', 'mid', 'high']:
        correct = confusion[(level, level)]
        total = sum(confusion[(level, p)] for p in ['low', 'mid', 'high'])
        level_accuracy[level] = correct / total if total > 0 else 0

    print(f"\n{name}")
    print("-" * 70)

    # Confusion matrix
    print("Confusion Matrix:")
    for true in ['low', 'mid', 'high']:
        row = []
        for pred in ['low', 'mid', 'high']:
            count = confusion[(true, pred)]
            pct = count / sum(confusion[(true, p)] for p in ['low', 'mid', 'high']) * 100
            row.append(f"{count:3d}({pct:5.1f}%)")
        print(f"  {true.upper():>6}: {' '.join(row)}")

    # Accuracy per level
    print(f"\nAccuracy per level:")
    for level in ['low', 'mid', 'high']:
        acc = level_accuracy[level]
        print(f"  {level.upper():>6}: {acc:.4f}")

    # Cost breakdown
    print(f"\nCost breakdown (per query):")
    print(f"  Avg latency:        {avg_latency:.3f} ms")
    print(f"  Avg latency cost:   {avg_latency_cost:.4f}")
    print(f"  Avg error cost:     {avg_error_cost:.2f}")
    print(f"  Total expected cost: {total_expected_cost:.4f}")

    # Over-prediction vs Under-prediction
    underpredict = sum(confusion[(t, p)] for t, p in confusion.keys()
                      if {'low':0,'mid':1,'high':2}[p] < {'low':0,'mid':1,'high':2}[t])
    overpredict = sum(confusion[(t, p)] for t, p in confusion.keys()
                     if {'low':0,'mid':1,'high':2}[p] > {'low':0,'mid':1,'high':2}[t])
    print(f"\nPrediction errors:")
    print(f"  Underpredict (missed risk):  {underpredict}")
    print(f"  Overpredict (wasted cost):   {overpredict}")

    return {
        'total_expected_cost': total_expected_cost,
        'avg_latency': avg_latency,
        'avg_error_cost': avg_error_cost,
        'underpredict': underpredict,
        'overpredict': overpredict,
        'level_accuracy': level_accuracy,
    }

results = {
    'routing': evaluate_detailed(routing_eval, "Semantic Routing"),
    'slm': evaluate_detailed(slm_eval, "SLM Distillation (checkpoint-31)"),
}

print("\n" + "=" * 70)
print("WHY SEMANTIC ROUTING WINS")
print("=" * 70)
print()
print("Semantic Routing has lower Total Expected Cost because:")
print(f"  1. Much lower latency: {results['routing']['avg_latency']:.3f}ms vs {results['slm']['avg_latency']:.3f}ms")
print(f"  2. Fewer overpredictions: {results['routing']['overpredict']} vs {results['slm']['overpredict']}")
print(f"  3. Better balance: {results['routing']['level_accuracy']['low']:.1%} low accuracy vs {results['slm']['level_accuracy']['low']:.1%}")
print()
print("SLM's asymmetric loss training made it extremely conservative,")
print("over-predicting risk to avoid missing high-risk cases.")
print("This wastes resources on low/mid queries, increasing expected cost.")
print()
print("Semantic routing achieves a better cost-benefit balance.")
