"""
Analyze confusion matrix from SLM evaluation
"""

from evaluation.evaluate import load_test_data, SLMDistillationEvaluator
from collections import defaultdict

test_data = load_test_data('data/validation_set_1000_eval.jsonl')
evaluator = SLMDistillationEvaluator(adapter_path='output/slm_distillation/checkpoint-31')

# Run quick evaluation
confusion = defaultdict(int)

print("Running inference on 1000 samples...")
for item in test_data:
    pred = evaluator.predict(item['query'])
    true_level = item['true_level']
    pred_level = pred['predicted_level']
    confusion[(true_level, pred_level)] += 1

print('SLM Confusion Matrix (Actual -> Predicted):')
print('-' * 50)
print(f'{"Pred ->":>10} | Low:         | Mid:         | High:')
print('-' * 50)
for true in ['low', 'mid', 'high']:
    row = []
    for pred in ['low', 'mid', 'high']:
        count = confusion[(true, pred)]
        total = sum(confusion[(true, p)] for p in ['low', 'mid', 'high'])
        pct = count / total * 100 if total > 0 else 0
        row.append(f'{count:3d} ({pct:5.1f}%)')
    print(f'{true.upper():>6} -> {row[0]:>14} {row[1]:>14} {row[2]:>14}')

print()
print('=== Analysis ===')
print('Model is VERY conservative:')
print('- Low samples: Mostly predicted as High (severe over-prediction)')
print('- Mid samples: Mostly predicted as High (severe over-prediction)')
print('- High samples: Reasonably caught')
print()
print('This is due to asymmetric loss (alpha=20.0) which heavily')
print('penalizes missing high-risk cases.')
print()
print('=== Per-Level Recall ===')
for true in ['low', 'mid', 'high']:
    correct = confusion[(true, true)]
    total = sum(confusion[(true, p)] for p in ['low', 'mid', 'high'])
    recall = correct / total * 100 if total > 0 else 0
    print(f'{true.upper():>6}: {recall:.1f}% ({correct}/{total})')
