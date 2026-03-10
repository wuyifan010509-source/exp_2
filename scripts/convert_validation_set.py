"""
Convert validation_set_1000.jsonl to the format expected by evaluate.py.

Mapping logic (based on fix_labels.py and cost hierarchy):
- Low (class=0): A1, B1 (cost=1)
- Mid (class=1): A2, B2, D1, D2, D3, OOS (cost=50)
- High (class=2): A3, C1, C2 (cost=1000)

Note: This simplifies the 11-category taxonomy to 3-level hierarchy.
"""

import json
from collections import Counter

# Category to level mapping
CATEGORY_TO_LEVEL = {
    "A1": "low",
    "A2": "mid",
    "A3": "high",
    "B1": "low",
    "B2": "mid",
    "C1": "high",
    "C2": "high",
    "D1": "mid",
    "D2": "mid",
    "D3": "mid",
    "OOS": "mid",
}

LEVEL_TO_CLASS = {"low": 0, "mid": 1, "high": 2}
LEVEL_TO_COST = {"low": 1, "mid": 50, "high": 1000}

def convert_validation_set(input_path: str, output_path: str):
    """Convert validation_set_1000.jsonl to evaluation format."""
    records = []
    level_counter = Counter()

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            category = item.get("category", "OOS")
            level = CATEGORY_TO_LEVEL.get(category, "mid")
            class_label = LEVEL_TO_CLASS[level]
            cost_label = LEVEL_TO_COST[level]

            converted = {
                "query": item["query"],
                "level": level,
                "true_level": level,
                "class_label": class_label,
                "cost_label": cost_label,
                "true_cost": cost_label,
                "original_category": category,
                "original_cost_label": item.get("cost_label"),
            }

            records.append(converted)
            level_counter[level] += 1

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Print summary
    print("=" * 70)
    print("Validation Set Conversion Summary")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Total:  {len(records)} records")
    print(f"\nLevel distribution:")
    for level in ["low", "mid", "high"]:
        count = level_counter[level]
        pct = count / len(records) * 100
        print(f"  {level:5s}: {count:4d} ({pct:5.1f}%)")
    print("=" * 70)

    # Check for high-risk samples
    high_risk_queries = [r for r in records if r["level"] == "high"]
    print(f"\nHigh-risk samples (first 5):")
    for item in high_risk_queries[:5]:
        print(f'  "{item["query"]}" → {item["level"]} ({item["original_category"]})')

    return records

if __name__ == "__main__":
    convert_validation_set(
        "data/validation_set_1000.jsonl",
        "data/validation_set_1000_eval.jsonl"
    )
