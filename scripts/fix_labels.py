"""
权威修复标注数据标签
根据 category 字段重新生成正确的 cost_label、class_label、level 字段

根因：原始标注数据的 category 和 cost_label 不匹配
例如：C1 (轻生) 类别 cost_label=1，应该是 2000
"""

import argparse
import json
from collections import Counter, defaultdict

# 类别到正确的 cost 和 class 映射
CATEGORY_MAPPING = {
    "A1": {"cost": 1, "class_label": 0, "level": "low"},   # 基础技术问题
    "A2": {"cost": 50, "class_label": 1, "level": "mid"},  # 交易规则
    "A3": {"cost": 1000, "class_label": 2, "level": "high"}, # 资金问题
    "B1": {"cost": 1, "class_label": 0, "level": "low"},   # 闲聊
    "B2": {"cost": 1, "class_label": 0, "level": "low"},   # 情感
    "C1": {"cost": 1000, "class_label": 2, "level": "high"}, # 爆仓/强平 - 包含情绪表达
    "C2": {"cost": 2000, "class_label": 2, "level": "high"}, # 止损/止盈 - 极高危
    "D1": {"cost": 50, "class_label": 1, "level": "mid"},  # 系统错误
    "D2": {"cost": 50, "class_label": 1, "level": "mid"},  # 账户问题
    "D3": {"cost": 50, "class_label": 1, "level": "mid"},  # 投诉/举报
    "OOS": {"cost": 50, "class_label": 1, "level": "mid"}, # 未知/其他
}

# C1 类别中包含特定高危关键词的样本（需要识别）
C1_HIGH_KEYWORDS = [
    '结束', '自杀', '不想活', '想死', '绝望', '痛苦', '亏', '输光', '跳楼',
    '轻生', '了断', '结束一切', '活不下去', '死', '受不了'
]


def classify_c1_sample(query: str) -> bool:
    """检查 C1 样本是否包含高危情绪表达"""
    query_lower = query.lower()
    for keyword in C1_HIGH_KEYWORDS:
        if keyword in query_lower:
            return True
    return False


def fix_labels(input_file: str, output_file: str) -> None:
    """修复数据标签"""
    records = []

    category_count = Counter()
    fixed_count = 0
    c1_high_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            category = data.get('category', 'OOS')
            query = data.get('query', '')

            category_count[category] += 1

            # 获取正确映射
            correct_mapping = CATEGORY_MAPPING.get(category, CATEGORY_MAPPING["OOS"])
            new_cost = correct_mapping["cost"]
            new_class = correct_mapping["class_label"]
            new_level = correct_mapping["level"]

            # C1 类别特殊处理：检查是否包含高危情绪表达
            if category == 'C1':
                if classify_c1_sample(query):
                    c1_high_count += 1
                    # C1 包含高危情绪 → High
                    new_cost = 2000
                    new_class = 2
                    new_level = "high"

            # 检查是否需要修复
            old_cost = data.get('cost_label', -1)
            old_class = data.get('class_label', -1)
            old_level = data.get('level', None)

            if (old_cost != new_cost or
                old_class != new_class or
                old_level != new_level):
                fixed_count += 1

            # 更新字段
            data['cost_label'] = new_cost
            data['class_label'] = new_class
            data['level'] = new_level

            records.append(data)

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for record in records:
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')

    # 打印统计
    print("=" * 70)
    print("数据标签修复完成")
    print("=" * 70)
    print(f"\n输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"总记录数: {len(records)}")
    print(f"修复记录数: {fixed_count}")

    print("\n类别分布:")
    for cat in sorted(category_count.keys()):
        mapping = CATEGORY_MAPPING.get(cat, CATEGORY_MAPPING["OOS"])
        print(f"  {cat}: {category_count[cat]:4d} 条 → level={mapping['level']:4s}, cost={mapping['cost']:4d}")

    print(f"\nC1 类高危情绪样本: {c1_high_count} 条")

    # 统计最终分布
    level_dist = defaultdict(int)
    for r in records:
        level_dist[r['level']] += 1

    print("\n最终 Level 分布:")
    total = len(records)
    for level in ["low", "mid", "high"]:
        count = level_dist[level]
        pct = count / total * 100
        print(f"  {level}: {count:4d} 条 ({pct:5.1f}%)")

    print("\n高危样本抽样 (前10条):")
    high_samples = [r for r in records if r['level'] == 'high'][:10]
    for i, sample in enumerate(high_samples, 1):
        query_short = sample['query'][:40]
        print(f"  {i}. [{sample['category']}] {query_short}...")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="修复标注数据标签")
    parser.add_argument("--input", required=True, help="输入文件路径")
    parser.add_argument("--output", required=True, help="输出文件路径")
    args = parser.parse_args()

    fix_labels(args.input, args.output)


if __name__ == "__main__":
    main()
