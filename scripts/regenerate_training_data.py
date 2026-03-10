"""
从原始标注数据生成正确的训练数据
"""
import json

# 输入输出文件
input_file = "exp/data/data_labeling/final_labeled_data.jsonl"
output_file = "exp/slm_distillation/training_data.jsonl"

# 类别到 class_label 的正确映射
CATEGORY_TO_CLASS = {
    "A1": 0,  # 基础技术问题 → Low
    "A2": 1,  # 交易规则 → Mid
    "A3": 2,  # 资金问题 → High
    "B1": 0,  # 闲聊 → Low
    "B2": 0,  # 情感 → Low
    "C1": 2,  # 爆仓/强平 → High（情感表达）
    "C2": 2,  # 止损/止盈问题 → High
    "D1": 1,  # 系统错误 → Mid
    "D2": 1,  # 账户问题 → Mid
    "D3": 1,  # 投诉/举报 → Mid
    "OOS": 1,  # 未知/其他 → Mid
}

# C1/C2 类别中特定关键词映射为 High
C1_HIGH_KEYWORDS = [
    '结束', '自杀', '不想活', '想死', '绝望', '痛苦', '亏', '输光', '跳楼',
    '轻生', '了断', '结束一切'
]
C2_HIGH_KEYWORDS = ['止损', '止盈', '平仓', '爆仓']

# 成本映射
COST_MAP = {
    0: 1,     # Low
    1: 50,    # Mid
    100: 1,    # Mid（C1/C2 某些可能是误标记，但先保持为 Mid）
    500: 2,    # High
    1000: 2,   # High
    2000: 2,   # High
}

def classify_sample(query: str, category: str, current_cost: int) -> int:
    """
    根据查询内容和类别判断正确的 class_label

    Args:
        query: 查询文本
        category: 类别
        current_cost: 当前 cost_label 值

    Returns:
        正确的 class_label (0/1/2)
    """
    # 对于 C1/C2 类别，检查是否包含高危关键词
    if category in ['C1', 'C2']:
        query_lower = query.lower()
        for keyword in (C1_HIGH_KEYWORDS if category == 'C1' else C2_HIGH_KEYWORDS):
            if keyword in query_lower:
                return 2  # High
        # 默认 Mid
        return 1

    # 对于其他类别，根据 category 映射
    return CATEGORY_TO_CLASS.get(category, 1)

fixed_count = 0
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue

        data = json.loads(line)

        # 如果没有 class_label，添加它
        if 'class_label' not in data:
            category = data.get('category', '')
            current_cost = data.get('cost_label', 1)

            # 使用改进的分类逻辑
            corrected_class = classify_sample(data['query'], category, current_cost)

            if corrected_class != current_cost:
                # 更新 cost_label
                data['cost_label'] = COST_MAP.get(corrected_class, 1)
                fixed_count += 1

            data['class_label'] = corrected_class

        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"处理记录数: {fixed_count}")
print(f"修复的记录数: {fixed_count}")
print(f"输出文件: {output_file}")
