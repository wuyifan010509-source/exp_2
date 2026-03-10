"""
修复 class_label 映射问题
"""
import json

# 原始数据文件
input_file = "exp/slm_distillation/training_data.jsonl"
output_file = "exp/slm_distillation/training_data_fixed.jsonl"

# 类别到 class_label 的正确映射
CATEGORY_TO_CLASS = {
    "A1": 0,  # 基础技术问题 → Low
    "A2": 1,  # 交易规则 → Mid
    "A3": 2,  # 资金问题 → High
    "B1": 0,  # 闲聊 → Low
    "B2": 0,  # 情感 → Low
    "C1": 2,  # 爆仓/强平 → High
    "C2": 2,  # 止损/止盈问题 → High
    "D1": 1,  # 系统错误 → Mid
    "D2": 1,  # 账户问题 → Mid
    "D3": 1,  # 投诉/举报 → Mid
    "OOS": 1,  # 未知/其他 → Mid
}

# Cost 到 class 的映射
COST_TO_CLASS = {
    1: 0,
    50: 1,
    100: 1,
    200: 1,
    500: 2,  # 500 属于 High
    1000: 2,
    2000: 2,
}

fixed_count = 0
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        if not line.strip():
            continue

        data = json.loads(line)

        # 使用 cost_label 来推导正确的 class_label
        cost = data.get('cost_label', 1)
        correct_class = COST_TO_CLASS.get(cost, 1)

        old_class = data.get('class_label', -1)
        if old_class != correct_class:
            fixed_count += 1

        # 更新 class_label
        data['class_label'] = correct_class

        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"总共处理行数: {fixed_count}")
print(f"修复的记录数: {fixed_count}")
print(f"输出文件: {output_file}")
