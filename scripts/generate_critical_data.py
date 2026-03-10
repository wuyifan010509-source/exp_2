#!/usr/bin/env python3
"""
紧急数据生成脚本 - 为 C1 和 A3 类别生成高质量数据
不依赖外部 API，使用模板 + 变体生成
"""

import json
import random
from pathlib import Path

# C1: 极端情绪/轻生 (Cost=2000)
C1_TEMPLATES = {
    "轻生意向明确": [
        "亏了{amount}，不想活了",
        "钱都没了，活着还有什么意思",
        "我想结束这一切",
        "活不下去了，真的太累了",
        "再也没脸见人了",
        "赔光了，没脸回家",
        "亏了这么多，怎么面对家人",
        "真的撑不下去了",
        "不想连累家人，想一了百了",
        "彻底绝望了，没希望了"
    ],
    "严重情绪崩溃": [
        "心态崩了，完全崩溃",
        "睡不着觉，一直在想这件事",
        "吃不下饭，瘦了很多",
        "每天都活在痛苦中",
        "精神快崩溃了",
        "感觉天都塌了",
        "整个人都废了",
        "什么都提不起兴趣",
        "看什么都是灰色的",
        "完全失去了信心"
    ],
    "求助无门": [
        "没人能帮我",
        "走投无路了",
        "救救我，不知道该怎么办",
        "谁能拉我一把",
        "实在没办法了",
        "求助无门，绝望至极",
        "没有人理解我的痛苦",
        "不知道还能找谁",
        "感觉被全世界抛弃了",
        "孤独无助，看不到希望"
    ]
}

# A3: 资金安全/爆仓 (Cost=1000)
A3_TEMPLATES = {
    "强平/爆仓": [
        "担保比例{ratio}%会被强平吗",
        "账户被强平了，能追回吗",
        "爆仓了，怎么办",
        "融资买入后爆仓了",
        "保证金不足要被平仓",
        "马上要强平了，紧急",
        "快帮我看看担保比例",
        "要爆仓了，快救救我",
        "强制平仓通知收到了",
        "账户要爆仓了，怎么办"
    ],
    "资金异常": [
        "账户里的钱突然少了{amount}",
        "资金被冻结了",
        "账户显示异常",
        "钱不见了，怎么回事",
        "账户余额对不上",
        "提现失败，钱去哪了",
        "资金被锁定了",
        "账户出现异常交易",
        "莫名其妙少了钱",
        "资金安全问题"
    ],
    "重大损失": [
        "期权到期忘记行权，损失{amount}",
        "系统故障导致巨额亏损",
        "交易延迟造成了损失",
        "因为系统问题亏了{amount}",
        "操作失误导致大额亏损",
        "一笔交易亏了{amount}",
        "止损没触发，损失惨重",
        "挂单失败导致损失",
        "成交价异常，亏了很多",
        "系统卡顿造成亏损"
    ]
}

def generate_variations(templates, category, cost, target_count):
    """生成多样化变体"""
    results = []
    amounts = ["几万", "十几万", "几十万", "上百万", "全部积蓄", "养老钱", "孩子的学费"]
    ratios = ["110", "120", "130", "140", "150", "160", "180"]

    all_templates = []
    for template_list in templates.values():
        all_templates.extend(template_list)

    # 生成基础变体
    for template in all_templates:
        # 原始模板
        if "{amount}" in template:
            for amount in random.sample(amounts, min(3, len(amounts))):
                query = template.format(amount=amount)
                results.append({
                    "query": query,
                    "category": category,
                    "cost_label": cost
                })
        elif "{ratio}" in template:
            for ratio in random.sample(ratios, min(3, len(ratios))):
                query = template.format(ratio=ratio)
                results.append({
                    "query": query,
                    "category": category,
                    "cost_label": cost
                })
        else:
            results.append({
                "query": template,
                "category": category,
                "cost_label": cost
            })

    # 如果还不够，随机生成更多变体
    while len(results) < target_count:
        template = random.choice(all_templates)
        if "{amount}" in template:
            amount = random.choice(amounts)
            query = template.format(amount=amount)
        elif "{ratio}" in template:
            ratio = random.choice(ratios)
            query = template.format(ratio=ratio)
        else:
            # 添加一些语气词或修饰
            prefixes = ["", "紧急求助：", "急！", "救命！", "很急！"]
            suffixes = ["", "，怎么办？", "，帮帮我", "，很着急", "，求回复"]
            query = random.choice(prefixes) + template + random.choice(suffixes)

        results.append({
            "query": query,
            "category": category,
            "cost_label": cost
        })

    return results[:target_count]

def main():
    output_dir = Path(__file__).parent / "checkpoints"
    output_dir.mkdir(exist_ok=True)

    # 生成 C1 数据 (80 条)
    print("=== 生成 C1 (极端情绪/轻生) 数据 ===")
    c1_data = generate_variations(C1_TEMPLATES, "C1", 2000, 80)
    print(f"  生成: {len(c1_data)} 条")

    # 生成 A3 数据 (250 条)
    print("\n=== 生成 A3 (资金安全/爆仓) 数据 ===")
    a3_data = generate_variations(A3_TEMPLATES, "A3", 1000, 250)
    print(f"  生成: {len(a3_data)} 条")

    # 合并并保存
    all_data = c1_data + a3_data
    output_file = output_dir / "expanded_C1_A3.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ 总计生成: {len(all_data)} 条")
    print(f"📄 保存至: {output_file}")
    print(f"  - C1: {len(c1_data)} 条")
    print(f"  - A3: {len(a3_data)} 条")

if __name__ == "__main__":
    main()
