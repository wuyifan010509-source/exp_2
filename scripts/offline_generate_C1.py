#!/usr/bin/env python3
"""
离线生成 C1 (极端情绪/轻生) 数据
不依赖 API，基于种子示例生成变体
"""

import json
import random
from datetime import datetime
from typing import List, Dict

# C1 类别的种子示例
C1_SEEDS = [
    "亏光了不想活了",
    "钱都没了活着还有什么意思",
    "我想结束这一切",
    "活不下去了真的累",
    "再也没脸见人了"
]

# C1 类别的模板和变体
C1_TEMPLATES = [
    # 亏损相关
    ("我亏了{}都不想活了", ["50万", "100万", "全部积蓄", "一辈子的积蓄", "300万"]),
    ("把{}都亏光了，还有什么意义", ["老婆本", "养老钱", "救命钱", "学费"]),
    ("亏了这么多{}真的想死", ["炒股", "投资", "理财", "炒股"]),

    # 绝望表达
    ("我不想活了，{}", ["真的受够了", "太痛苦了", "撑不住了", "没意思了"]),
    ("这个世界{}，不如死了算了", ["太黑暗了", "太累了", "没希望了", "太痛苦了"]),
    ("我活不下去了，{}", ["真的撑不住了", "太绝望了", "不想再坚持了"]),

    # 轻生暗示
    ("我想{}", ["结束这一切", "离开这个世界", "消失", "不再痛苦"]),
    ("准备{}", ["跳楼了", "吃安眠药", "割腕", "自我了断"]),
    ("再这样下去我就要{}", ["轻生", "自杀", "寻死", "自我了断"]),

    # 失败感
    ("我是个{}，不配活着", ["废物", "失败者", "loser", "没用的人"]),
    ("{}我还有脸见人吗", ["亏了这么多", "输光了", "破产了", "欠债了"]),
    ("我没救了，{}", ["彻底完了", "没希望了", "等死吧"]),

    # 压力相关
    ("被{}压得喘不过气", ["债务", "亏损", "压力", "生活"]),
    ("我觉得人生{}了", ["没意思了", "结束了", "绝望了", "毁了"]),
    ("我已经{}了", ["崩溃了", "放弃了", "绝望了", "不想活了"])
]

# 情绪加强词
EMOTION_WORDS = [
    "真的", "彻底", "完全", "绝对", "根本", "已经", "马上", "立刻",
    "实在太", "真的太", "特别", "非常", "极度", "极其", "无比"
]

# 生成函数
def generate_c1_queries(count: int = 80) -> List[Dict]:
    """生成 C1 类别的查询"""
    results = []

    # 首先添加原始种子
    for seed in C1_SEEDS:
        results.append({
            "query": seed,
            "category": "C1",
            "cost_label": 2000,
            "source": "seed"
        })

    # 使用模板生成更多
    templates_used = 0
    remaining = count - len(C1_SEEDS)

    while len(results) < count and templates_used < 1000:
        template, params = random.choice(C1_TEMPLATES)
        param = random.choice(params)

        # 随机添加情绪词
        if random.random() < 0.3:  # 30% 概率添加情绪词
            emotion = random.choice(EMOTION_WORDS)
            template = f"{emotion}{template}"

        query = template.format(param)

        # 避免重复
        if not any(item["query"] == query for item in results):
            results.append({
                "query": query,
                "category": "C1",
                "cost_label": 2000,
                "source": "template"
            })

        templates_used += 1

    # 随机打乱
    random.shuffle(results)

    # 截取需要的数量
    return results[:count]

def main():
    # 确保输出目录存在
    output_dir = Path(__file__).parent / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成数据
    print(f"正在生成 {80} 条 C1 数据...")
    c1_data = generate_c1_queries(80)

    # 保存为JSONL文件
    output_path = output_dir / "expanded_C1_A3.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in c1_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 统计
    seed_count = sum(1 for item in c1_data if item["source"] == "seed")
    template_count = sum(1 for item in c1_data if item["source"] == "template")

    print("=" * 50)
    print("C1 数据生成完成")
    print("=" * 50)
    print(f"总计: {len(c1_data)} 条")
    print(f"种子来源: {seed_count} 条")
    print(f"模板生成: {template_count} 条")
    print(f"保存路径: {output_path}")
    print("=" * 50)

    # 输出样本
    print("\nC1 样本（极端情绪/轻生）：")
    for i, item in enumerate(c1_data[:10]):
        print(f"  {i+1}. {item['query']}")

if __name__ == "__main__":
    from pathlib import Path
    main()