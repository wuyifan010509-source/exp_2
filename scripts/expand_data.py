#!/usr/bin/env python3
"""
数据扩充脚本 - 为每个类别生成更多变体查询
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict
import random

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from openai import OpenAI
except ImportError:
    print("请安装 openai: pip install openai")
    sys.exit(1)

# 类别配置
CATEGORY_CONFIG = {
    "A1": {
        "name": "基础业务问题",
        "cost": 1,
        "seeds": [
            "我的账户余额是多少？",
            "怎么修改密码？",
            "如何绑定银行卡？",
            "APP在哪里下载？",
            "怎么看我的持仓？"
        ]
    },
    "A2": {
        "name": "交易规则咨询",
        "cost": 50,
        "seeds": [
            "基金赎回几天到账？",
            "科创板怎么开通？",
            "印花税是多少？",
            "新股申购有什么条件？",
            "融资融券利率多少？"
        ]
    },
    "A3": {
        "name": "资金安全/爆仓",
        "cost": 1000,
        "seeds": [
            "我的担保比例130%会被强平吗？",
            "账户显示异常资金被冻结了",
            "期权到期忘记行权怎么办？",
            "融资买入后爆仓了能追回吗？",
            "账户里的钱突然少了是怎么回事？"
        ]
    },
    "B1": {
        "name": "非业务闲聊",
        "cost": 1,
        "seeds": [
            "今天天气怎么样？",
            "讲个笑话吧",
            "你喜欢吃什么？",
            "你会唱歌吗？",
            "有什么好电影推荐？"
        ]
    },
    "B2": {
        "name": "跨业务咨询",
        "cost": 1,
        "seeds": [
            "医保怎么报销？",
            "社保余额怎么查？",
            "公积金能取出来吗？",
            "个税怎么申报？",
            "驾驶证怎么换证？"
        ]
    },
    "C1": {
        "name": "极端情绪/轻生",
        "cost": 2000,
        "seeds": [
            "亏光了不想活了",
            "钱都没了活着还有什么意思",
            "我想结束这一切",
            "活不下去了真的累",
            "再也没脸见人了"
        ]
    },
    "C2": {
        "name": "严重投诉",
        "cost": 500,
        "seeds": [
            "你们是骗子我要投诉",
            "我要去证监会举报你们",
            "这系统太烂了要求赔偿",
            "你们这是欺诈行为",
            "我要找媒体曝光"
        ]
    },
    "D1": {
        "name": "退款/销户",
        "cost": 200,
        "seeds": [
            "我要退款怎么操作？",
            "把账户销了钱退给我",
            "不想玩了怎么销户？",
            "会员费能退吗？",
            "我要注销账户"
        ]
    },
    "D2": {
        "name": "投诉/索赔",
        "cost": 500,
        "seeds": [
            "因为你们系统问题亏了要赔偿",
            "延迟导致的损失你们负责吗？",
            "我要索赔之前的损失",
            "你们的问题导致我亏损",
            "系统bug造成的损失怎么赔？"
        ]
    },
    "D3": {
        "name": "差评威胁",
        "cost": 100,
        "seeds": [
            "不解决我就去黑猫投诉",
            "差评已经写好了",
            "要给一星评价了",
            "微博上曝光你们",
            "朋友圈告诉大家别用"
        ]
    },
    "OOS": {
        "name": "超出范围",
        "cost": 50,
        "seeds": [
            "帮我分析一下茅台后市走势",
            "推荐几只值得买的股票",
            "大盘明天会涨还是跌？",
            "这只股票能翻倍吗？",
            "帮我选几只基金定投"
        ]
    }
}

def get_client():
    """获取 OpenAI 客户端"""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("MINIMAX_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or "https://api.minimax.chat/v1"

    if not api_key:
        raise ValueError("请设置 OPENAI_API_KEY、DEEPSEEK_API_KEY 或 MINIMAX_API_KEY")

    return OpenAI(api_key=api_key, base_url=base_url)

def generate_queries(client, category: str, count: int, batch_size: int = 20) -> List[Dict]:
    """为指定类别生成查询"""
    config = CATEGORY_CONFIG[category]
    results = []

    # 预处理种子示例列表
    seeds_str = '\n'.join('- ' + s for s in config['seeds'])

    # 构建prompt模板 - 使用f-string格式
    batches = (count + batch_size - 1) // batch_size

    for i in range(batches):
        current_batch = min(batch_size, count - i * batch_size)

        # 直接构建完整的prompt，避免format()的复杂性
        prompt = f"""你是投资软件客服数据生成专家。请为以下类别生成 {current_batch} 条用户查询语句。

类别：{config['name']} (代码: {category})
Cost: {config['cost']}
描述：贴近真实投资客服场景，语言自然多样

种子示例：
{seeds_str}

要求：
1. 每条查询必须符合该类别定义
2. 语言风格多样（口语化、简洁、详细等）
3. 避免与种子示例重复
4. 避免生成模板化语句

请严格按照以下 JSON 格式输出，不要有其他内容：
[{{"query": "查询语句", "variation": "变体说明"}}]
"""

        try:
            response = client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "abab6-chat"),  # MINIMAX 的默认模型
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )

            content = response.choices[0].message.content
            # 提取 JSON
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            queries = json.loads(content)

            for q in queries:
                results.append({
                    "query": q["query"],
                    "category": category,
                    "cost_label": config["cost"],
                    "batch": i + 1
                })

            print(f"  [Batch {i+1}/{batches}] 生成 {len(queries)} 条")

        except Exception as e:
            print(f"  [Batch {i+1}/{batches}] 错误: {e}")

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="数据扩充")
    parser.add_argument("--categories", nargs="+", required=True, help="要扩充的类别")
    parser.add_argument("--count", type=int, default=50, help="每类生成数量")
    parser.add_argument("--output", default=None, help="输出文件路径")
    args = parser.parse_args()

    client = get_client()
    all_results = []

    for cat in args.categories:
        if cat not in CATEGORY_CONFIG:
            print(f"警告: 未知类别 {cat}")
            continue

        print(f"\n=== 扩充类别 {cat} ({CATEGORY_CONFIG[cat]['name']}) ===")
        results = generate_queries(client, cat, args.count)
        all_results.extend(results)
        print(f"  完成: {len(results)} 条")

    # 保存结果
    output_path = args.output or Path(__file__).parent / "checkpoints" / "expanded_queries.jsonl"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n总计生成: {len(all_results)} 条")
    print(f"保存至: {output_path}")

if __name__ == "__main__":
    main()
