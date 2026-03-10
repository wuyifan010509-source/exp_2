#!/usr/bin/env python3
"""
验证集生成脚本 - 使用大模型生成自然多样的验证数据

本脚本使用 LLM API 生成高质量的验证集数据，避免模板化，
确保数据自然、贴近真实用户表达。
"""

import json
import os
import sys
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from openai import OpenAI
except ImportError:
    print("请安装 openai: uv add openai")
    sys.exit(1)

# 加载 .env 文件
def load_env_file(env_path: Path = None):
    """加载 .env 文件"""
    if env_path is None:
        env_path = Path(__file__).parent.parent / ".env"
    
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key, value)
        print(f"[Config] 已加载 .env: {env_path}")
    else:
        print(f"[Config] .env 文件不存在: {env_path}")

# 启动时加载环境变量
load_env_file()

# 类别配置 - 验证集配额 (总计 1000 条)
VALIDATION_CONFIG = {
    "A1": {"name": "基础业务问题", "cost": 1, "count": 100,
           "desc": "APP操作、开户、密码等低风险问题"},
    "A2": {"name": "交易规则咨询", "cost": 50, "count": 250,
           "desc": "印花税、科创板、逆回购等规则问题"},
    "A3": {"name": "资金安全/爆仓", "cost": 1000, "count": 100,
           "desc": "资金异常、强平、爆仓等高风险问题"},
    "B1": {"name": "非业务闲聊", "cost": 1, "count": 80,
           "desc": "天气、笑话、心情等完全无关问题"},
    "B2": {"name": "跨业务咨询", "cost": 1, "count": 50,
           "desc": "医保、社保、其他行业问题"},
    "C1": {"name": "极端情绪/轻生", "cost": 2000, "count": 30,
           "desc": "明确轻生意向、严重情绪崩溃"},
    "C2": {"name": "严重投诉/危机", "cost": 500, "count": 50,
           "desc": "激烈投诉、可能引发公关危机"},
    "D1": {"name": "退款/销户", "cost": 200, "count": 50,
           "desc": "客户流失风险，需人工挽留"},
    "D2": {"name": "投诉/索赔", "cost": 500, "count": 50,
           "desc": "法律风险，需安抚或法务介入"},
    "D3": {"name": "差评威胁", "cost": 100, "count": 40,
           "desc": "舆情风险，需优先处理"},
    "OOS": {"name": "超出范围", "cost": 50, "count": 200,
            "desc": "复杂分析、荐股请求等"},
}

# 各类别的种子示例 (用于 few-shot)
CATEGORY_EXAMPLES = {
    "A1": [
        "密码忘了怎么重置",
        "APP闪退怎么办",
        "怎么看我的资产",
        "怎么绑定银行卡",
        "K线图怎么看",
    ],
    "A2": [
        "科创板开通条件",
        "印花税怎么算",
        "新股申购规则",
        "逆回购怎么操作",
        "融资融券利率",
    ],
    "A3": [
        "为什么强制平仓",
        "担保比例不足",
        "账户被冻结",
        "穿仓要赔钱吗",
        "强平价格怎么定",
    ],
    "B1": [
        "今天天气怎么样",
        "讲个笑话吧",
        "推荐个电影",
        "附近有什么好吃的",
        "早上好",
    ],
    "B2": [
        "医保怎么报销",
        "社保怎么转移",
        "公积金提取",
        "个税申报",
        "驾照换证",
    ],
    "C1": [
        "亏光了不想活了",
        "活着没意思",
        "结束这一切",
        "对不起家人",
        "压力太大了",
    ],
    "C2": [
        "我要投诉到证监会",
        "媒体曝光你们",
        "集体诉讼",
        "严重欺诈",
        "实名举报",
    ],
    "D1": [
        "我要销户",
        "把钱退给我",
        "不玩了",
        "全部清仓",
        "注销账户",
    ],
    "D2": [
        "要求赔偿损失",
        "系统故障导致亏损",
        "索赔",
        "你们的责任",
        "法律途径",
    ],
    "D3": [
        "不给一星差评",
        "应用商店差评",
        "曝光到微博",
        "朋友圈避雷",
        "论坛发帖",
    ],
    "OOS": [
        "分析一下茅台走势",
        "推荐几只股票",
        "明天大盘涨跌",
        "这股票能买吗",
        "选股建议",
    ],
}


def get_client():
    """获取 OpenAI 客户端 - 优先使用 .env 配置"""
    # 优先使用 .env 中的配置
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL_BASE", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY", "sk-xxx")
    
    # 如果没有 .env 配置，尝试其他 API
    if base_url == "https://api.openai.com/v1" and not api_key.startswith("sk-"):
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("MINIMAX_API_KEY")
        if os.getenv("DEEPSEEK_API_KEY"):
            base_url = "https://api.deepseek.com/v1"
            model = "deepseek-chat"
        elif os.getenv("MINIMAX_API_KEY"):
            base_url = "https://api.minimax.chat/v1"
            model = "abab6.5s-chat"
    
    if not api_key:
        print("错误：请设置 API Key 或配置 .env 文件")
        sys.exit(1)
    
    print(f"[API] Base URL: {base_url}")
    print(f"[API] Model: {model}")
    
    return OpenAI(api_key=api_key, base_url=base_url), model


def create_prompt(category: str, count: int) -> str:
    """创建生成数据的 Prompt"""
    config = VALIDATION_CONFIG[category]
    examples = CATEGORY_EXAMPLES[category]
    
    prompt = f"""你是金融客服场景的数据生成专家。请为以下类别生成 {count} 条用户查询语句。

类别信息：
- 代码：{category}
- 名称：{config['name']}
- 描述：{config['desc']}
- 代价权重：{config['cost']}

生成要求：
1. 语句必须贴近真实用户表达，口语化、自然
2. 表达方式多样化：
   - 长短不一（短至3个字，长至50字）
   - 语气多样（疑问、陈述、感叹、命令）
   - 情绪层次（平静、焦虑、愤怒、绝望等）
3. 避免模板化表达，每句话要有独特性
4. 可以包含错别字、口语词、网络用语
5. 不要与示例重复

参考示例（仅作风格参考，不要复制）：
{chr(10).join(f'  - {ex}' for ex in examples)}

请严格按照以下 JSON 格式输出，不要有任何其他内容：
```json
[
  {{"query": "用户查询语句", "style": "描述风格特点"}},
  ...
]
```

生成 {count} 条不同的查询："""

    return prompt


def generate_batch(client, model: str, category: str, batch_size: int, max_retries: int = 3) -> List[Dict]:
    """生成一批数据"""
    prompt = create_prompt(category, batch_size)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专业的金融客服数据生成专家，擅长生成自然、多样的用户查询语句。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.85,  # 较高的温度增加多样性
                max_tokens=2000,
            )
            
            content = response.choices[0].message.content.strip()
            
            # 提取 JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].strip()
            else:
                json_str = content
            
            data = json.loads(json_str)
            
            # 验证并添加元数据
            results = []
            for item in data:
                if "query" in item:
                    results.append({
                        "query": item["query"],
                        "category": category,
                        "cost_label": VALIDATION_CONFIG[category]["cost"],
                        "style": item.get("style", "未知"),
                    })
            
            return results
            
        except Exception as e:
            print(f"  ⚠️ 尝试 {attempt + 1}/{max_retries} 失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
            else:
                print(f"  ❌ {category} 批次生成失败")
                return []
    
    return []


def generate_category_data(client, model: str, category: str, target_count: int, 
                          batch_size: int = 10) -> List[Dict]:
    """生成单个类别的全部数据"""
    print(f"\n[{category}] 目标: {target_count} 条")
    
    all_data = []
    batches = (target_count + batch_size - 1) // batch_size
    
    for i in range(batches):
        current_batch_size = min(batch_size, target_count - len(all_data))
        if current_batch_size <= 0:
            break
        
        print(f"  批次 {i+1}/{batches}: 生成 {current_batch_size} 条...", end=" ")
        batch_data = generate_batch(client, model, category, current_batch_size)
        
        if batch_data:
            all_data.extend(batch_data)
            print(f"✓ 成功 {len(batch_data)} 条")
        else:
            print("✗ 失败")
        
        # 避免 rate limit
        time.sleep(0.5)
    
    print(f"  [{category}] 完成: {len(all_data)}/{target_count} 条")
    return all_data


def deduplicate_queries(data: List[Dict]) -> List[Dict]:
    """去重"""
    seen = set()
    unique_data = []
    
    for item in data:
        query = item["query"].strip()
        if query and query not in seen:
            seen.add(query)
            unique_data.append(item)
    
    return unique_data


def generate_validation_set(output_path: str = "validation_set_llm_1000.jsonl"):
    """生成完整验证集"""
    print("=" * 70)
    print("验证集生成 - 使用大模型 (LLM)")
    print("=" * 70)
    
    client, model = get_client()
    print(f"[模型] {model}")
    print(f"[输出] {output_path}")
    
    all_data = []
    
    # 按优先级顺序生成
    for category in VALIDATION_CONFIG.keys():
        config = VALIDATION_CONFIG[category]
        target = config["count"]
        
        # 生成数据
        category_data = generate_category_data(
            client, model, category, target, batch_size=10
        )
        
        all_data.extend(category_data)
        
        # 保存中间结果
        checkpoint_path = output_path.replace(".jsonl", f"_{category}.jsonl")
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            for item in category_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"  💾 已保存 checkpoint: {checkpoint_path}")
    
    # 去重
    print("\n" + "-" * 70)
    print("后处理...")
    before_dedup = len(all_data)
    all_data = deduplicate_queries(all_data)
    after_dedup = len(all_data)
    print(f"  去重: {before_dedup} → {after_dedup} 条 (移除 {before_dedup - after_dedup} 条重复)")
    
    # 随机打乱
    random.shuffle(all_data)
    
    # 保存最终结果
    output_full_path = Path(output_path)
    output_full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_full_path, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 统计报告
    print("\n" + "=" * 70)
    print("验证集生成完成！")
    print("=" * 70)
    
    from collections import Counter
    cat_counts = Counter(d['category'] for d in all_data)
    cost_counts = Counter(d['cost_label'] for d in all_data)
    
    print(f"\n总计: {len(all_data)} 条")
    print("\n类别分布:")
    for cat in sorted(cat_counts.keys()):
        count = cat_counts[cat]
        target = VALIDATION_CONFIG[cat]["count"]
        cost = VALIDATION_CONFIG[cat]["cost"]
        status = "✓" if count >= target * 0.9 else "⚠️"
        print(f"  {status} {cat:4s} (Cost={cost:4d}): {count:4d} 条 (目标 {target})")
    
    print("\nCost 分布:")
    for cost in sorted(cost_counts.keys()):
        count = cost_counts[cost]
        pct = count / len(all_data) * 100
        bar = "█" * int(pct / 2)
        print(f"  Cost={cost:4d}: {count:4d} 条 ({pct:5.1f}%) {bar}")
    
    print(f"\n✅ 输出文件: {output_full_path.absolute()}")
    
    return all_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用大模型生成验证集")
    parser.add_argument("--output", type=str, default="validation_set_llm_1000.jsonl",
                       help="输出文件路径")
    parser.add_argument("--category", type=str, default=None,
                       help="仅生成指定类别 (如: A1, C1 等)")
    
    args = parser.parse_args()
    
    if args.category:
        # 仅生成指定类别
        if args.category not in VALIDATION_CONFIG:
            print(f"错误：未知类别 {args.category}")
            print(f"可用类别: {', '.join(VALIDATION_CONFIG.keys())}")
            sys.exit(1)
        
        client, model = get_client()
        config = VALIDATION_CONFIG[args.category]
        data = generate_category_data(client, model, args.category, config["count"])
        
        # 保存
        output_path = f"validation_set_{args.category}.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"\n✅ 已保存: {output_path} ({len(data)} 条)")
    else:
        # 生成完整验证集
        generate_validation_set(args.output)
