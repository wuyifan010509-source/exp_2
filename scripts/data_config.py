"""
数据标注 Pipeline 配置文件
"""

# 类别定义与 Cost 映射
CATEGORIES = {
    "A1": {"name": "基础业务问题", "cost": 1, "target_pct": 10,
           "desc": "APP操作、开户、密码等低风险问题"},
    "A2": {"name": "交易规则咨询", "cost": 50, "target_pct": 25,
           "desc": "印花税、科创板、逆回购等规则问题"},
    "A3": {"name": "资金安全/爆仓", "cost": 1000, "target_pct": 10,
           "desc": "资金异常、强平、爆仓等高风险问题"},
    "B1": {"name": "非业务闲聊", "cost": 1, "target_pct": 8,
           "desc": "天气、笑话、心情等完全无关问题"},
    "B2": {"name": "跨业务咨询", "cost": 1, "target_pct": 5,
           "desc": "医保、社保、其他行业问题"},
    "C1": {"name": "极端情绪/轻生", "cost": 2000, "target_pct": 3,
           "desc": "明确轻生意向、严重情绪崩溃"},
    "C2": {"name": "严重投诉/危机", "cost": 500, "target_pct": 5,
           "desc": "激烈投诉、可能引发公关危机"},
    "D1": {"name": "退款/销户", "cost": 200, "target_pct": 5,
           "desc": "客户流失风险，需人工挽留"},
    "D2": {"name": "投诉/索赔", "cost": 500, "target_pct": 5,
           "desc": "法律风险，需安抚或法务介入"},
    "D3": {"name": "差评威胁", "cost": 100, "target_pct": 4,
           "desc": "舆情风险，需优先处理"},
    "OOS": {"name": "超出范围", "cost": 50, "target_pct": 20,
            "desc": "复杂分析、荐股请求等"},
}

# 扩充目标 (总计约 3000 条)
TOTAL_TARGET = 3000
EXPANSION_TARGETS = {
    cat: int(info["target_pct"] / 100 * TOTAL_TARGET)
    for cat, info in CATEGORIES.items()
}

# 优先级排序 (高 Cost 优先)
PRIORITY_ORDER = ["C1", "A3", "C2", "D2", "D1", "D3", "A2", "OOS", "B1", "B2", "A1"]

# API 配置
API_CONFIG = {
    "openai": {
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
    },
    "deepseek": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
    }
}

# Prompt 模板
EXPANSION_PROMPT_TEMPLATE = """你是一位投资软件客服场景的数据生成专家。请为分类 {category} 生成 {count} 条用户查询语句。

分类定义：
- 类别代码：{category}
- 类别名称：{category_name}
- 风险描述：{description}
- 业务代价权重：{cost}

要求：
1. 查询需贴近真实客服对话，涵盖各种表达方式（口语化、简洁、详细等）
2. 必须符合该类别的语义特征，不要与其他类别混淆
3. 语言要自然多样，避免模板化
4. 严格按照 JSON 格式输出

输出格式：
```json
[
  {{"query": "用户查询语句", "category": "{category}", "description": "简短说明"}},
  ...
]
```

请直接输出 JSON，不要有其他内容。"""

LABELING_PROMPT_TEMPLATE = """请判断以下用户查询的类别和业务代价。

查询：{query}

类别选项：
- A1: 基础业务问题（APP操作、开户、密码）Cost=1
- A2: 交易规则咨询（印花税、科创板、逆回购）Cost=50
- A3: 资金安全/爆仓（强平、爆仓、资金异常）Cost=1000
- B1: 非业务闲聊（天气、笑话）Cost=1
- B2: 跨业务咨询（医保、社保）Cost=1
- C1: 极端情绪/轻生意向 Cost=2000
- C2: 严重投诉/危机 Cost=500
- D1: 退款/销户 Cost=200
- D2: 投诉/索赔 Cost=500
- D3: 差评威胁 Cost=100
- OOS: 超出范围（复杂分析、荐股）Cost=50

请严格按照以下 JSON 格式输出：
{{"category": "类别代码", "cost_label": 数字, "reasoning": "简短判断理由"}}"""
