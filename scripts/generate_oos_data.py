#!/usr/bin/env python3
"""
生成OOS（超出范围）类别数据
目标：生成343条数据，补充至500条
"""

import json
import random
from datetime import datetime, timedelta

# 生成OOS数据的模板和关键词
oos_templates = {
    "个股分析": [
        "帮我分析一下{company}的基本面",
        "{company}现在值得买入吗？",
        "帮我分析一下{company}的技术面",
        "{company}现在适合加仓吗？",
        "帮我分析一下{company}的财务数据",
        "{company}现在可以抄底了吗？",
        "帮我预测一下{company}的未来走势",
        "{company}现在适合持有吗？",
        "帮我分析一下{company}的行业地位",
        "{company}的估值合理吗？"
    ],
    "荐股请求": [
        "给我推荐几个有潜力的股票",
        "推荐几只好股票给我",
        "有什么股票可以关注？",
        "帮我看看现在哪些股票值得关注",
        "给我推荐几个低估值股票",
        "有什么成长性好的股票推荐吗？",
        "推荐几只绩优股",
        "有什么龙头股值得买入？",
        "帮我推荐几个热点板块的股票",
        "有什么底部反弹的股票吗？"
    ],
    "大盘预测": [
        "明天大盘会涨还是跌？",
        "帮我预测一下下周的行情",
        "大盘最近走势怎么样？",
        "帮我分析一下大盘的趋势",
        "明天A股会怎么走？",
        "下周大盘能突破XX点位吗？",
        "帮我预测一下本月的行情",
        "大盘什么时候会反转？",
        "最近市场情绪如何？",
        "帮我分析一下大盘的支撑位和压力位"
    ],
    "板块分析": [
        "帮我预测一下{sector}板块的走势",
        "{sector}板块最近表现如何？",
        "帮我分析一下{sector}板块的机会",
        "现在哪个板块最有投资价值？",
        "帮我分析一下{sector}板块的前景",
        "{sector}板块什么时候能启动？",
        "推荐几个有潜力的板块",
        "帮我分析一下各板块的轮动规律",
        "最近哪个板块最强势？",
        "帮我分析一下{sector}板块的龙头股"
    ],
    "行情分析": [
        "最近市场整体走势如何？",
        "帮我分析一下当前的行情",
        "现在是牛市还是熊市？",
        "市场情绪怎么样？",
        "帮我预测一下明天的行情",
        "最近有什么投资机会？",
        "市场什么时候会回暖？",
        "帮我分析一下现在的投资环境",
        "最近的资金流向如何？",
        "帮我分析一下市场的热点"
    ],
    "个股比较": [
        "帮我比较一下{company1}和{company2}哪个更好",
        "{company1}和{company2}哪个更有投资价值？",
        "帮我分析一下{company1}和{company2}的优劣",
        "买{company1}还是{company2}？",
        "帮我比较一下{company1}和{company2}的基本面",
        "{company1}和{company2}的估值哪个更合理？",
        "帮我分析一下{company1}和{company2}的成长性",
        "{company1}和{company2}哪个风险更小？",
        "帮我比较一下{company1}和{company2}的技术面",
        "{company1}和{company2}哪个分红更高？"
    ],
    "投资建议": [
        "我现在有XX万，该怎么投资？",
        "帮我制定一个投资策略",
        "我是新手，应该怎么炒股？",
        "帮我分析一下我的投资组合",
        "现在适合满仓吗？",
        "帮我推荐一个资产配置方案",
        "我现在应该买入还是卖出？",
        "帮我分析一下现在的投资时机",
        "我的持股比例合理吗？",
        "帮我制定一个长期投资计划"
    ],
    "技术分析": [
        "帮我分析一下大盘的K线形态",
        "现在是什么技术指标？",
        "帮我分析一下当前的量价关系",
        "现在适合做空吗？",
        "帮我分析一下支撑位和压力位",
        "现在是什么技术形态？",
        "帮我分析一下MACD指标",
        "现在适合抄底吗？",
        "帮我分析一下RSI指标",
        "现在的走势符合什么技术特征？"
    ]
}

# 公司列表
companies = [
    "贵州茅台", "腾讯控股", "宁德时代", "比亚迪", "美团", "隆基绿能", "招商银行",
    "平安银行", "工商银行", "恒瑞医药", "药明康德", "五粮液", "海天味业", "美的集团",
    "格力电器", "长江电力", "三峡能源", "中石油", "中石化", "中国移动", "中国联通",
    "中国神华", "陕西煤业", "紫金矿业", "江西铜业", "海螺水泥", "华新水泥", "三一重工",
    "中联重科", "立讯精密", "歌尔股份", "京东方A", "TCL科技", "中际旭创", "新易盛",
    "寒武纪", "海光信息", "中科软", "金山办公", "用友网络", "广联达", "中国平安",
    "中国人寿", "中信证券", "东方财富", "中航沈飞", "中航西飞", "牧原股份", "温氏股份",
    "顺丰控股", "恒生电子", "中金公司", "中信建投", "贵州茅台", "腾讯控股", "宁德时代",
    "比亚迪", "美团", "隆基绿能", "招商银行", "平安银行", "工商银行", "恒瑞医药",
    "药明康德", "五粮液", "海天味业", "美的集团", "格力电器", "长江电力", "三峡能源"
]

# 板块列表
sectors = [
    "新能源", "医药", "消费", "科技", "金融", "地产", "基建", "军工", "农业",
    "食品饮料", "汽车", "电子", "通信", "计算机", "传媒", "环保", "化工", "钢铁",
    "有色金属", "机械", "电气设备", "银行", "非银金融", "煤炭", "石油", "电力",
    "交通运输", "轻工制造", "商业贸易", "纺织服装", "休闲服务", "综合", "公用事业"
]

# 需要生成的数据数量
target_count = 343

# 生成数据
generated_data = []

def generate_random_date():
    """生成随机日期"""
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    random_date = start_date + timedelta(
        seconds=random.randint(0, int((end_date - start_date).total_seconds()))
    )
    return random_date.strftime("%Y-%m-%d")

def generate_oos_query():
    """生成一个OOS查询"""
    template_type = random.choice(list(oos_templates.keys()))
    template = random.choice(oos_templates[template_type])

    # 根据模板类型填充参数
    if "{company}" in template:
        company = random.choice(companies)
        query = template.replace("{company}", company)
    elif "{company1}" in template and "{company2}" in template:
        company1, company2 = random.sample(companies, 2)
        query = template.replace("{company1}", company1).replace("{company2}", company2)
    elif "{sector}" in template:
        sector = random.choice(sectors)
        query = template.replace("{sector}", sector)
    elif "XX万" in template:
        amount = random.randint(10, 1000)
        query = template.replace("XX万", f"{amount}万")
    else:
        query = template

    # 添加一些随机的语气词和表达方式
    if random.random() < 0.3:
        prefixes = ["请问", "我想", "帮我", "能不能", "能不能帮我", "麻烦问一下"]
        suffixes = ["？", "啊？", "呢？", "一下？", "看看？"]
        query = random.choice(prefixes) + query + random.choice(suffixes)

    return query

# 生成目标数量的数据
while len(generated_data) < target_count:
    query = generate_oos_query()

    # 确保查询长度在合理范围内
    if 10 <= len(query) <= 100:
        entry = {
            "query": query,
            "category": "OOS",
            "cost_label": 50
        }
        generated_data.append(entry)

# 保存到文件
output_file = "/home/iilab9/scholar-papers/experiments/intention/exp-2/exp/data/data_labeling/checkpoints/expanded_D3_A2_OOS.jsonl"

# 追加模式写入
with open(output_file, "a", encoding="utf-8") as f:
    for entry in generated_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"成功生成 {len(generated_data)} 条OOS数据")
print(f"已追加到文件: {output_file}")
print("生成完成！")