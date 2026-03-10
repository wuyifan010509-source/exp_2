#!/usr/bin/env python3
"""
补充剩余缺口数据
"""

import json
import random
from pathlib import Path

def generate_gap_data():
    """生成缺口数据"""

    # A2: 交易规则咨询 (还需 70 条)
    A2_TEMPLATES = [
        # 开户流程 (10)
        {"query": "开户需要准备什么资料？", "category": "A2", "cost_label": 50},
        {"query": "创业板怎么开通权限？", "category": "A2", "cost_label": 50},
        {"query": "融资融券怎么申请？", "category": "A2", "cost_label": 50},
        {"query": "期权交易有什么门槛？", "category": "A2", "cost_label": 50},
        {"query": "港股通怎么开通？", "category": "A2", "cost_label": 50},
        {"query": "北交所开户条件是什么？", "category": "A2", "cost_label": 50},
        {"query": "退市整理期股票还能交易吗？", "category": "A2", "cost_label": 50},
        {"query": "ST股票的交易规则是什么？", "category": "A2", "cost_label": 50},
        {"query": "科创板打新需要多少市值？", "category": "A2", "cost_label": 50},
        {"query": "可转债转股怎么操作？", "category": "A2", "cost_label": 50},

        # 交易规则 (10)
        {"query": "涨跌幅限制是多少？", "category": "A2", "cost_label": 50},
        {"query": "集合竞价是什么时候？", "category": "A2", "cost_label": 50},
        {"query": "龙虎榜交易规则？", "category": "A2", "cost_label": 50},
        {"query": "大宗交易怎么操作？", "category": "A2", "cost_label": 50},
        {"query": "盘中停牌规则？", "category": "A2", "cost_label": 50},
        {"query": "新三板交易时间？", "category": "A2", "cost_label": 50},
        {"query": "港股交易时间跟A股一样吗？", "category": "A2", "cost_label": 50},
        {"query": "美股交易有休市吗？", "category": "A2", "cost_label": 50},
        {"query": "债券交易有涨跌幅限制吗？", "category": "A2", "cost_label": 50},
        {"query": "ETF交易有特殊规则吗？", "category": "A2", "cost_label": 50},

        # 费用相关 (10)
        {"query": "交易佣金怎么算？", "category": "A2", "cost_label": 50},
        {"query": "印花税现在多少？", "category": "A2", "cost_label": 50},
        {"query": "融资融券利息多少？", "category": "A2", "cost_label": 50},
        {"query": "港股通过桥费？", "category": "A2", "cost_label": 50},
        {"query": "期权手续费怎么算？", "category": "A2", "cost_label": 50},
        {"query": "可转债费用怎么算？", "category": "A2", "cost_label": 50},
        {"query": "账户管理费怎么收？", "category": "A2", "cost_label": 50},
        {"query": "资金转账有手续费吗？", "category": "A2", "cost_label": 50},
        {"query": "ETF交易费率优惠？", "category": "A2", "cost_label": 50},
        {"query": "国债逆回购利息怎么算？", "category": "A2", "cost_label": 50},

        # 交易操作 (10)
        {"query": "撤单要手续费吗？", "category": "A2", "cost_label": 50},
        {"query": "市价单和限价单区别？", "category": "A2", "cost_label": 50},
        {"query": "止损止盈怎么设置？", "category": "A2", "cost_label": 50},
        {"query": "条件单怎么用？", "category": "A2", "cost_label": 50},
        {"query": "网格交易怎么设置？", "category": "A2", "cost_label": 50},
        {"query": "股票复权是什么意思？", "category": "A2", "cost_label": 50},
        {"query": "分红派息规则？", "category": "A2", "cost_label": 50},
        {"query": "股票除权除息怎么算？", "category": "A2", "cost_label": 50},
        {"query": "新股中签后怎么操作？", "category": "A2", "cost_label": 50},
        {"query": "可转债中签后怎么缴款？", "category": "A2", "cost_label": 50},

        # 特殊产品 (30)
        {"query": "分级基金怎么交易？", "category": "A2", "cost_label": 50},
        {"query": "LOF基金和ETF区别？", "category": "A2", "cost_label": 50},
        {"query": "公募基金申购赎回时间？", "category": "A2", "cost_label": 50},
        {"query": "私募基金认购门槛？", "category": "A2", "cost_label": 50},
        {"query": "银行理财怎么买？", "category": "A2", "cost_label": 50},
        {"query": "信托产品怎么选？", "category": "A2", "cost_label": 50},
        {"query": "债券逆回购怎么买？", "category": "A2", "cost_label": 50},
        {"query": "国债怎么买？", "category": "A2", "cost_label": 50},
        {"query": "可转债打新怎么操作？", "category": "A2", "cost_label": 50},
        {"query": "期权怎么行权？", "category": "A2", "cost_label": 50},
        {"query": "期货怎么开户？", "category": "A2", "cost_label": 50},
        {"query": "外汇交易规则？", "category": "A2", "cost_label": 50},
        {"query": "黄金怎么投资？", "category": "A2", "cost_label": 50},
        {"query": "原油宝是什么？", "category": "A2", "cost_label": 50},
        {"query": "股指期货怎么买？", "category": "A2", "cost_label": 50},
        {"query": "国债期货怎么交易？", "category": "A2", "cost_label": 50},
        {"query": "商品期货保证金比例？", "category": "A2", "cost_label": 50},
        {"query": "期货合约到期怎么处理？", "category": "A2", "cost_label": 50},
        {"query": "期权杠杆多少倍？", "category": "A2", "cost_label": 50},
        {"query": "期权希腊字母什么意思？", "category": "A2", "cost_label": 50},
        {"query": "波动率怎么算？", "category": "A2", "cost_label": 50},
        {"query": "隐含波动率怎么用？", "category": "A2", "cost_label": 50},
        {"query": "期权套利策略？", "category": "A2", "cost_label": 50},
        {"query": "期货套保怎么做？", "category": "A2", "cost_label": 50},
        {"query": "期现套利怎么做？", "category": "A2", "cost_label": 50},
        {"query": "跨期套利怎么做？", "category": "A2", "cost_label": 50},
        {"query": "期权卖方风险？", "category": "A2", "cost_label": 50},
        {"query": "期货爆仓怎么计算？", "category": "A2", "cost_label": 50},
    ]

    # OOS: 超出范围 (还需 107 条)
    OOS_TEMPLATES = [
        # 个股分析 (30)
        {"query": "分析一下贵州茅台的投资价值", "category": "OOS", "cost_label": 50},
        {"query": "比亚迪未来三年前景怎么样？", "category": "OOS", "cost_label": 50},
        {"query": "腾讯控股现在能买吗？", "category": "OOS", "cost_label": 50},
        {"query": "宁德时代的竞争力分析", "category": "OOS", "cost_label": 50},
        {"query": "招商银行的财务状况如何？", "category": "OOS", "cost_label": 50},
        {"query": "中国移动未来发展趋势", "category": "OOS", "cost_label": 50},
        {"query": "阿里巴巴的护城河在哪里？", "category": "OOS", "cost_label": 50},
        {"query": "美团外卖业务前景分析", "category": "OOS", "cost_label": 50},
        {"query": "小米汽车的市场竞争力", "category": "OOS", "cost_label": 50},
        {"query": "华为手机业务的未来", "category": "OOS", "cost_label": 50},
        {"query": "中石油的投资价值分析", "category": "OOS", "cost_label": 50},
        {"query": "工商银行能长期持有吗？", "category": "OOS", "cost_label": 50},
        {"query": "中国平安的估值分析", "category": "OOS", "cost_label": 50},
        {"query": "五粮液的竞争优势", "category": "OOS", "cost_label": 50},
        {"query": "海天味业的增长潜力", "category": "OOS", "cost_label": 50},
        {"query": "格力电器的盈利能力", "category": "OOS", "cost_label": 50},
        {"query": "美的集团的多元化布局", "category": "OOS", "cost_label": 50},
        {"query": "万科地产的未来走向", "category": "OOS", "cost_label": 50},
        {"query": "比亚迪刀片电池技术分析", "category": "OOS", "cost_label": 50},
        {"query": "宁德时代的技术壁垒", "category": "OOS", "cost_label": 50},
        {"query": "隆基绿能的光伏业务", "category": "OOS", "cost_label": 50},
        {"query": "通威股份的硅料业务", "category": "OOS", "cost_label": 50},
        {"query": "天齐锂业的锂资源", "category": "OOS", "cost_label": 50},
        {"query": "赣锋锂业的产能布局", "category": "OOS", "cost_label": 50},
        {"query": "药明康德的医药研发", "category": "OOS", "cost_label": 50},
        {"query": "恒瑞医药的创新药", "category": "OOS", "cost_label": 50},
        {"query": "迈瑞医疗的器械业务", "category": "OOS", "cost_label": 50},
        {"query": "爱尔眼科的扩张模式", "category": "OOS", "cost_label": 50},
        {"query": "通策医疗的口腔连锁", "category": "OOS", "cost_label": 50},
        {"query": "片仔癀的中药业务", "category": "OOS", "cost_label": 50},

        # 市场预测 (30)
        {"query": "明天大盘会涨还是跌？", "category": "OOS", "cost_label": 50},
        {"query": "下周股市走势预测", "category": "OOS", "cost_label": 50},
        {"query": "本月会突破3500点吗？", "category": "OOS", "cost_label": 50},
        {"query": "A股牛市什么时候来？", "category": "OOS", "cost_label": 50},
        {"query": "港股未来走势如何？", "category": "OOS", "cost_label": 50},
        {"query": "美股还会跌多久？", "category": "OOS", "cost_label": 50},
        {"query": "科技股现在能买吗？", "category": "OOS", "cost_label": 50},
        {"query": "新能源板块还能涨吗？", "category": "OOS", "cost_label": 50},
        {"query": "医药股什么时候反弹？", "category": "OOS", "cost_label": 50},
        {"query": "地产股有投资价值吗？", "category": "OOS", "cost_label": 50},
        {"query": "银行股适合长期持有吗？", "category": "OOS", "cost_label": 50},
        {"query": "保险股现在怎么样？", "category": "OOS", "cost_label": 50},
        {"query": "券商股后市怎么看？", "category": "OOS", "cost_label": 50},
        {"query": "军工板块有投资机会吗？", "category": "OOS", "cost_label": 50},
        {"query": "半导体行业前景分析", "category": "OOS", "cost_label": 50},
        {"query": "消费股现在能买吗？", "category": "OOS", "cost_label": 50},
        {"query": "白酒股未来怎么走？", "category": "OOS", "cost_label": 50},
        {"query": "光伏板块还能涨多久？", "category": "OOS", "cost_label": 50},
        {"query": "风电行业投资机会", "category": "OOS", "cost_label": 50},
        {"query": "储能板块未来走势", "category": "OOS", "cost_label": 50},
        {"query": "氢能源产业前景", "category": "OOS", "cost_label": 50},
        {"query": "AI概念股还能炒吗？", "category": "OOS", "cost_label": 50},
        {"query": "元宇宙相关股票分析", "category": "OOS", "cost_label": 50},
        {"query": "区块链股票投资价值", "category": "OOS", "cost_label": 50},
        {"query": "数字货币板块分析", "category": "OOS", "cost_label": 50},
        {"query": "5G概念股还有机会吗？", "category": "OOS", "cost_label": 50},
        {"query": "6G概念股能买吗？", "category": "OOS", "cost_label": 50},
        {"query": "量子计算概念股分析", "category": "OOS", "cost_label": 50},
        {"query": "机器人板块投资机会", "category": "OOS", "cost_label": 50},

        # 板块分析 (47)
        {"query": "分析一下白酒行业", "category": "OOS", "cost_label": 50},
        {"query": "医药板块龙头股有哪些？", "category": "OOS", "cost_label": 50},
        {"query": "新能源板块估值分析", "category": "OOS", "cost_label": 50},
        {"query": "光伏产业链分析", "category": "OOS", "cost_label": 50},
        {"query": "储能行业竞争格局", "category": "OOS", "cost_label": 50},
        {"query": "锂电池产业链梳理", "category": "OOS", "cost_label": 50},
        {"query": "芯片半导体板块分析", "category": "OOS", "cost_label": 50},
        {"query": "汽车整车板块对比", "category": "OOS", "cost_label": 50},
        {"query": "房地产板块现状", "category": "OOS", "cost_label": 50},
        {"query": "银行板块投资逻辑", "category": "OOS", "cost_label": 50},
        {"query": "券商板块后市展望", "category": "OOS", "cost_label": 50},
        {"query": "保险板块价值分析", "category": "OOS", "cost_label": 50},
        {"query": "军工板块投资机会", "category": "OOS", "cost_label": 50},
        {"query": "消费电子板块分析", "category": "OOS", "cost_label": 50},
        {"query": "家电板块竞争格局", "category": "OOS", "cost_label": 50},
        {"query": "食品饮料板块分析", "category": "OOS", "cost_label": 50},
        {"query": "纺织服装板块现状", "category": "OOS", "cost_label": 50},
        {"query": "化工板块龙头分析", "category": "OOS", "cost_label": 50},
        {"query": "钢铁板块周期性分析", "category": "OOS", "cost_label": 50},
        {"query": "煤炭板块投资价值", "category": "OOS", "cost_label": 50},
        {"query": "有色金属板块分析", "category": "OOS", "cost_label": 50},
        {"query": "石油石化板块现状", "category": "OOS", "cost_label": 50},
        {"query": "电力板块投资机会", "category": "OOS", "cost_label": 50},
        {"query": "公用事业板块分析", "category": "OOS", "cost_label": 50},
        {"query": "交通运输板块对比", "category": "OOS", "cost_label": 50},
        {"query": "快递物流行业分析", "category": "OOS", "cost_label": 50},
        {"query": "零售连锁板块现状", "category": "OOS", "cost_label": 50},
        {"query": "旅游酒店板块分析", "category": "OOS", "cost_label": 50},
        {"query": "教育板块政策影响", "category": "OOS", "cost_label": 50},
        {"query": "传媒娱乐板块分析", "category": "OOS", "cost_label": 50},
        {"query": "游戏板块未来走势", "category": "OOS", "cost_label": 50},
        {"query": "影视板块投资机会", "category": "OOS", "cost_label": 50},
        {"query": "出版传媒板块分析", "category": "OOS", "cost_label": 50},
        {"query": "广告营销板块现状", "category": "OOS", "cost_label": 50},
        {"query": "环保板块投资价值", "category": "OOS", "cost_label": 50},
        {"query": "水务板块分析", "category": "OOS", "cost_label": 50},
        {"query": "固废处理行业现状", "category": "OOS", "cost_label": 50},
        {"query": "大气治理板块分析", "category": "OOS", "cost_label": 50},
        {"query": "水务工程板块投资", "category": "OOS", "cost_label": 50},
        {"query": "城投板块风险分析", "category": "OOS", "cost_label": 50},
        {"query": "地方债板块分析", "category": "OOS", "cost_label": 50},
    ]

    # B1: 非业务闲聊 (还需 50 条)
    B1_TEMPLATES = [
        {"query": "最近有什么好电影推荐？", "category": "B1", "cost_label": 1},
        {"query": "今天天气怎么样？", "category": "B1", "cost_label": 1},
        {"query": "你是真人吗？", "category": "B1", "cost_label": 1},
        {"query": "你多大了？", "category": "B1", "cost_label": 1},
        {"query": "你从哪里来？", "category": "B1", "cost_label": 1},
        {"query": "你会做什么？", "category": "B1", "cost_label": 1},
        {"query": "你叫什么名字？", "category": "B1", "cost_label": 1},
        {"query": "你是男的还是女的？", "category": "B1", "cost_label": 1},
        {"query": "你有感情吗？", "category": "B1", "cost_label": 1},
        {"query": "你喜欢什么？", "category": "B1", "cost_label": 1},
        {"query": "你现在在干什么？", "category": "B1", "cost_label": 1},
        {"query": "你能陪我聊天吗？", "category": "B1", "cost_label": 1},
        {"query": "我好无聊，陪我说说话", "category": "B1", "cost_label": 1},
        {"query": "今天过得怎么样？", "category": "B1", "cost_label": 1},
        {"query": "最近有什么新闻？", "category": "B1", "cost_label": 1},
        {"query": "你听说过这个吗？", "category": "B1", "cost_label": 1},
        {"query": "你认识谁谁谁吗？", "category": "B1", "cost_label": 1},
        {"query": "你喜欢吃什么？", "category": "B1", "cost_label": 1},
        {"query": "你有什么爱好？", "category": "B1", "cost_label": 1},
        {"query": "你平时做什么？", "category": "B1", "cost_label": 1},
        {"query": "你忙吗？", "category": "B1", "cost_label": 1},
        {"query": "你休息了吗？", "category": "B1", "cost_label": 1},
        {"query": "你吃饭了吗？", "category": "B1", "cost_label": 1},
        {"query": "你睡觉了吗？", "category": "B1", "cost_label": 1},
        {"query": "你累吗？", "category": "B1", "cost_label": 1},
        {"query": "你开心吗？", "category": "B1", "cost_label": 1},
        {"query": "你难过吗？", "category": "B1", "cost_label": 1},
        {"query": "你生气吗？", "category": "B1", "cost_label": 1},
        {"query": "你会生气吗？", "category": "B1", "cost_label": 1},
        {"query": "你会难过吗？", "category": "B1", "cost_label": 1},
        {"query": "你有什么梦想？", "category": "B1", "cost_label": 1},
        {"query": "你有什么愿望？", "category": "B1", "cost_label": 1},
        {"query": "你有什么目标？", "category": "B1", "cost_label": 1},
        {"query": "你有什么计划？", "category": "B1", "cost_label": 1},
        {"query": "你有什么打算？", "category": "B1", "cost_label": 1},
        {"query": "你有什么安排？", "category": "B1", "cost_label": 1},
        {"query": "你有什么故事？", "category": "B1", "cost_label": 1},
        {"query": "你有什么经历？", "category": "B1", "cost_label": 1},
        {"query": "你有什么经验？", "category": "B1", "cost_label": 1},
        {"query": "你有什么建议？", "category": "B1", "cost_label": 1},
        {"query": "你有什么想法？", "category": "B1", "cost_label": 1},
        {"query": "你有什么看法？", "category": "B1", "cost_label": 1},
        {"query": "你有什么观点？", "category": "B1", "cost_label": 1},
        {"query": "你有什么意见？", "category": "B1", "cost_label": 1},
        {"query": "你有什么感想？", "category": "B1", "cost_label": 1},
        {"query": "你有什么感受？", "category": "B1", "cost_label": 1},
    ]

    # D3: 差评威胁 (还需 17 条)
    D3_TEMPLATES = [
        {"query": "如果不解决，我就去黑猫投诉！", "category": "D3", "cost_label": 100},
        {"query": "差评已经写好了", "category": "D3", "cost_label": 100},
        {"query": "要给一星评价了", "category": "D3", "cost_label": 100},
        {"query": "微博上曝光你们", "category": "D3", "cost_label": 100},
        {"query": "朋友圈告诉大家别用", "category": "D3", "cost_label": 100},
        {"query": "知乎上写避雷指南", "category": "D3", "cost_label": 100},
        {"query": "B站发视频曝光", "category": "D3", "cost_label": 100},
        {"query": "抖音上吐槽你们", "category": "D3", "cost_label": 100},
        {"query": "小红书上发笔记", "category": "D3", "cost_label": 100},
        {"query": "贴吧上开帖骂你们", "category": "D3", "cost_label": 100},
        {"query": "今日头条上爆料", "category": "D3", "cost_label": 100},
        {"query": "豆瓣上写差评", "category": "D3", "cost_label": 100},
        {"query": "应用商店刷一星", "category": "D3", "cost_label": 100},
        {"query": "12315投诉你们", "category": "D3", "cost_label": 100},
        {"query": "12345举报你们", "category": "D3", "cost_label": 100},
        {"query": "找媒体曝光你们", "category": "D3", "cost_label": 100},
        {"query": "让所有人知道你们的垃圾", "category": "D3", "cost_label": 100},
    ]

    # C2: 严重投诉 (还需 21 条)
    C2_TEMPLATES = [
        {"query": "你们是骗子我要投诉", "category": "C2", "cost_label": 500},
        {"query": "我要去证监会举报你们", "category": "C2", "cost_label": 500},
        {"query": "这系统太烂了要求赔偿", "category": "C2", "cost_label": 500},
        {"query": "你们这是欺诈行为", "category": "C2", "cost_label": 500},
        {"query": "我要找媒体曝光", "category": "C2", "cost_label": 500},
        {"query": "我要把你们告上法庭", "category": "C2", "cost_label": 500},
        {"query": "你们涉嫌非法经营", "category": "C2", "cost_label": 500},
        {"query": "我要向银保监会投诉", "category": "C2", "cost_label": 500},
        {"query": "你们侵犯了我的权益", "category": "C2", "cost_label": 500},
        {"query": "我要申请集体诉讼", "category": "C2", "cost_label": 500},
        {"query": "你们平台有严重漏洞", "category": "C2", "cost_label": 500},
        {"query": "我要向媒体曝光你们的黑幕", "category": "C2", "cost_label": 500},
        {"query": "我要向消协投诉你们", "category": "C2", "cost_label": 500},
        {"query": "你们的服务太差劲了", "category": "C2", "cost_label": 500},
        {"query": "我要找律师起诉你们", "category": "C2", "cost_label": 500},
        {"query": "你们的行为已经违法了", "category": "C2", "cost_label": 500},
        {"query": "我要向监管部门举报", "category": "C2", "cost_label": 500},
        {"query": "你们涉嫌虚假宣传", "category": "C2", "cost_label": 500},
        {"query": "我要在微博上讨伐你们", "category": "C2", "cost_label": 500},
        {"query": "你们这是在坑客户", "category": "C2", "cost_label": 500},
        {"query": "我要让你们付出代价", "category": "C2", "cost_label": 500},
    ]

    # 合并所有缺口数据
    all_gap_data = A2_TEMPLATES + OOS_TEMPLATES + B1_TEMPLATES + D3_TEMPLATES + C2_TEMPLATES

    # 追加到修复后的数据文件
    output_file = Path(__file__).parent / "final_labeled_data_fixed.jsonl"

    with open(output_file, "a", encoding="utf-8") as f:
        for item in all_gap_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 补充数据已生成并追加到: {output_file}")
    print(f"   补充数量: {len(all_gap_data)} 条")
    print(f"   - A2: {len(A2_TEMPLATES)} 条")
    print(f"   - OOS: {len(OOS_TEMPLATES)} 条")
    print(f"   - B1: {len(B1_TEMPLATES)} 条")
    print(f"   - D3: {len(D3_TEMPLATES)} 条")
    print(f"   - C2: {len(C2_TEMPLATES)} 条")

    return all_gap_data

if __name__ == "__main__":
    generate_gap_data()
