#!/usr/bin/env python3
"""
生成扩充数据：B1 (非业务闲聊), B2 (跨业务咨询), A1 (基础业务)
"""

import json
import os
from pathlib import Path

# B1 非业务闲聊数据 (约150条)
B1_DATA = [
    # 天气相关
    {"query": "今天天气怎么样", "category": "B1", "cost_label": 1},
    {"query": "明天会下雨吗", "category": "B1", "cost_label": 1},
    {"query": "最近天气好热啊", "category": "B1", "cost_label": 1},
    {"query": "今天要带伞吗", "category": "B1", "cost_label": 1},
    {"query": "外面冷不冷", "category": "B1", "cost_label": 1},
    {"query": "今天多少度", "category": "B1", "cost_label": 1},
    {"query": "这周天气怎么样", "category": "B1", "cost_label": 1},
    {"query": "这几天太阳好大", "category": "B1", "cost_label": 1},
    {"query": "冬天来了好冷", "category": "B1", "cost_label": 1},
    {"query": "夏天终于到了", "category": "B1", "cost_label": 1},
    {"query": "今天风好大", "category": "B1", "cost_label": 1},
    {"query": "会下雪吗", "category": "B1", "cost_label": 1},
    {"query": "空气质量怎么样", "category": "B1", "cost_label": 1},

    # 笑话娱乐
    {"query": "给我讲个笑话", "category": "B1", "cost_label": 1},
    {"query": "你真幽默", "category": "B1", "cost_label": 1},
    {"query": "哈哈哈太搞笑了", "category": "B1", "cost_label": 1},
    {"query": "再来一个", "category": "B1", "cost_label": 1},
    {"query": "你太逗了", "category": "B1", "cost_label": 1},
    {"query": "讲个冷笑话", "category": "B1", "cost_label": 1},
    {"query": "今天心情不错", "category": "B1", "cost_label": 1},
    {"query": "开心的一天", "category": "B1", "cost_label": 1},
    {"query": "你真可爱", "category": "B1", "cost_label": 1},
    {"query": "聊天很开心", "category": "B1", "cost_label": 1},
    {"query": "你在开玩笑吧", "category": "B1", "cost_label": 1},

    # 心情问候
    {"query": "你好", "category": "B1", "cost_label": 1},
    {"query": "在吗", "category": "B1", "cost_label": 1},
    {"query": "早上好", "category": "B1", "cost_label": 1},
    {"query": "晚上好", "category": "B1", "cost_label": 1},
    {"query": "下午好", "category": "B1", "cost_label": 1},
    {"query": "晚安", "category": "B1", "cost_label": 1},
    {"query": "再见", "category": "B1", "cost_label": 1},
    {"query": "嗨", "category": "B1", "cost_label": 1},
    {"query": "哈喽", "category": "B1", "cost_label": 1},
    {"query": "你叫什么名字", "category": "B1", "cost_label": 1},
    {"query": "你是谁", "category": "B1", "cost_label": 1},
    {"query": "你好啊", "category": "B1", "cost_label": 1},
    {"query": "你好呀", "category": "B1", "cost_label": 1},
    {"query": "早上好啊", "category": "B1", "cost_label": 1},
    {"query": "大家早上好", "category": "B1", "cost_label": 1},

    # 情绪表达
    {"query": "我好累啊", "category": "B1", "cost_label": 1},
    {"query": "今天工作好忙", "category": "B1", "cost_label": 1},
    {"query": "终于下班了", "category": "B1", "cost_label": 1},
    {"query": "周末愉快", "category": "B1", "cost_label": 1},
    {"query": "期待周末", "category": "B1", "cost_label": 1},
    {"query": "又是新的一天", "category": "B1", "cost_label": 1},
    {"query": "心情不太好", "category": "B1", "cost_label": 1},
    {"query": "有点烦躁", "category": "B1", "cost_label": 1},
    {"query": "没事的", "category": "B1", "cost_label": 1},
    {"query": "我会努力的", "category": "B1", "cost_label": 1},
    {"query": "加油", "category": "B1", "cost_label": 1},
    {"query": "相信自己", "category": "B1", "cost_label": 1},

    # 生活话题
    {"query": "今天吃什么", "category": "B1", "cost_label": 1},
    {"query": "中午吃什么", "category": "B1", "cost_label": 1},
    {"query": "想吃火锅", "category": "B1", "cost_label": 1},
    {"query": "饿了", "category": "B1", "cost_label": 1},
    {"query": "好困", "category": "B1", "cost_label": 1},
    {"query": "想睡觉", "category": "B1", "cost_label": 1},
    {"query": "今天要加班", "category": "B1", "cost_label": 1},
    {"query": "不想上班", "category": "B1", "cost_label": 1},
    {"query": "什么时候放假", "category": "B1", "cost_label": 1},
    {"query": "还有几天过年", "category": "B1", "cost_label": 1},
    {"query": "假期要去旅游", "category": "B1", "cost_label": 1},
    {"query": "去哪里玩好", "category": "B1", "cost_label": 1},
    {"query": "推荐个地方", "category": "B1", "cost_label": 1},

    # 赞美/感谢
    {"query": "谢谢", "category": "B1", "cost_label": 1},
    {"query": "谢谢你", "category": "B1", "cost_label": 1},
    {"query": "非常感谢", "category": "B1", "cost_label": 1},
    {"query": "太感谢了", "category": "B1", "cost_label": 1},
    {"query": "你真棒", "category": "B1", "cost_label": 1},
    {"query": "厉害", "category": "B1", "cost_label": 1},
    {"query": "真厉害", "category": "B1", "cost_label": 1},
    {"query": "你真聪明", "category": "B1", "cost_label": 1},
    {"query": "不错", "category": "B1", "cost_label": 1},
    {"query": "挺好的", "category": "B1", "cost_label": 1},

    # 随意聊天
    {"query": "在干嘛", "category": "B1", "cost_label": 1},
    {"query": "你好吗", "category": "B1", "cost_label": 1},
    {"query": "怎么样", "category": "B1", "cost_label": 1},
    {"query": "最近怎么样", "category": "B1", "cost_label": 1},
    {"query": "今天怎么样", "category": "B1", "cost_label": 1},
    {"query": "有什么好玩的", "category": "B1", "cost_label": 1},
    {"query": "无聊啊", "category": "B1", "cost_label": 1},
    {"query": "聊聊天吧", "category": "B1", "cost_label": 1},
    {"query": "随便聊聊", "category": "B1", "cost_label": 1},
    {"query": "没事找你聊", "category": "B1", "cost_label": 1},
    {"query": "随便问问", "category": "B1", "cost_label": 1},
    {"query": "只是问问", "category": "B1", "cost_label": 1},
    {"query": "哈哈", "category": "B1", "cost_label": 1},
    {"query": "嘻嘻", "category": "B1", "cost_label": 1},
    {"query": "呵呵", "category": "B1", "cost_label": 1},
    {"query": "哈哈笑死我了", "category": "B1", "cost_label": 1},
    {"query": "好玩吗", "category": "B1", "cost_label": 1},
    {"query": "有意思", "category": "B1", "cost_label": 1},
    {"query": "还行吧", "category": "B1", "cost_label": 1},
    {"query": "还好", "category": "B1", "cost_label": 1},
    {"query": "差不多", "category": "B1", "cost_label": 1},
    {"query": "就那样", "category": "B1", "cost_label": 1},

    # 时节日话题
    {"query": "新年快乐", "category": "B1", "cost_label": 1},
    {"query": "圣诞快乐", "category": "B1", "cost_label": 1},
    {"query": "节日快乐", "category": "B1", "cost_label": 1},
    {"query": "生日快乐", "category": "B1", "cost_label": 1},
    {"query": "儿童节快乐", "category": "B1", "cost_label": 1},
    {"query": "母亲节快乐", "category": "B1", "cost_label": 1},
    {"query": "今天是什么日子", "category": "B1", "cost_label": 1},
    {"query": "今天几号", "category": "B1", "cost_label": 1},
    {"query": "今天是星期几", "category": "B1", "cost_label": 1},
    {"query": "今天周末吗", "category": "B1", "cost_label": 1},
    {"query": "明天休息吗", "category": "B1", "cost_label": 1},

    # 其他闲聊
    {"query": "你知道今天是什么节日吗", "category": "B1", "cost_label": 1},
    {"query": "今天有什么新闻", "category": "B1", "cost_label": 1},
    {"query": "最近有什么电影", "category": "B1", "cost_label": 1},
    {"query": "推荐首歌", "category": "B1", "cost_label": 1},
    {"query": "好无聊啊", "category": "B1", "cost_label": 1},
    {"query": "没事做", "category": "B1", "cost_label": 1},
    {"query": "随便说点什么", "category": "B1", "cost_label": 1},
    {"query": "跟我说说话", "category": "B1", "cost_label": 1},
    {"query": "陪我聊聊", "category": "B1", "cost_label": 1},
    {"query": "没人聊天", "category": "B1", "cost_label": 1},
    {"query": "找个人说话", "category": "B1", "cost_label": 1},
    {"query": "测试一下", "category": "B1", "cost_label": 1},
    {"query": "试一下", "category": "B1", "cost_label": 1},
    {"query": "看看能不能用", "category": "B1", "cost_label": 1},
    {"query": "这个能用吗", "category": "B1", "cost_label": 1},
    {"query": "随便测试", "category": "B1", "cost_label": 1},

    # 额外补充 B1
    {"query": "今天是个好天气", "category": "B1", "cost_label": 1},
    {"query": "真热啊", "category": "B1", "cost_label": 1},
    {"query": "好冷啊", "category": "B1", "cost_label": 1},
    {"query": "起风了", "category": "B1", "cost_label": 1},
    {"query": "下雨了", "category": "B1", "cost_label": 1},
    {"query": "下雪了", "category": "B1", "cost_label": 1},
    {"query": "出太阳了", "category": "B1", "cost_label": 1},
    {"query": "阴天", "category": "B1", "cost_label": 1},
    {"query": "晴天", "category": "B1", "cost_label": 1},
    {"query": "今天不错", "category": "B1", "cost_label": 1},
    {"query": "今天一般", "category": "B1", "cost_label": 1},
    {"query": "今天不太好", "category": "B1", "cost_label": 1},
    {"query": "还好", "category": "B1", "cost_label": 1},
    {"query": "还行", "category": "B1", "cost_label": 1},
    {"query": "不好不坏", "category": "B1", "cost_label": 1},
    {"query": "就那样吧", "category": "B1", "cost_label": 1},
    {"query": "随便吧", "category": "B1", "cost_label": 1},
    {"query": "无所谓", "category": "B1", "cost_label": 1},
    {"query": "没关系", "category": "B1", "cost_label": 1},
    {"query": "没事", "category": "B1", "cost_label": 1},
    {"query": "不要紧", "category": "B1", "cost_label": 1},
    {"query": "算了", "category": "B1", "cost_label": 1},
    {"query": "好吧", "category": "B1", "cost_label": 1},
    {"query": "嗯", "category": "B1", "cost_label": 1},
    {"query": "啊", "category": "B1", "cost_label": 1},
    {"query": "哦", "category": "B1", "cost_label": 1},
    {"query": "对", "category": "B1", "cost_label": 1},
    {"query": "是", "category": "B1", "cost_label": 1},
]

# B2 跨业务咨询数据 (约100条)
B2_DATA = [
    # 医保相关
    {"query": "医保怎么报销", "category": "B2", "cost_label": 1},
    {"query": "医保卡丢了怎么办", "category": "B2", "cost_label": 1},
    {"query": "异地医保能用吗", "category": "B2", "cost_label": 1},
    {"query": "医保报销比例多少", "category": "B2", "cost_label": 1},
    {"query": "医保卡怎么激活", "category": "B2", "cost_label": 1},
    {"query": "怎么查医保余额", "category": "B2", "cost_label": 1},
    {"query": "医保怎么缴费", "category": "B2", "cost_label": 1},
    {"query": "医保断交了怎么办", "category": "B2", "cost_label": 1},
    {"query": "医保怎么转移", "category": "B2", "cost_label": 1},
    {"query": "医保报销需要什么材料", "category": "B2", "cost_label": 1},

    # 社保相关
    {"query": "社保怎么交", "category": "B2", "cost_label": 1},
    {"query": "社保卡怎么办", "category": "B2", "cost_label": 1},
    {"query": "社保能补交吗", "category": "B2", "cost_label": 1},
    {"query": "社保怎么查询", "category": "B2", "cost_label": 1},
    {"query": "社保缴费比例", "category": "B2", "cost_label": 1},
    {"query": "社保断交有什么影响", "category": "B2", "cost_label": 1},
    {"query": "社保转移怎么办理", "category": "B2", "cost_label": 1},
    {"query": "社保交满多少年退休", "category": "B2", "cost_label": 1},
    {"query": "灵活就业怎么交社保", "category": "B2", "cost_label": 1},
    {"query": "社保个人和公司各交多少", "category": "B2", "cost_label": 1},

    # 养老保险
    {"query": "养老保险怎么交", "category": "B2", "cost_label": 1},
    {"query": "养老金怎么领", "category": "B2", "cost_label": 1},
    {"query": "退休金多少", "category": "B2", "cost_label": 1},
    {"query": "退休年龄是多少", "category": "B2", "cost_label": 1},
    {"query": "养老保险交够多少年", "category": "B2", "cost_label": 1},

    # 其他行业问题
    {"query": "银行利息是多少", "category": "B2", "cost_label": 1},
    {"query": "房贷利率多少", "category": "B2", "cost_label": 1},
    {"query": "怎么申请信用卡", "category": "B2", "cost_label": 1},
    {"query": "信用卡怎么还款", "category": "B2", "cost_label": 1},
    {"query": "公积金怎么用", "category": "B2", "cost_label": 1},
    {"query": "公积金怎么提取", "category": "B2", "cost_label": 1},
    {"query": "公积金贷款利率", "category": "B2", "cost_label": 1},
    {"query": "公积金怎么交", "category": "B2", "cost_label": 1},
    {"query": "公积金转移", "category": "B2", "cost_label": 1},
    {"query": "公积金查询", "category": "B2", "cost_label": 1},

    # 教育相关
    {"query": "幼儿园报名", "category": "B2", "cost_label": 1},
    {"query": "学区房政策", "category": "B2", "cost_label": 1},
    {"query": "入学资格", "category": "B2", "cost_label": 1},
    {"query": "异地高考", "category": "B2", "cost_label": 1},
    {"query": "高考报名条件", "category": "B2", "cost_label": 1},

    # 证件办理
    {"query": "身份证怎么补办", "category": "B2", "cost_label": 1},
    {"query": "护照怎么办理", "category": "B2", "cost_label": 1},
    {"query": "港澳通行证", "category": "B2", "cost_label": 1},
    {"query": "户口迁移", "category": "B2", "cost_label": 1},
    {"query": "居住证怎么办", "category": "B2", "cost_label": 1},
    {"query": "驾驶证换证", "category": "B2", "cost_label": 1},
    {"query": "车辆年检", "category": "B2", "cost_label": 1},
    {"query": "违章查询", "category": "B2", "cost_label": 1},

    # 交通出行
    {"query": "地铁线路", "category": "B2", "cost_label": 1},
    {"query": "公交路线查询", "category": "B2", "cost_label": 1},
    {"query": "高速费怎么交", "category": "B2", "cost_label": 1},
    {"query": "打车价格", "category": "B2", "cost_label": 1},

    # 房产相关
    {"query": "房产证怎么办理", "category": "B2", "cost_label": 1},
    {"query": "房产税怎么交", "category": "B2", "cost_label": 1},
    {"query": "二手房交易流程", "category": "B2", "cost_label": 1},
    {"query": "房屋过户", "category": "B2", "cost_label": 1},

    # 生活服务
    {"query": "水电费怎么交", "category": "B2", "cost_label": 1},
    {"query": "燃气费怎么交", "category": "B2", "cost_label": 1},
    {"query": "网费怎么交", "category": "B2", "cost_label": 1},
    {"query": "手机号怎么注销", "category": "B2", "cost_label": 1},
    {"query": "快递怎么查", "category": "B2", "cost_label": 1},
    {"query": "怎么寄快递", "category": "B2", "cost_label": 1},
    {"query": "快递怎么收费", "category": "B2", "cost_label": 1},
    {"query": "快递延误怎么赔偿", "category": "B2", "cost_label": 1},

    # 法律咨询
    {"query": "怎么打官司", "category": "B2", "cost_label": 1},
    {"query": "律师怎么找", "category": "B2", "cost_label": 1},
    {"query": "法律援助", "category": "B2", "cost_label": 1},
    {"query": "怎么投诉商家", "category": "B2", "cost_label": 1},
    {"query": "消费者权益", "category": "B2", "cost_label": 1},
    {"query": "劳动仲裁", "category": "B2", "cost_label": 1},
    {"query": "怎么申请工伤", "category": "B2", "cost_label": 1},
    {"query": "离职补偿", "category": "B2", "cost_label": 1},
    {"query": "加班费怎么算", "category": "B2", "cost_label": 1},

    # 额外补充 B2
    {"query": "公积金贷款额度", "category": "B2", "cost_label": 1},
    {"query": "社保转移流程", "category": "B2", "cost_label": 1},
    {"query": "医保报销范围", "category": "B2", "cost_label": 1},
    {"query": "养老保险缴费标准", "category": "B2", "cost_label": 1},
    {"query": "失业保险怎么领", "category": "B2", "cost_label": 1},
    {"query": "工伤保险认定", "category": "B2", "cost_label": 1},
    {"query": "生育保险报销", "category": "B2", "cost_label": 1},
    {"query": "社保断交影响", "category": "B2", "cost_label": 1},
    {"query": "异地社保怎么交", "category": "B2", "cost_label": 1},
    {"query": "灵活就业社保", "category": "B2", "cost_label": 1},
    {"query": "社保查询网站", "category": "B2", "cost_label": 1},
    {"query": "医保异地结算", "category": "B2", "cost_label": 1},
    {"query": "门诊怎么报销", "category": "B2", "cost_label": 1},
    {"query": "住院报销比例", "category": "B2", "cost_label": 1},
    {"query": "医保定点医院", "category": "B2", "cost_label": 1},
    {"query": "特殊病种报销", "category": "B2", "cost_label": 1},
    {"query": "大病保险报销", "category": "B2", "cost_label": 1},
    {"query": "医保个人账户", "category": "B2", "cost_label": 1},
    {"query": "医保家庭账户", "category": "B2", "cost_label": 1},
    {"query": "生育津贴怎么领", "category": "B2", "cost_label": 1},
    {"query": "产假多少天", "category": "B2", "cost_label": 1},
    {"query": "陪产假", "category": "B2", "cost_label": 1},
    {"query": "育儿假", "category": "B2", "cost_label": 1},
    {"query": "产检报销", "category": "B2", "cost_label": 1},
    {"query": "分娩报销", "category": "B2", "cost_label": 1},
    {"query": "哺乳假", "category": "B2", "cost_label": 1},
    {"query": "病假怎么算工资", "category": "B2", "cost_label": 1},
    {"query": "事假工资怎么算", "category": "B2", "cost_label": 1},
]

# A1 基础业务数据 (约150条)
A1_DATA = [
    # 账户信息
    {"query": "怎么查看账户信息", "category": "A1", "cost_label": 1},
    {"query": "我的账户余额多少", "category": "A1", "cost_label": 1},
    {"query": "怎么查持仓", "category": "A1", "cost_label": 1},
    {"query": "我的资产在哪里看", "category": "A1", "cost_label": 1},
    {"query": "账户总资产怎么查", "category": "A1", "cost_label": 1},
    {"query": "怎么查我的收益", "category": "A1", "cost_label": 1},
    {"query": "交易记录怎么查", "category": "A1", "cost_label": 1},
    {"query": "历史订单在哪里", "category": "A1", "cost_label": 1},
    {"query": "怎么查我的可用资金", "category": "A1", "cost_label": 1},
    {"query": "我的持仓成本是多少", "category": "A1", "cost_label": 1},
    {"query": "怎么查今日盈亏", "category": "A1", "cost_label": 1},
    {"query": "账户流水怎么查", "category": "A1", "cost_label": 1},
    {"query": "资金流水在哪里", "category": "A1", "cost_label": 1},

    # 行情查询
    {"query": "今天的股票行情怎么样", "category": "A1", "cost_label": 1},
    {"query": "大盘涨了多少", "category": "A1", "cost_label": 1},
    {"query": "今天沪指多少", "category": "A1", "cost_label": 1},
    {"query": "深指现在多少点", "category": "A1", "cost_label": 1},
    {"query": "创业板指数", "category": "A1", "cost_label": 1},
    {"query": "今天大盘涨跌", "category": "A1", "cost_label": 1},
    {"query": "北向资金流入多少", "category": "A1", "cost_label": 1},
    {"query": "成交量多少", "category": "A1", "cost_label": 1},
    {"query": "市场热度如何", "category": "A1", "cost_label": 1},
    {"query": "个股行情怎么查", "category": "A1", "cost_label": 1},
    {"query": "这只股票现在多少钱", "category": "A1", "cost_label": 1},
    {"query": "今天涨跌幅多少", "category": "A1", "cost_label": 1},
    {"query": "这只股票的最新价格", "category": "A1", "cost_label": 1},
    {"query": "股票代码怎么查", "category": "A1", "cost_label": 1},
    {"query": "怎么查K线图", "category": "A1", "cost_label": 1},

    # APP操作
    {"query": "APP怎么下载", "category": "A1", "cost_label": 1},
    {"query": "手机怎么安装软件", "category": "A1", "cost_label": 1},
    {"query": "APP打不开怎么办", "category": "A1", "cost_label": 1},
    {"query": "APP闪退了", "category": "A1", "cost_label": 1},
    {"query": "APP卡住了", "category": "A1", "cost_label": 1},
    {"query": "APP加载很慢", "category": "A1", "cost_label": 1},
    {"query": "怎么更新APP", "category": "A1", "cost_label": 1},
    {"query": "APP在哪里下载", "category": "A1", "cost_label": 1},
    {"query": "苹果手机怎么装APP", "category": "A1", "cost_label": 1},
    {"query": "安卓手机怎么装APP", "category": "A1", "cost_label": 1},
    {"query": "怎么注册账号", "category": "A1", "cost_label": 1},
    {"query": "注册流程是什么", "category": "A1", "cost_label": 1},
    {"query": "账号怎么激活", "category": "A1", "cost_label": 1},
    {"query": "怎么完善个人信息", "category": "A1", "cost_label": 1},
    {"query": "个人资料在哪里修改", "category": "A1", "cost_label": 1},

    # 开户相关
    {"query": "怎么开户", "category": "A1", "cost_label": 1},
    {"query": "开户需要什么资料", "category": "A1", "cost_label": 1},
    {"query": "开户流程是怎样的", "category": "A1", "cost_label": 1},
    {"query": "网上怎么开户", "category": "A1", "cost_label": 1},
    {"query": "手机能开户吗", "category": "A1", "cost_label": 1},
    {"query": "开户要多少钱", "category": "A1", "cost_label": 1},
    {"query": "开户有门槛吗", "category": "A1", "cost_label": 1},
    {"query": "开户需要多长时间", "category": "A1", "cost_label": 1},
    {"query": "开户后什么时候能交易", "category": "A1", "cost_label": 1},
    {"query": "没有银行卡能开户吗", "category": "A1", "cost_label": 1},
    {"query": "未成年人能开户吗", "category": "A1", "cost_label": 1},
    {"query": "开几个户", "category": "A1", "cost_label": 1},

    # 密码相关
    {"query": "密码忘了怎么办", "category": "A1", "cost_label": 1},
    {"query": "怎么重置密码", "category": "A1", "cost_label": 1},
    {"query": "登录密码怎么改", "category": "A1", "cost_label": 1},
    {"query": "交易密码怎么改", "category": "A1", "cost_label": 1},
    {"query": "密码记不住了", "category": "A1", "cost_label": 1},
    {"query": "密码输入错误太多", "category": "A1", "cost_label": 1},
    {"query": "密码输入错误账号被锁了", "category": "A1", "cost_label": 1},
    {"query": "账号解锁怎么办", "category": "A1", "cost_label": 1},
    {"query": "怎么找回密码", "category": "A1", "cost_label": 1},
    {"query": "密码重置需要什么", "category": "A1", "cost_label": 1},

    # 简单操作
    {"query": "怎么买入股票", "category": "A1", "cost_label": 1},
    {"query": "怎么卖出股票", "category": "A1", "cost_label": 1},
    {"query": "交易界面在哪里", "category": "A1", "cost_label": 1},
    {"query": "怎么下单", "category": "A1", "cost_label": 1},
    {"query": "限价单怎么下", "category": "A1", "cost_label": 1},
    {"query": "市价单怎么下", "category": "A1", "cost_label": 1},
    {"query": "怎么撤销订单", "category": "A1", "cost_label": 1},
    {"query": "订单怎么撤", "category": "A1", "cost_label": 1},
    {"query": "成交在哪里看", "category": "A1", "cost_label": 1},
    {"query": "待成交订单在哪里", "category": "A1", "cost_label": 1},
    {"query": "交易时间是什么时候", "category": "A1", "cost_label": 1},
    {"query": "周末能交易吗", "category": "A1", "cost_label": 1},
    {"query": "节假日怎么交易", "category": "A1", "cost_label": 1},

    # 费用相关
    {"query": "手续费多少", "category": "A1", "cost_label": 1},
    {"query": "交易费率是多少", "category": "A1", "cost_label": 1},
    {"query": "佣金怎么算", "category": "A1", "cost_label": 1},
    {"query": "印花税怎么收", "category": "A1", "cost_label": 1},
    {"query": "过户费是多少", "category": "A1", "cost_label": 1},
    {"query": "费用怎么算", "category": "A1", "cost_label": 1},
    {"query": "最低手续费多少", "category": "A1", "cost_label": 1},

    # 资金相关
    {"query": "怎么充值", "category": "A1", "cost_label": 1},
    {"query": "怎么提现", "category": "A1", "cost_label": 1},
    {"query": "资金怎么入金", "category": "A1", "cost_label": 1},
    {"query": "资金怎么出金", "category": "A1", "cost_label": 1},
    {"query": "银证转账怎么做", "category": "A1", "cost_label": 1},
    {"query": "转账需要多久", "category": "A1", "cost_label": 1},
    {"query": "提现多久到账", "category": "A1", "cost_label": 1},
    {"query": "充值多久到账", "category": "A1", "cost_label": 1},
    {"query": "怎么绑卡", "category": "A1", "cost_label": 1},
    {"query": "银行卡怎么解绑", "category": "A1", "cost_label": 1},
    {"query": "能绑定几张卡", "category": "A1", "cost_label": 1},
    {"query": "转账限额是多少", "category": "A1", "cost_label": 1},

    # 其他简单操作
    {"query": "自选股怎么添加", "category": "A1", "cost_label": 1},
    {"query": "怎么删除自选股", "category": "A1", "cost_label": 1},
    {"query": "自选股在哪里看", "category": "A1", "cost_label": 1},
    {"query": "怎么设置提醒", "category": "A1", "cost_label": 1},
    {"query": "价格提醒怎么设置", "category": "A1", "cost_label": 1},
    {"query": "消息提醒怎么开", "category": "A1", "cost_label": 1},
    {"query": "怎么关闭提醒", "category": "A1", "cost_label": 1},
    {"query": "个人中心在哪里", "category": "A1", "cost_label": 1},
    {"query": "设置在哪里", "category": "A1", "cost_label": 1},
    {"query": "怎么联系客服", "category": "A1", "cost_label": 1},
    {"query": "人工客服怎么找", "category": "A1", "cost_label": 1},
    {"query": "在线客服在哪里", "category": "A1", "cost_label": 1},
    {"query": "客服电话多少", "category": "A1", "cost_label": 1},

    # 额外补充 A1
    {"query": "账户信息在哪看", "category": "A1", "cost_label": 1},
    {"query": "怎么查我的资产", "category": "A1", "cost_label": 1},
    {"query": "持仓盈亏怎么看", "category": "A1", "cost_label": 1},
    {"query": "今日盈亏在哪里", "category": "A1", "cost_label": 1},
    {"query": "历史盈亏怎么查", "category": "A1", "cost_label": 1},
    {"query": "总资产怎么看", "category": "A1", "cost_label": 1},
    {"query": "可用资金怎么查", "category": "A1", "cost_label": 1},
    {"query": "我的市值是多少", "category": "A1", "cost_label": 1},
    {"query": "持仓成本价", "category": "A1", "cost_label": 1},
    {"query": "持仓数量怎么查", "category": "A1", "cost_label": 1},
    {"query": "我有几只股票", "category": "A1", "cost_label": 1},
    {"query": "股票今天涨跌", "category": "A1", "cost_label": 1},
    {"query": "大盘今天怎么样", "category": "A1", "cost_label": 1},
    {"query": "指数是多少", "category": "A1", "cost_label": 1},
    {"query": "沪指点位", "category": "A1", "cost_label": 1},
    {"query": "深证成指", "category": "A1", "cost_label": 1},
    {"query": "创业板指", "category": "A1", "cost_label": 1},
    {"query": "上证50", "category": "A1", "cost_label": 1},
    {"query": "沪深300", "category": "A1", "cost_label": 1},
    {"query": "中证500", "category": "A1", "cost_label": 1},
    {"query": "个股行情在哪里", "category": "A1", "cost_label": 1},
    {"query": "怎么查股票价格", "category": "A1", "cost_label": 1},
    {"query": "现在多少价格", "category": "A1", "cost_label": 1},
    {"query": "涨跌多少", "category": "A1", "cost_label": 1},
    {"query": "涨幅百分比", "category": "A1", "cost_label": 1},
    {"query": "跌幅百分比", "category": "A1", "cost_label": 1},
    {"query": "开盘价是多少", "category": "A1", "cost_label": 1},
    {"query": "收盘价是多少", "category": "A1", "cost_label": 1},
    {"query": "最高价", "category": "A1", "cost_label": 1},
    {"query": "最低价", "category": "A1", "cost_label": 1},
    {"query": "成交量多大", "category": "A1", "cost_label": 1},
    {"query": "成交额多少", "category": "A1", "cost_label": 1},
    {"query": "换手率", "category": "A1", "cost_label": 1},
    {"query": "市盈率", "category": "A1", "cost_label": 1},
    {"query": "市净率", "category": "A1", "cost_label": 1},
    {"query": "总股本", "category": "A1", "cost_label": 1},
    {"query": "流通股本", "category": "A1", "cost_label": 1},
    {"query": "K线图怎么放大", "category": "A1", "cost_label": 1},
    {"query": "K线图怎么缩小", "category": "A1", "cost_label": 1},
    {"query": "分时图", "category": "A1", "cost_label": 1},
    {"query": "日K线", "category": "A1", "cost_label": 1},
    {"query": "周K线", "category": "A1", "cost_label": 1},
    {"query": "月K线", "category": "A1", "cost_label": 1},
]

def main():
    # 确保输出目录存在
    output_dir = Path("/home/iilab9/scholar-papers/experiments/intention/exp-2/exp/data/data_labeling/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 合并所有数据
    all_data = B1_DATA + B2_DATA + A1_DATA

    # 保存为JSONL文件
    output_path = output_dir / "expanded_B1_B2_A1.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 统计各类别数量
    b1_count = len(B1_DATA)
    b2_count = len(B2_DATA)
    a1_count = len(A1_DATA)
    total_count = len(all_data)

    print("=" * 50)
    print("数据扩充完成")
    print("=" * 50)
    print(f"B1 (非业务闲聊): {b1_count} 条")
    print(f"B2 (跨业务咨询): {b2_count} 条")
    print(f"A1 (基础业务): {a1_count} 条")
    print(f"总计: {total_count} 条")
    print(f"保存路径: {output_path}")
    print("=" * 50)

    # 输出样本
    print("\nB1 样本 (非业务闲聊):")
    for i, item in enumerate(B1_DATA[:5]):
        print(f"  {i+1}. {item['query']}")

    print("\nB2 样本 (跨业务咨询):")
    for i, item in enumerate(B2_DATA[:5]):
        print(f"  {i+1}. {item['query']}")

    print("\nA1 样本 (基础业务):")
    for i, item in enumerate(A1_DATA[:5]):
        print(f"  {i+1}. {item['query']}")

if __name__ == "__main__":
    main()
