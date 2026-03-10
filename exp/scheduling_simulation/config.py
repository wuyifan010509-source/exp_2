"""
人机混合智能体调度仿真系统
Human-AI Hybrid Agent Scheduling Simulation System

核心组件：
1. M/M/c 排队模型 - 模拟人类专家队列
2. 四种调度策略对比
3. 实时成本评估
"""

# 代价配置（统一量纲：元）
COST_CONFIG = {
    # 错误代价（业务损失）- 用户指定权重
    "error_cost": {
        "low": 1,      # 低风险问题答错损失
        "mid": 10,     # 中风险问题答错损失  
        "high": 200,   # 高风险问题答错损失
    },
    # 延迟代价（客户流失成本，每毫秒）
    # 假设等待30秒导致客户不满意，成本约等于mid错误
    "latency_cost_per_ms": 0.005,  # 每秒5元 = 每毫秒0.005元
    # 人工固定成本（每次介入）
    "human_fixed_cost": 25,  # 每次人工服务成本
}

# 排队模型参数（更真实的客服场景）
QUEUE_CONFIG = {
    "service_rate_mu": 1/180,  # 人工服务率：每秒处理 1/180 个请求（平均180秒=3分钟/单）
    "num_servers_c": 3,        # 人工坐席数量（减少坐席，增加压力测试）
    "max_queue_length": 50,    # 最大队列长度（超过则强制LLM处理）
}

# 流量模式（模拟交易时间）
TRAFFIC_PATTERNS = {
    "normal": {"lambda": 2.0, "description": "平时流量，2 req/s"},
    "morning_peak": {"lambda": 10.0, "description": "早盘高峰 9:30-10:00，10 req/s"},
    "afternoon_peak": {"lambda": 6.0, "description": "午盘高峰 13:00-13:30，6 req/s"},
    "night": {"lambda": 0.5, "description": "夜间低谷，0.5 req/s"},
}

# LLM 推理延迟（毫秒）
LLM_LATENCY = {
    "slm": 16,      # 小模型（Qwen2.5-0.5B）
    "llm_32b": 95,  # 大模型（Qwen2.5-32B）
}

# SLA 约束（高价值请求最大等待时间）
SLA_CONSTRAINTS = {
    "high_risk_max_wait_ms": 30000,  # 高危请求30秒内必须响应
}
