"""
人机混合智能体调度仿真系统
Human-AI Hybrid Agent Scheduling Simulation

核心功能：
1. M/M/c 排队模型仿真
2. 四种调度策略对比（LLM-Only / Vanilla L2D / Static Cost / Dynamic Queue-Aware）
3. 流量洪峰仿真
4. 综合评估指标（ETC, CSA, CER, SLA等）

使用示例：
    uv run python -m exp.scheduling_simulation --test_data ./data/validation_set_1000_eval.jsonl
"""

from .queue_model import MMcQueue, Request, TrafficGenerator
from .routers import (
    BaseRouter,
    LLMOnlyRouter,
    VanillaL2DRouter,
    StaticCostRouter,
    DynamicQueueAwareRouter,
    RoutingDecision,
)
from .simulator import SchedulingSimulator, SimulationResult
from .config import COST_CONFIG, QUEUE_CONFIG, TRAFFIC_PATTERNS

__all__ = [
    'MMcQueue',
    'Request',
    'TrafficGenerator',
    'BaseRouter',
    'LLMOnlyRouter',
    'VanillaL2DRouter',
    'StaticCostRouter',
    'DynamicQueueAwareRouter',
    'RoutingDecision',
    'SchedulingSimulator',
    'SimulationResult',
    'COST_CONFIG',
    'QUEUE_CONFIG',
    'TRAFFIC_PATTERNS',
]
