"""
基于非抢占式优先队列的Dynamic路由器

特点：
- High风险任务优先处理（排在High队列前面，Mid/Low前面）
- 非抢占式：不会中断正在处理的任务
- 同优先级FIFO：先来的High优先于后来的High
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .routers import BaseRouter, RoutingDecision
from .priority_queue import NonPreemptivePriorityQueue
from .queue_model import Request
from .config import COST_CONFIG, LLM_LATENCY


class DynamicPriorityQueueRouter(BaseRouter):
    """
    Ours+: 基于非抢占式优先队列的动态调度
    
    核心不等式（按优先级分别计算）：
    P(LLM错) × C_error  vs  E[该优先级等待时间] × C_delay + C_human_fixed
    
    优先级队列结构：
    High队列: H, H, H... (新来的H排在现有H后面)
    Mid队列: M, M, M...
    Low队列: L, L, L...
    
    调度规则：
    1. 优先级: High > Mid > Low
    2. 非抢占: 正在处理的任务不中断
    3. 服务器空闲时按优先级取任务
    """
    
    def __init__(
        self,
        latency_cost_per_ms: float = None,
        human_fixed_cost: float = None
    ):
        super().__init__("Dynamic Priority Queue (Ours+)")
        
        self.latency_cost_per_ms = latency_cost_per_ms or COST_CONFIG["latency_cost_per_ms"]
        self.human_fixed_cost = human_fixed_cost or COST_CONFIG["human_fixed_cost"]
        
        # 决策统计
        self.stats = {
            "llm_decisions": 0,
            "human_decisions": 0,
            "by_priority": {"high": {"llm": 0, "human": 0},
                          "mid": {"llm": 0, "human": 0},
                          "low": {"llm": 0, "human": 0}},
        }
    
    def route(self, request: Request, queue: NonPreemptivePriorityQueue, current_time: float) -> RoutingDecision:
        """
        基于优先级队列的动态决策
        
        关键：High风险使用High队列的预期等待时间
        """
        # 1. 计算 LLM 期望损失
        cost_llm_error = request.llm_error_prob * request.true_cost
        cost_llm_latency = LLM_LATENCY["slm"] * self.latency_cost_per_ms
        cost_llm_total = cost_llm_error + cost_llm_latency
        
        # 2. 计算人类期望损失（按优先级分别计算）
        # 获取该风险等级对应的预期等待时间
        risk_level = request.true_risk_level
        expected_wait_seconds = queue.get_expected_waiting_time(risk_level)
        expected_wait_ms = expected_wait_seconds * 1000
        
        # 延迟代价
        delay_cost = expected_wait_ms * self.latency_cost_per_ms
        # 总人类成本
        cost_human_total = self.human_fixed_cost + delay_cost
        
        # 3. 决策（纯成本决策，无强制）
        if cost_llm_total > cost_human_total:
            assign_to = "human"
            reason = (
                f"优先队列决策[{risk_level.upper()}]: LLM期望损失({cost_llm_total:.2f}) > "
                f"人类期望损失({cost_human_total:.2f}, 等待{expected_wait_seconds:.1f}s)"
            )
            self.stats["human_decisions"] += 1
            self.stats["by_priority"][risk_level]["human"] += 1
        else:
            assign_to = "llm"
            reason = (
                f"优先队列决策[{risk_level.upper()}]: LLM期望损失({cost_llm_total:.2f}) ≤ "
                f"人类期望损失({cost_human_total:.2f}, 等待{expected_wait_seconds:.1f}s)"
            )
            self.stats["llm_decisions"] += 1
            self.stats["by_priority"][risk_level]["llm"] += 1
        
        decision = RoutingDecision(
            assign_to=assign_to,
            expected_cost_llm=cost_llm_total,
            expected_cost_human=cost_human_total,
            reason=reason
        )
        
        self.decision_history.append({
            "time": current_time,
            "request_id": request.id,
            "risk_level": risk_level,
            "decision": assign_to,
            "cost_llm": cost_llm_total,
            "cost_human": cost_human_total,
            "queue_wait": expected_wait_seconds,
        })
        
        return decision
    
    def get_stats(self) -> Dict:
        """获取决策统计"""
        total = self.stats["llm_decisions"] + self.stats["human_decisions"]
        
        return {
            "total_decisions": total,
            "llm_ratio": self.stats["llm_decisions"] / total if total > 0 else 0,
            "human_ratio": self.stats["human_decisions"] / total if total > 0 else 0,
            "by_priority": self.stats["by_priority"],
        }
