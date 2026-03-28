"""
四种调度策略实现
Four Scheduling Strategies for Human-AI Routing
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .queue_model import MMcQueue, Request
from .config import COST_CONFIG, LLM_LATENCY


@dataclass
class RoutingDecision:
    """路由决策结果"""
    assign_to: str  # "llm" 或 "human"
    expected_cost_llm: float  # 给LLM的期望损失
    expected_cost_human: float  # 给人的期望损失
    reason: str  # 决策原因
    

class BaseRouter(ABC):
    """路由器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.decision_history: List[Dict] = []
    
    @abstractmethod
    def route(self, request: Request, queue: MMcQueue, current_time: float) -> RoutingDecision:
        """
        做出路由决策
        
        Args:
            request: 待路由的请求
            queue: 人类队列状态
            current_time: 当前时间（秒）
            
        Returns:
            RoutingDecision
        """
        pass
    
    def get_name(self) -> str:
        return self.name


class LLMOnlyRouter(BaseRouter):
    """
    Baseline 1: LLM-Only
    所有请求全部由大模型回答，完全没有人类介入
    """
    
    def __init__(self):
        super().__init__("LLM-Only")
    
    def route(self, request: Request, queue: MMcQueue, current_time: float) -> RoutingDecision:
        """
        永远分配给 LLM
        期望损失 = P(错) * C_error
        """
        expected_cost = request.llm_error_prob * request.true_cost
        
        decision = RoutingDecision(
            assign_to="llm",
            expected_cost_llm=expected_cost,
            expected_cost_human=float('inf'),  # 不使用人类
            reason="LLM-Only策略：永远分配给LLM"
        )
        
        self.decision_history.append({
            "time": current_time,
            "request_id": request.id,
            "decision": "llm",
            "expected_cost": expected_cost,
        })
        
        return decision


class VanillaL2DRouter(BaseRouter):
    """
    Baseline 2: Vanilla L2D (Learning to Defer)
    仅看模型信心。只要 P(LLM错) > 阈值，一律扔给人类
    """
    
    def __init__(self, threshold: float = 0.2):
        """
        Args:
            threshold: 置信度阈值，P(LLM错) > threshold 时转人工
        """
        super().__init__(f"Vanilla L2D (threshold={threshold})")
        self.threshold = threshold
    
    def route(self, request: Request, queue: MMcQueue, current_time: float) -> RoutingDecision:
        """
        基于固定阈值决策
        """
        # LLM期望损失
        cost_llm = request.llm_error_prob * request.true_cost
        
        # 人类期望损失（静态估计，不考虑排队）
        # 假设平均等待 30 秒
        expected_wait = 30.0
        delay_cost = expected_wait * 1000 * COST_CONFIG["latency_cost_per_ms"]  # 转换为毫秒
        cost_human = COST_CONFIG["human_fixed_cost"] + delay_cost
        
        if request.llm_error_prob > self.threshold:
            assign_to = "human"
            reason = f"P(LLM错)={request.llm_error_prob:.3f} > 阈值{self.threshold}，触发人工介入"
        else:
            assign_to = "llm"
            reason = f"P(LLM错)={request.llm_error_prob:.3f} ≤ 阈值{self.threshold}，LLM处理"
        
        decision = RoutingDecision(
            assign_to=assign_to,
            expected_cost_llm=cost_llm,
            expected_cost_human=cost_human,
            reason=reason
        )
        
        self.decision_history.append({
            "time": current_time,
            "request_id": request.id,
            "decision": assign_to,
            "p_error": request.llm_error_prob,
        })
        
        return decision


class StaticCostRouter(BaseRouter):
    """
    Baseline 3: Post-hoc Cost-Sensitive Router
    静态代价感知级联
    计算 R_LLM = P(LLM错) * C_error
    当 R_LLM > 固定人类成本 C_human 时触发交接
    （无视排队拥堵）
    """
    
    def __init__(self, human_cost_threshold: float = 100.0):
        """
        Args:
            human_cost_threshold: 人类介入成本阈值
        """
        super().__init__(f"Static Cost-Sensitive (C_human={human_cost_threshold})")
        self.human_cost_threshold = human_cost_threshold
    
    def route(self, request: Request, queue: MMcQueue, current_time: float) -> RoutingDecision:
        """
        基于静态代价比较决策
        """
        # LLM期望损失
        cost_llm = request.llm_error_prob * request.true_cost
        
        # 人类期望损失（静态估计）
        # 假设平均等待 30 秒
        expected_wait = 30.0
        delay_cost = expected_wait * 1000 * COST_CONFIG["latency_cost_per_ms"]
        cost_human = COST_CONFIG["human_fixed_cost"] + delay_cost
        
        if cost_llm > 50:
            print("阈值",self.human_cost_threshold)
            assign_to = "human"
            reason = f"R_LLM={cost_llm:.2f} > 阈值{self.human_cost_threshold}，转人工"
        else:
            assign_to = "llm"
            reason = f"R_LLM={cost_llm:.2f} ≤ 阈值{self.human_cost_threshold}，LLM处理"
        
        decision = RoutingDecision(
            assign_to=assign_to,
            expected_cost_llm=cost_llm,
            expected_cost_human=cost_human,
            reason=reason
        )
        
        self.decision_history.append({
            "time": current_time,
            "request_id": request.id,
            "decision": assign_to,
            "cost_llm": cost_llm,
        })
        
        return decision


class DynamicQueueAwareRouter(BaseRouter):
    """
    Ours: Dynamic Queue-Aware Routing
    动态排队感知算法
    
    核心不等式：
    P(LLM错) * C_error  vs  E[实时排队延迟] * C_delay + C_human_fixed
    
    当 LLM期望损失 > 人类期望损失 时，转人工
    
    支持两种延迟成本模式：
    - linear: 线性增长（默认，与原代码兼容）
    - saturating: 饱和递增函数 C_delay(t) = α(1 - e^(-βt))
    """
    
    def __init__(
        self,
        latency_cost_per_ms: float = None,
        human_fixed_cost: float = None,
        delay_cost_mode: str = "linear",
        saturating_alpha: float = 36.0,  # 饱和成本上限（约120秒*0.3元/秒）
        saturating_beta: float = 0.025   # 增长速率，使60秒时约60%饱和
    ):
        """
        Args:
            latency_cost_per_ms: 每毫秒延迟成本（仅linear模式使用）
            human_fixed_cost: 人工固定成本
            delay_cost_mode: "linear" 或 "saturating"
            saturating_alpha: 饱和函数最大成本（元）
            saturating_beta: 饱和函数增长速率（1/秒）
        """
        mode_suffix = " (Saturating)" if delay_cost_mode == "saturating" else ""
        super().__init__(f"Dynamic Queue-Aware{mode_suffix} (Ours)")
        
        self.latency_cost_per_ms = latency_cost_per_ms or COST_CONFIG["latency_cost_per_ms"]
        self.human_fixed_cost = human_fixed_cost or COST_CONFIG["human_fixed_cost"]
        self.delay_cost_mode = delay_cost_mode
        self.saturating_alpha = saturating_alpha
        self.saturating_beta = saturating_beta
        
        # 决策统计
        self.stats = {
            "llm_decisions": 0,
            "human_decisions": 0,
            "avg_queue_wait_when_deferred": [],
        }
    
    def _compute_delay_cost(self, wait_seconds: float) -> float:
        """
        计算延迟成本
        
        Args:
            wait_seconds: 预计等待时间（秒）
            
        Returns:
            延迟成本（元）
        """
        if self.delay_cost_mode == "saturating":
            # 饱和递增函数: C(t) = α(1 - e^(-βt))
            # 参数设计：α=36元（约120秒线性成本），β=0.025
            # 在典型等待时间（0-120秒）内与线性成本（0.3元/秒）相近
            return self.saturating_alpha * (1 - np.exp(-self.saturating_beta * wait_seconds))
        else:
            # 线性增长（默认）
            wait_ms = wait_seconds * 1000
            return wait_ms * self.latency_cost_per_ms
    
    def route(self, request: Request, queue: MMcQueue, current_time: float) -> RoutingDecision:
        """
        基于实时队列状态的动态决策
        """
        # 1. 计算 LLM 期望损失
        cost_llm_error = request.llm_error_prob * request.true_cost
        cost_llm_latency = LLM_LATENCY["slm"] * self.latency_cost_per_ms  # SLM延迟代价
        cost_llm_total = cost_llm_error + cost_llm_latency
        
        # 2. 计算人类期望损失（实时）
        # 获取当前队列状态
        queue_state = queue.get_queue_state()
        expected_wait_seconds = queue_state["expected_wait"]
        
        # 延迟代价（根据模式选择计算方式）
        delay_cost = self._compute_delay_cost(expected_wait_seconds)
        # 总人类成本
        cost_human_total = self.human_fixed_cost + delay_cost
        
        # 3. 决策
        if cost_llm_total > cost_human_total:
            assign_to = "human"
            reason = (
                f"动态决策：LLM期望损失({cost_llm_total:.2f}) > "
                f"人类期望损失({cost_human_total:.2f}, 等待{expected_wait_seconds:.1f}s)"
            )
            self.stats["human_decisions"] += 1
            self.stats["avg_queue_wait_when_deferred"].append(expected_wait_seconds)
        else:
            assign_to = "llm"
            reason = (
                f"动态决策：LLM期望损失({cost_llm_total:.2f}) ≤ "
                f"人类期望损失({cost_human_total:.2f}, 等待{expected_wait_seconds:.1f}s)"
            )
            self.stats["llm_decisions"] += 1
        
        decision = RoutingDecision(
            assign_to=assign_to,
            expected_cost_llm=cost_llm_total,
            expected_cost_human=cost_human_total,
            reason=reason
        )
        
        self.decision_history.append({
            "time": current_time,
            "request_id": request.id,
            "decision": assign_to,
            "cost_llm": cost_llm_total,
            "cost_human": cost_human_total,
            "queue_wait": expected_wait_seconds,
            "queue_length": queue_state["queue_length"],
        })
        
        return decision
    
    def get_stats(self) -> Dict:
        """获取决策统计"""
        total = self.stats["llm_decisions"] + self.stats["human_decisions"]
        avg_wait = np.mean(self.stats["avg_queue_wait_when_deferred"]) if self.stats["avg_queue_wait_when_deferred"] else 0
        
        return {
            "total_decisions": total,
            "llm_ratio": self.stats["llm_decisions"] / total if total > 0 else 0,
            "human_ratio": self.stats["human_decisions"] / total if total > 0 else 0,
            "avg_queue_wait_when_deferred": avg_wait,
        }
