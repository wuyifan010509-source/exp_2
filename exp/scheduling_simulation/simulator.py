"""
仿真引擎 - 完整的调度仿真循环
Simulation Engine for Human-AI Hybrid Scheduling
"""

import numpy as np
import json
import sys
import os
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import time

from .queue_model import MMcQueue, Request, TrafficGenerator
from .routers import BaseRouter, RoutingDecision
from .config import COST_CONFIG, LLM_LATENCY, QUEUE_CONFIG

# 尝试导入SLM模型（如果可用）
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from exp.slm_distillation.inference import SLMCostPredictor
    from exp.slm_distillation.config import OUTPUT_DIR as SLM_OUTPUT_DIR
    SLM_AVAILABLE = True
except ImportError:
    SLM_AVAILABLE = False
    print("[Warning] SLM模型未找到，将使用启发式估计")


@dataclass
class SimulationResult:
    """仿真结果"""
    router_name: str
    total_requests: int
    
    # 成本指标
    total_error_cost: float
    total_latency_cost: float
    total_human_cost: float
    expected_total_cost: float
    
    # 风险指标
    catastrophic_error_count: int
    catastrophic_error_rate: float
    
    # 队列指标
    avg_queueing_delay: float
    max_queueing_delay: float
    sla_violation_rate: float
    server_utilization: float
    
    # 调度行为
    llm_count: int
    human_count: int
    deferral_rate: float
    
    # 按风险等级的细分
    risk_distribution: Dict = field(default_factory=dict)
    
    # 原始数据（用于后续分析）
    completed_requests: List[Request] = field(default_factory=list)


class SchedulingSimulator:
    """
    调度仿真器
    
    模拟流程：
    1. 生成带时间戳的请求流（泊松到达）
    2. 对每个请求调用 Router 做决策
    3. LLM请求立即完成，人类请求进入队列
    4. 推进时间，处理队列完成
    5. 统计结果
    """
    
    def __init__(
        self,
        router: BaseRouter,
        num_servers: int = QUEUE_CONFIG["num_servers_c"],
        service_rate: float = QUEUE_CONFIG["service_rate_mu"],
        max_time: float = 3600.0,  # 默认仿真1小时
    ):
        """
        Args:
            router: 路由策略
            num_servers: 人工坐席数
            service_rate: 人工服务率
            max_time: 最大仿真时间（秒）
        """
        self.router = router
        self.queue = MMcQueue(num_servers, service_rate)
        self.max_time = max_time
        
        # 结果收集
        self.completed_requests: List[Request] = []
        self.llm_requests: List[Request] = []
        self.failed_requests: List[Request] = []
        
        # SLM预测器（按需初始化）
        self.slm_predictor = None
        
    def _init_slm_predictor(self):
        """初始化SLM预测器（延迟加载）"""
        if self.slm_predictor is None and SLM_AVAILABLE:
            try:
                print("[SLM] 初始化预测器...")
                self.slm_predictor = SLMCostPredictor(
                    adapter_path=SLM_OUTPUT_DIR
                )
                print("[SLM] 预测器初始化完成")
            except Exception as e:
                print(f"[SLM] 初始化失败: {e}")
                self.slm_predictor = None
    
    def _get_slm_error_prob(self, query: str, true_level: str) -> float:
        """
        使用SLM模型计算P(LLM错)
        
        策略：
        1. SLM预测风险等级
        2. 如果预测等级 < 真实等级 → 高错误概率（漏报）
        3. 如果预测等级 = 真实等级 → 低错误概率
        4. 如果预测等级 > 真实等级 → 中等错误概率（误报但不危险）
        """
        if self.slm_predictor is None:
            return None
        
        try:
            result = self.slm_predictor.predict(query)
            pred_class = result.get("predicted_class", 1)  # 默认mid
            probs = result.get("probabilities", {})
            
            # 类别映射
            level_to_class = {"low": 0, "mid": 1, "high": 2}
            true_class = level_to_class.get(true_level, 1)
            
            # 计算P(错)：1 - P(真实类别的概率)
            class_names = ["low", "mid", "high"]
            true_prob = probs.get(class_names[true_class], 0.33)
            error_prob = 1.0 - true_prob
            
            # 调整：如果是High风险且预测不是High，增加错误概率
            if true_level == "high" and pred_class < 2:
                error_prob = max(error_prob, 0.5)  # 至少50%概率答错
            
            return min(error_prob, 0.95)  # 上限95%
            
        except Exception as e:
            print(f"[SLM] 预测失败: {e}")
            return None
        
    def simulate(
        self,
        test_data: List[Dict],
        llm_error_func: Optional[Callable[[str, str], float]] = None,
        verbose: bool = True
    ) -> SimulationResult:
        """
        运行仿真
        
        Args:
            test_data: 测试数据集
                [{"query": str, "true_level": str, "true_cost": float}]
            llm_error_func: 计算 P(LLM错) 的函数
                func(query: str, true_level: str) -> float
            verbose: 是否打印进度
            
        Returns:
            SimulationResult
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"开始仿真: {self.router.get_name()}")
            print(f"{'='*70}")
        
        # 初始化
        current_time = 0.0
        next_arrival_idx = 0
        
        # 为每个请求生成到达时间（泊松过程）
        arrival_times = self._generate_arrival_times(len(test_data))
        
        # 事件驱动仿真主循环
        while current_time < self.max_time and next_arrival_idx < len(test_data):
            # 获取下一个事件时间
            next_arrival_time = arrival_times[next_arrival_idx] if next_arrival_idx < len(test_data) else float('inf')
            next_completion_time = self._get_next_completion_time()
            
            # 推进到下一个事件
            if next_arrival_time <= next_completion_time:
                # 处理到达事件
                current_time = next_arrival_time
                self._handle_arrival(
                    test_data[next_arrival_idx],
                    current_time,
                    next_arrival_idx,
                    llm_error_func
                )
                next_arrival_idx += 1
            else:
                # 处理完成事件
                current_time = next_completion_time
                self._handle_completion(current_time)
            
            if verbose and next_arrival_idx % 100 == 0:
                print(f"  进度: {next_arrival_idx}/{len(test_data)} "
                      f"(t={current_time:.1f}s, queue={len(self.queue.queue)})")
        
        # 处理剩余队列中的请求
        if verbose:
            print(f"\n处理剩余队列...")
        while self.queue.queue or any(s.busy for s in self.queue.servers):
            next_completion_time = self._get_next_completion_time()
            if next_completion_time == float('inf'):
                break
            current_time = next_completion_time
            self._handle_completion(current_time)
        
        # 计算结果
        result = self._compute_results()
        
        if verbose:
            self._print_results(result)
        
        return result
    
    def _generate_arrival_times(self, n_requests: int) -> List[float]:
        """
        生成泊松到达时间序列
        模拟流量洪峰：早盘(9:30)高峰，午盘(13:00)次高峰
        """
        times = []
        current_time = 0.0
        
        # 定义流量模式（4小时仿真周期）
        # 0-30min: 早盘高峰，30-120min: 平时，120-150min: 午盘高峰，150-240min: 平时
        phases = [
            (1800, 10.0),   # 0-30min: 10 req/s (早盘高峰)
            (5400, 2.0),    # 30-120min: 2 req/s (平时)
            (1800, 6.0),    # 120-150min: 6 req/s (午盘高峰)
            (5400, 2.0),    # 150-240min: 2 req/s (平时)
        ]
        
        phase_idx = 0
        phase_elapsed = 0.0
        
        for i in range(n_requests):
            # 获取当前相位的到达率
            phase_duration, lambda_rate = phases[phase_idx % len(phases)]
            
            # 指数分布生成到达间隔
            inter_arrival = np.random.exponential(1.0 / lambda_rate)
            current_time += inter_arrival
            phase_elapsed += inter_arrival
            
            # 检查是否需要切换相位
            while phase_elapsed >= phase_duration:
                phase_elapsed -= phase_duration
                phase_idx += 1
            
            times.append(current_time)
        
        return times
    
    def _get_next_completion_time(self) -> float:
        """获取下一个服务完成时间"""
        min_time = float('inf')
        for server in self.queue.servers:
            if server.busy and server.service_completion_time < min_time:
                min_time = server.service_completion_time
        return min_time
    
    def _handle_arrival(
        self,
        data: Dict,
        current_time: float,
        request_id: int,
        llm_error_func: Optional[Callable]
    ):
        """处理请求到达"""
        # 创建请求对象
        request = Request(
            id=request_id,
            arrival_time=current_time,
            query=data["query"],
            true_risk_level=data["true_level"],
            true_cost=data["true_cost"],
        )
        
        # 计算 LLM 错误概率
        # 优先使用传入的函数，其次尝试SLM模型，最后回退到启发式估计
        if llm_error_func:
            request.llm_error_prob = llm_error_func(data["query"], data["true_level"])
        else:
            # 尝试使用SLM模型
            if self.slm_predictor is None:
                self._init_slm_predictor()
            
            slm_prob = self._get_slm_error_prob(data["query"], data["true_level"])
            
            if slm_prob is not None:
                request.llm_error_prob = slm_prob
                if request_id % 50 == 0:  # 每50条打印一次
                    print(f"  [SLM] Query {request_id}: P(error)={slm_prob:.3f}")
            else:
                # SLM失败，回退到启发式估计
                if data["true_level"] == "high":
                    request.llm_error_prob = 0.35
                elif data["true_level"] == "mid":
                    request.llm_error_prob = 0.15
                else:
                    request.llm_error_prob = 0.05
        
        # 调用 Router 做决策
        decision = self.router.route(request, self.queue, current_time)
        request.assigned_to = decision.assign_to
        
        if decision.assign_to == "llm":
            # LLM 立即处理完成
            request.completion_time = current_time + LLM_LATENCY["slm"] / 1000.0
            request.actual_cost = self._compute_llm_actual_cost(request)
            self.llm_requests.append(request)
            self.completed_requests.append(request)
        else:
            # 加入人类队列
            success = self.queue.enqueue(request, current_time)
            if not success:
                # 队列满，强制用 LLM
                request.assigned_to = "llm_forced"
                request.completion_time = current_time + LLM_LATENCY["slm"] / 1000.0
                request.actual_cost = self._compute_llm_actual_cost(request) * 2  # 惩罚
                self.failed_requests.append(request)
                self.completed_requests.append(request)
    
    def _handle_completion(self, current_time: float):
        """处理服务完成"""
        completed = self.queue.process_completions(current_time)
        for request in completed:
            # 人类服务的实际成本（没有错误，只有延迟和固定成本）
            request.actual_cost = COST_CONFIG["human_fixed_cost"]
            self.completed_requests.append(request)
    
    def _compute_llm_actual_cost(self, request: Request) -> float:
        """
        计算 LLM 处理的实际成本
        实际有概率答错
        """
        # 随机决定是否答错
        is_error = np.random.random() < request.llm_error_prob
        
        if is_error:
            # 答错了，产生错误代价
            return request.true_cost
        else:
            # 答对了，只有延迟代价
            return LLM_LATENCY["slm"] * COST_CONFIG["latency_cost_per_ms"]
    
    def _compute_results(self) -> SimulationResult:
        """计算仿真结果"""
        # 分类统计
        llm_requests = [r for r in self.completed_requests if r.assigned_to == "llm"]
        human_requests = [r for r in self.completed_requests if r.assigned_to == "human"]
        
        # 成本统计
        total_error_cost = 0.0
        total_latency_cost = 0.0
        total_human_cost = 0.0
        catastrophic_errors = 0
        
        # 错误代价阈值（用于判断是否为错误）
        error_threshold = 5  # 代价 > 5元认为是错误（排除low风险）
        
        for r in self.completed_requests:
            if r.assigned_to == "llm":
                # LLM 处理：错误代价 + 延迟代价
                # 检查是否为错误（实际代价远大于延迟代价）
                is_error = r.actual_cost > error_threshold
                error_cost = r.actual_cost if is_error else 0
                latency_cost = LLM_LATENCY["slm"] * COST_CONFIG["latency_cost_per_ms"]
                total_error_cost += error_cost
                total_latency_cost += latency_cost
                
                # 灾难性错误：高风险问题答错（错误代价 >= high风险的一半）
                high_cost_threshold = COST_CONFIG["error_cost"]["high"] / 2
                if r.true_risk_level == "high" and error_cost >= high_cost_threshold:
                    catastrophic_errors += 1
            else:
                # 人类处理：固定成本 + 延迟代价
                total_human_cost += COST_CONFIG["human_fixed_cost"]
                wait_cost = r.waiting_time * 1000 * COST_CONFIG["latency_cost_per_ms"]
                total_latency_cost += wait_cost
        
        # 队列指标
        human_delays = [r.waiting_time * 1000 for r in human_requests]  # 转换为毫秒
        avg_delay = np.mean(human_delays) if human_delays else 0.0
        max_delay = np.max(human_delays) if human_delays else 0.0
        
        # SLA 违约：高危请求等待超过30秒
        high_risk_human = [r for r in human_requests if r.true_risk_level == "high"]
        sla_violations = sum(1 for r in high_risk_human if r.waiting_time > 30)
        sla_violation_rate = sla_violations / len(high_risk_human) if high_risk_human else 0.0
        
        # 按风险等级分布
        risk_dist = defaultdict(lambda: {"llm": 0, "human": 0})
        for r in self.completed_requests:
            risk_dist[r.true_risk_level][r.assigned_to] += 1
        
        return SimulationResult(
            router_name=self.router.get_name(),
            total_requests=len(self.completed_requests),
            total_error_cost=total_error_cost,
            total_latency_cost=total_latency_cost,
            total_human_cost=total_human_cost,
            expected_total_cost=total_error_cost + total_latency_cost + total_human_cost,
            catastrophic_error_count=catastrophic_errors,
            catastrophic_error_rate=catastrophic_errors / len(self.completed_requests) if self.completed_requests else 0.0,
            avg_queueing_delay=avg_delay,
            max_queueing_delay=max_delay,
            sla_violation_rate=sla_violation_rate,
            server_utilization=self.queue.get_server_utilization(),
            llm_count=len(llm_requests),
            human_count=len(human_requests),
            deferral_rate=len(human_requests) / len(self.completed_requests) if self.completed_requests else 0.0,
            risk_distribution=dict(risk_dist),
            completed_requests=self.completed_requests,
        )
    
    def _print_results(self, result: SimulationResult):
        """打印结果"""
        print(f"\n{'='*70}")
        print(f"仿真结果: {result.router_name}")
        print(f"{'='*70}")
        
        print(f"\n[成本指标]")
        print(f"  总错误代价: {result.total_error_cost:.2f} 元")
        print(f"  总延迟代价: {result.total_latency_cost:.2f} 元")
        print(f"  总人力成本: {result.total_human_cost:.2f} 元")
        print(f"  ★ 总期望成本(ETC): {result.expected_total_cost:.2f} 元")
        
        print(f"\n[风险指标]")
        print(f"  灾难性错误数: {result.catastrophic_error_count}")
        print(f"  灾难性错误率(CER): {result.catastrophic_error_rate:.4f}")
        
        print(f"\n[队列指标]")
        print(f"  平均排队延迟: {result.avg_queueing_delay:.1f} ms")
        print(f"  最大排队延迟: {result.max_queueing_delay:.1f} ms")
        print(f"  SLA违约率: {result.sla_violation_rate:.4f}")
        print(f"  服务器利用率: {result.server_utilization:.4f}")
        
        print(f"\n[调度行为]")
        print(f"  LLM处理: {result.llm_count} ({result.llm_count/result.total_requests*100:.1f}%)")
        print(f"  人工处理: {result.human_count} ({result.human_count/result.total_requests*100:.1f}%)")
        print(f"  转交率: {result.deferral_rate:.4f}")
        
        print(f"\n[按风险等级分布]")
        for level, counts in result.risk_distribution.items():
            total = counts["llm"] + counts["human"]
            if total > 0:
                human_ratio = counts["human"] / total * 100
                print(f"  {level.upper()}: LLM={counts['llm']}, Human={counts['human']} "
                      f"(人工占比{human_ratio:.1f}%)")
