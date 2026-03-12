"""
仿真引擎 - 完整的调度仿真循环
Simulation Engine for Human-AI Hybrid Scheduling
"""

import numpy as np
import json
import sys
import os
from typing import List, Dict, Optional, Callable, Tuple
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
        queue_type: str = "default",  # "default" 或 "priority"
        custom_traffic_phases: Optional[List[Tuple[float, float]]] = None,  # 自定义流量配置
    ):
        """
        Args:
            router: 路由策略
            num_servers: 人工坐席数
            service_rate: 人工服务率
            max_time: 最大仿真时间（秒）
            queue_type: 队列类型 ("default"=普通队列, "priority"=优先级队列)
            custom_traffic_phases: 自定义流量阶段 [(持续时间秒, 到达率req/s), ...]
        """
        self.router = router
        self.queue_type = queue_type
        self.custom_traffic_phases = custom_traffic_phases
        
        # 根据类型创建队列
        if queue_type == "priority":
            from .priority_queue import NonPreemptivePriorityQueue
            self.queue = NonPreemptivePriorityQueue(num_servers, service_rate, QUEUE_CONFIG["max_queue_length"])
        else:
            self.queue = MMcQueue(num_servers, service_rate, QUEUE_CONFIG["max_queue_length"])
        
        self.max_time = max_time
        
        # 日志文件句柄
        self.log_fp = None
        
        # 结果收集
        self.completed_requests: List[Request] = []
        self.llm_requests: List[Request] = []
        self.failed_requests: List[Request] = []
        
        # SLM预测器（按需初始化）
        self.slm_predictor = None
        
        # 详细日志累加器
        self._log_accumulators = {
            'cumulative_error': 0.0,
            'cumulative_latency': 0.0,
            'cumulative_human': 0.0,
            'llm_count': {'high': 0, 'mid': 0, 'low': 0},
            'human_count': {'high': 0, 'mid': 0, 'low': 0},
        }
        
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
        verbose: bool = True,
        log_file: Optional[str] = None
    ) -> SimulationResult:
        """
        运行仿真
        
        Args:
            test_data: 测试数据集
                [{"query": str, "true_level": str, "true_cost": float}]
            llm_error_func: 计算 P(LLM错) 的函数
                func(query: str, true_level: str) -> float
            verbose: 是否打印进度
            log_file: 日志文件路径（如果指定，将详细日志写入文件）
            
        Returns:
            SimulationResult
        """
        # 设置日志文件
        self.log_fp = None
        if log_file:
            import os
            from datetime import datetime
            os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
            self.log_fp = open(log_file, 'w', encoding='utf-8')
            self.log_fp.write("="*100 + "\n")
            self.log_fp.write(f"完整仿真日志 - {self.router.get_name()}\n")
            self.log_fp.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_fp.write("="*100 + "\n")
            
            self.log_fp.write(f"\n【系统配置】\n")
            self.log_fp.write(f"  专家数量: {QUEUE_CONFIG['num_servers_c']}人\n")
            self.log_fp.write(f"  服务时间: {1/QUEUE_CONFIG['service_rate_mu']:.0f}秒/单\n")
            self.log_fp.write(f"  延迟成本系数: {COST_CONFIG['latency_cost_per_ms']}元/ms\n")
            self.log_fp.write(f"  人工固定成本: {COST_CONFIG['human_fixed_cost']}元\n")
            self.log_fp.write(f"  SLM延迟: {LLM_LATENCY['slm']}ms\n")
            
            self.log_fp.write(f"\n【代价权重配置】\n")
            self.log_fp.write(f"  High错误代价: {COST_CONFIG['error_cost']['high']}元 (P(error)=0.8)\n")
            self.log_fp.write(f"  Mid错误代价: {COST_CONFIG['error_cost']['mid']}元 (P(error)=0.4)\n")
            self.log_fp.write(f"  Low错误代价: {COST_CONFIG['error_cost']['low']}元 (P(error)=0.1)\n")
            
            self.log_fp.write("\n" + "="*100 + "\n")
            self.log_fp.write("开始处理请求...\n")
            self.log_fp.write("="*100 + "\n")
            self.log_fp.flush()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"开始仿真: {self.router.get_name()}")
            print(f"{'='*70}")
            
            # 打印详细的系统配置
            print("\n【系统配置】")
            print(f"  坐席人数: {QUEUE_CONFIG['num_servers_c']}人")
            print(f"  服务效率: {1/QUEUE_CONFIG['service_rate_mu']:.0f}秒/单 ({60/(1/QUEUE_CONFIG['service_rate_mu']):.0f}单/小时)")
            print(f"  延迟惩罚: {COST_CONFIG['latency_cost_per_ms']}元/ms ({COST_CONFIG['latency_cost_per_ms']*1000:.2f}元/秒)")
            print(f"  人工固定成本: {COST_CONFIG['human_fixed_cost']}元")
            
            # 打印到达洪峰配置
            print("\n【到达洪峰配置】")
            if self.custom_traffic_phases:
                phases = self.custom_traffic_phases
            else:
                phases = [
                    (1800, 30.0/60), (5400, 6.0/60), (1800, 18.0/60), (5400, 6.0/60)
                ]
            phase_names = ["早盘高峰(0-30min)", "平时(30-120min)", "午盘高峰(120-150min)", "平时(150-240min)"]
            for i, (duration, rate) in enumerate(phases):
                print(f"  {phase_names[i]}: {rate*60:.1f} req/min")
            
            # 打印代价权重
            print("\n【代价权重配置】")
            print(f"  High错误代价: {COST_CONFIG['error_cost']['high']}元")
            print(f"  Mid错误代价: {COST_CONFIG['error_cost']['mid']}元")
            print(f"  Low错误代价: {COST_CONFIG['error_cost']['low']}元")
            
            # 打印策略阈值
            print("\n【策略阈值配置】")
            print(f"  Vanilla L2D阈值: 0.2")
            print(f"  Static Cost阈值: 100.0元")
            print(f"  Dynamic: 实时队列感知")
        
        # 初始化
        current_time = 0.0
        next_arrival_idx = 0
        
        # 为每个请求生成到达时间（泊松过程）
        arrival_times = self._generate_arrival_times(len(test_data))
        
        # 事件驱动仿真主循环
        # 漏洞修复：由数据长度决定仿真结束，而非max_time
        while next_arrival_idx < len(test_data):
            # 获取下一个事件时间
            next_arrival_time = arrival_times[next_arrival_idx] if next_arrival_idx < len(test_data) else float('inf')
            next_completion_time = self._get_next_completion_time()
            
            # 推进到下一个事件
            # 漏洞修复：使用 < 而非 <=，让完成事件优先（释放资源）
            if next_arrival_time < next_completion_time:
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
                # 处理完成事件（优先处理，释放资源）
                current_time = next_completion_time
                self._handle_completion(current_time)
            
            if verbose and next_arrival_idx % 100 == 0:
                print(f"  进度: {next_arrival_idx}/{len(test_data)} "
                      f"(t={current_time:.1f}s, queue={len(self.queue.queue)})")
        
        # 处理剩余队列中的请求
        if verbose:
            print(f"\n处理剩余队列...")
        # 漏洞修复：使用is_empty()方法确保兼容优先级队列
        while (not self.queue.is_empty()) or any(s.busy for s in self.queue.servers):
            next_completion_time = self._get_next_completion_time()
            
            # 漏洞修复：添加死锁检测
            if next_completion_time == float('inf'):
                if not self.queue.is_empty():
                    raise RuntimeError(
                        "仿真死锁: 队列中有请求等待，但所有专家都处于空闲状态！"
                        "请检查 Queue 类的 process_completions() 方法是否正确让空闲专家拉取了下一个队列任务。"
                    )
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
        
        # 使用自定义流量配置（如果提供），否则使用默认配置
        if self.custom_traffic_phases is not None:
            phases = self.custom_traffic_phases
        else:
            # 定义流量模式（4小时仿真周期）
            # 注意：lambda 单位是 req/min，需要转换为 req/s 用于指数分布
            # 0-30min: 早盘高峰，30-120min: 平时，120-150min: 午盘高峰，150-240min: 平时
            phases = [
                (18000, 10.0/60),   # 0-30min: 10 req/min = 0.167 req/s (早盘高峰)
                (5400, 6.0/60),    # 30-120min: 2 req/min = 0.033 req/s (平时)
                (1800, 18.0/60),    # 120-150min: 6 req/min = 0.1 req/s (午盘高峰)
                (5400, 6.0/60),    # 150-240min: 2 req/min = 0.033 req/s (平时)
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
        
        # 获取队列状态（用于日志记录和DEBUG）
        queue_state = self.queue.get_queue_state()
        
        # 计算 LLM 错误概率
        # 优先级：1)数据中的固定概率 > 2)传入的函数 > 3)SLM模型 > 4)启发式估计
        if "llm_error_prob" in data:
            # 使用数据集中预生成的固定错误概率
            request.llm_error_prob = data["llm_error_prob"]
        elif llm_error_func:
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
                # 修正后：高风险错误率0.8，中风险错误率0.4，低风险错误率0.1
                if data["true_level"] == "high":
                    request.llm_error_prob = 0.80  # 80%错误率
                elif data["true_level"] == "mid":
                    request.llm_error_prob = 0.40  # 40%错误率
                else:
                    request.llm_error_prob = 0.10  # 10%错误率
        
        # DEBUG: 打印所有请求的队列状态和延迟惩罚
        if request_id % 50 == 0 or data["true_level"] == "high":
            print(f"\n  [DEBUG] 请求 #{request_id} [{data['true_level'].upper()}]: {data['query'][:25]}...")
            print(f"    P(error)={request.llm_error_prob:.3f}, Cost={data['true_cost']}元")
            
            # 打印队列状态（High/Mid/Low/人工）
            if hasattr(self.queue, 'high_queue'):
                # 优先级队列
                high_len = len(self.queue.high_queue)
                mid_len = len(self.queue.mid_queue)
                low_len = len(self.queue.low_queue)
                busy_count = sum(1 for s in self.queue.servers if s.busy)
                print(f"    队列状态: High={high_len}, Mid={mid_len}, Low={low_len} | 坐席忙碌={busy_count}/{len(self.queue.servers)}")
            else:
                # 普通队列
                queue_len = len(self.queue.queue)
                busy_count = sum(1 for s in self.queue.servers if s.busy)
                print(f"    队列状态: 长度={queue_len} | 坐席忙碌={busy_count}/{len(self.queue.servers)}")
            
            # 打印预期等待和延迟惩罚
            expected_wait = queue_state.get('expected_wait', 0)
            delay_penalty = expected_wait * 1000 * COST_CONFIG['latency_cost_per_ms']
            print(f"    预期等待: {expected_wait:.1f}s | 延迟惩罚: {delay_penalty:.2f}元")
        
        # 调用 Router 做决策
        decision = self.router.route(request, self.queue, current_time)
        request.assigned_to = decision.assign_to
        
        # DEBUG: 打印决策结果
        if request_id % 50 == 0 or data["true_level"] == "high":
            print(f"    决策: {decision.assign_to.upper()}")
            print(f"    LLM期望损失: {decision.expected_cost_llm:.2f}元")
            print(f"    人工期望损失: {decision.expected_cost_human:.2f}元")
            if decision.expected_cost_llm > decision.expected_cost_human:
                print(f"    理论: 应该转人工")
            else:
                print(f"    理论: 应该给LLM")
        
        # 获取队列状态（用于日志记录）
        queue_state = self.queue.get_queue_state()
        
        # 记录详细日志（调用日志方法）
        if self.log_fp:
            llm_error_cost = request.llm_error_prob * data['true_cost']
            llm_latency_cost = LLM_LATENCY['slm'] * COST_CONFIG['latency_cost_per_ms']
            llm_total = llm_error_cost + llm_latency_cost
            
            wait_seconds = queue_state['expected_wait']
            wait_ms = wait_seconds * 1000
            delay_cost = wait_ms * COST_CONFIG['latency_cost_per_ms']
            human_total = COST_CONFIG['human_fixed_cost'] + delay_cost
            
            if decision.assign_to == "human":
                actual_cost_breakdown = {
                    'human': COST_CONFIG['human_fixed_cost'],
                    'wait': delay_cost,
                    'error': 0,
                    'latency': 0,
                    'total': COST_CONFIG['human_fixed_cost'] + delay_cost
                }
                self._log_accumulators['cumulative_human'] += COST_CONFIG['human_fixed_cost']
                self._log_accumulators['cumulative_latency'] += delay_cost
            else:
                actual_cost_breakdown = {
                    'human': 0,
                    'wait': 0,
                    'error': llm_error_cost,
                    'latency': llm_latency_cost,
                    'total': llm_total
                }
                self._log_accumulators['cumulative_error'] += llm_error_cost
                self._log_accumulators['cumulative_latency'] += llm_latency_cost
            
            self._log_request_detail(
                request_id, data, decision, queue_state,
                request.llm_error_prob, llm_total, human_total,
                actual_cost_breakdown, current_time
            )
        
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
    
    def _log_request_detail(self, request_id: int, data: Dict, decision, queue_state: Dict, 
                           p_error: float, llm_total: float, human_total: float,
                           actual_cost_breakdown: Dict, current_time: float):
        """记录单个请求的详细日志"""
        if not self.log_fp:
            return
            
        level = data['true_level']
        cost = data['true_cost']
        query = data.get('query', '')[:50] + '...' if len(data.get('query', '')) > 50 else data.get('query', '')
        
        # 累加器状态
        accum = self._log_accumulators
        
        self.log_fp.write(f"\n{'─'*100}\n")
        self.log_fp.write(f"【请求 #{request_id:03d}】 {query}\n")
        self.log_fp.write(f"风险等级: {level.upper()} | 错误代价: {cost}元\n")
        self.log_fp.write(f"{'─'*100}\n")
        
        # 步骤1：队列状态
        self.log_fp.write(f"  【步骤1】队列状态:\n")
        self.log_fp.write(f"    - 队列长度: {queue_state['queue_length']}条\n")
        self.log_fp.write(f"    - 忙碌专家: {queue_state['busy_servers']}/{QUEUE_CONFIG['num_servers_c']}\n")
        self.log_fp.write(f"    - 预期等待: {queue_state['expected_wait']:.1f}秒\n")
        
        # 步骤2：LLM期望损失
        llm_error_cost = p_error * cost
        llm_latency_cost = LLM_LATENCY['slm'] * COST_CONFIG['latency_cost_per_ms']
        self.log_fp.write(f"\n  【步骤2】LLM期望损失:\n")
        self.log_fp.write(f"    期望错误代价 = P(error) × 错误代价\n")
        self.log_fp.write(f"                 = {p_error} × {cost}\n")
        self.log_fp.write(f"                 = {llm_error_cost:.2f}元\n")
        self.log_fp.write(f"    延迟代价 = {LLM_LATENCY['slm']}ms × {COST_CONFIG['latency_cost_per_ms']}元/ms\n")
        self.log_fp.write(f"            = {llm_latency_cost:.4f}元\n")
        self.log_fp.write(f"    → LLM总期望损失: {llm_total:.2f}元\n")
        
        # 步骤3：人工期望损失
        wait_seconds = queue_state['expected_wait']
        wait_ms = wait_seconds * 1000
        delay_cost = wait_ms * COST_CONFIG['latency_cost_per_ms']
        self.log_fp.write(f"\n  【步骤3】人工期望损失:\n")
        self.log_fp.write(f"    固定成本: {COST_CONFIG['human_fixed_cost']}元\n")
        self.log_fp.write(f"    等待时间: {wait_seconds:.1f}秒 = {wait_ms:.0f}ms\n")
        self.log_fp.write(f"    等待代价: {wait_ms:.0f}ms × {COST_CONFIG['latency_cost_per_ms']}元/ms = {delay_cost:.2f}元\n")
        self.log_fp.write(f"    → 人工总期望损失: {human_total:.2f}元\n")
        
        # 步骤4：决策
        self.log_fp.write(f"\n  【步骤4】决策:\n")
        self.log_fp.write(f"    LLM: {llm_total:.2f}元 vs 人工: {human_total:.2f}元\n")
        
        # 步骤5：实际成本
        if decision.assign_to == "human":
            self.log_fp.write(f"    比较: {llm_total:.2f} > {human_total:.2f}\n")
            self.log_fp.write(f"    → 转人工\n")
            self.log_fp.write(f"\n  【步骤5】实际成本:\n")
            self.log_fp.write(f"    人工成本: {actual_cost_breakdown['human']:.2f}元\n")
            self.log_fp.write(f"    等待代价: {actual_cost_breakdown['wait']:.2f}元\n")
            self.log_fp.write(f"    错误代价: 0元\n")
            self.log_fp.write(f"    → 本请求成本: {actual_cost_breakdown['total']:.2f}元\n")
            accum['human_count'][level] += 1
        else:
            self.log_fp.write(f"    比较: {llm_total:.2f} ≤ {human_total:.2f}\n")
            self.log_fp.write(f"    → 给LLM\n")
            self.log_fp.write(f"\n  【步骤5】实际成本:\n")
            self.log_fp.write(f"    错误代价: {actual_cost_breakdown['error']:.2f}元 (期望损失)\n")
            self.log_fp.write(f"    延迟代价: {actual_cost_breakdown['latency']:.4f}元\n")
            self.log_fp.write(f"    人工成本: 0元\n")
            self.log_fp.write(f"    → 本请求成本: {actual_cost_breakdown['total']:.2f}元\n")
            accum['llm_count'][level] += 1
        
        # 累加
        total_so_far = accum['cumulative_error'] + accum['cumulative_latency'] + accum['cumulative_human']
        self.log_fp.write(f"\n  【累加】错误:{accum['cumulative_error']:.2f} 延迟:{accum['cumulative_latency']:.2f} 人工:{accum['cumulative_human']:.2f} = {total_so_far:.2f}元\n")
        self.log_fp.flush()

    def _log_final_summary(self, total_error_cost: float, total_latency_cost: float, total_human_cost: float):
        """记录最终统计日志"""
        if not self.log_fp:
            return
            
        accum = self._log_accumulators
        total_cost = total_error_cost + total_latency_cost + total_human_cost
        
        self.log_fp.write(f"\n{'='*100}\n")
        self.log_fp.write("最终统计\n")
        self.log_fp.write(f"{'='*100}\n")
        
        self.log_fp.write(f"\n【总成本构成】\n")
        self.log_fp.write(f"  错误代价: {total_error_cost:.2f}元 ({total_error_cost/total_cost*100:.1f}%)\n")
        self.log_fp.write(f"  延迟代价: {total_latency_cost:.2f}元 ({total_latency_cost/total_cost*100:.1f}%)\n")
        self.log_fp.write(f"  人工成本: {total_human_cost:.2f}元 ({total_human_cost/total_cost*100:.1f}%)\n")
        self.log_fp.write(f"  总计: {total_cost:.2f}元\n")
        
        self.log_fp.write(f"\n【路由分布】\n")
        for level in ['high', 'mid', 'low']:
            llm = accum['llm_count'][level]
            human = accum['human_count'][level]
            total = llm + human
            if total > 0:
                self.log_fp.write(f"  {level.upper()}: LLM={llm}, 人工={human} (人工占比{human/total*100:.1f}%)\n")
        
        self.log_fp.flush()

    def _handle_completion(self, current_time: float):
        """处理服务完成"""
        completed = self.queue.process_completions(current_time)
        for request in completed:
            # 人类服务的实际成本（没有错误，只有延迟和固定成本）
            request.actual_cost = COST_CONFIG["human_fixed_cost"]
            self.completed_requests.append(request)
    
    def _compute_llm_actual_cost(self, request: Request) -> float:
        """
        计算 LLM 处理的期望成本（固定值，非随机）
        期望成本 = P(error) × 错误代价 + (1-P(error)) × 延迟代价
        这样结果可复现
        """
        # 期望错误代价
        expected_error_cost = request.llm_error_prob * request.true_cost
        # 期望延迟代价（无论答对答错都有延迟）
        latency_cost = LLM_LATENCY["slm"] * COST_CONFIG["latency_cost_per_ms"]
        # 总期望成本
        return expected_error_cost + latency_cost
    
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
                # LLM 处理：使用期望损失（固定值，可复现）
                # 期望错误代价 = P(error) × 错误代价
                expected_error_cost = r.llm_error_prob * r.true_cost
                latency_cost = LLM_LATENCY["slm"] * COST_CONFIG["latency_cost_per_ms"]
                total_error_cost += expected_error_cost
                total_latency_cost += latency_cost
                
                # 灾难性错误期望：所有High风险（无论P(error)多大）
                if r.true_risk_level == "high":
                    catastrophic_errors += r.llm_error_prob  # 期望错误数（小数）
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
            # 处理 llm_forced 情况，归类到 llm
            assigned = r.assigned_to if r.assigned_to in ["llm", "human"] else "llm"
            risk_dist[r.true_risk_level][assigned] += 1
        
        # 统计大模型认知不确定性分布
        print("\n【大模型认知不确定性分布】")
        for level in ["high", "mid", "low"]:
            level_requests = [r for r in self.completed_requests if r.true_risk_level == level]
            if level_requests:
                probs = [r.llm_error_prob for r in level_requests]
                avg_prob = np.mean(probs)
                min_prob = np.min(probs)
                max_prob = np.max(probs)
                std_prob = np.std(probs)
                print(f"  {level.upper()}: 均值={avg_prob:.3f}, 范围=[{min_prob:.3f}, {max_prob:.3f}], 标准差={std_prob:.3f}")
        
        # 如果有日志文件，写入汇总并关闭
        if self.log_fp:
            # 写入详细最终统计
            self._log_final_summary(total_error_cost, total_latency_cost, total_human_cost)
            
            # 写入简化汇总
            self.log_fp.write(f"\n{'='*100}\n")
            self.log_fp.write("仿真完成\n")
            self.log_fp.write(f"{'='*100}\n")
            self.log_fp.write(f"总请求: {len(self.completed_requests)}条\n")
            self.log_fp.write(f"转人工: {len(human_requests)}条 ({len(human_requests)/len(self.completed_requests)*100:.1f}%)\n")
            self.log_fp.write(f"给LLM: {len(llm_requests)}条 ({len(llm_requests)/len(self.completed_requests)*100:.1f}%)\n")
            self.log_fp.write(f"\n成本构成:\n")
            self.log_fp.write(f"  错误代价: {total_error_cost:.2f}元\n")
            self.log_fp.write(f"  延迟代价: {total_latency_cost:.2f}元\n")
            self.log_fp.write(f"  人工成本: {total_human_cost:.2f}元\n")
            self.log_fp.write(f"  总成本: {total_error_cost + total_latency_cost + total_human_cost:.2f}元\n")
            self.log_fp.flush()
            self.log_fp.close()
        
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
