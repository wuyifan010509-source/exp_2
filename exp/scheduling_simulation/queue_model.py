"""
M/M/c 排队模型仿真器
支持时变到达率的排队系统模拟
"""

import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from collections import deque
import time


@dataclass
class Request:
    """请求对象"""
    id: int
    arrival_time: float  # 到达时间戳（秒）
    query: str
    true_risk_level: str  # low/mid/high
    true_cost: float
    
    # 调度决策相关
    llm_error_prob: float = 0.0  # P(LLM错)
    assigned_to: str = ""  # "llm" 或 "human"
    start_service_time: Optional[float] = None
    completion_time: Optional[float] = None
    
    # 结果
    actual_cost: float = 0.0
    waiting_time: float = 0.0
    service_time: float = 0.0


@dataclass
class Server:
    """服务器（人工坐席）"""
    id: int
    busy: bool = False
    current_request: Optional[Request] = None
    service_completion_time: float = 0.0


class MMcQueue:
    """
    M/M/c 排队模型
    - 到达过程：泊松过程（时变到达率 λ(t)）
    - 服务过程：指数分布（固定服务率 μ）
    - c 个并行服务器
    """
    
    def __init__(
        self,
        num_servers: int,
        service_rate: float,
        max_queue_length: int = 100,
        name: str = "HumanQueue"
    ):
        """
        Args:
            num_servers: 服务器数量 c
            service_rate: 服务率 μ（每秒处理的请求数）
            max_queue_length: 最大队列长度
            name: 队列名称
        """
        self.num_servers = num_servers
        self.service_rate = service_rate
        self.max_queue_length = max_queue_length
        self.name = name
        
        # 初始化服务器
        self.servers = [Server(i) for i in range(num_servers)]
        
        # 等待队列
        self.queue: deque = deque()
        
        # 统计信息
        self.stats = {
            "total_arrivals": 0,
            "total_completions": 0,
            "total_rejections": 0,
            "total_waiting_time": 0.0,
            "total_service_time": 0.0,
            "queue_length_history": [],
            "waiting_time_history": [],
        }
        
        # 当前时间
        self.current_time = 0.0
        
    def get_idle_server(self) -> Optional[Server]:
        """获取空闲服务器"""
        for server in self.servers:
            if not server.busy:
                return server
        return None
    
    def get_expected_waiting_time(self) -> float:
        """
        计算预期等待时间 E[W]
        使用 Erlang C 公式近似
        """
        # 简化的等待时间估计：
        # E[W] = (队列中人数 + 正在服务的人数) / (c * μ)
        busy_count = sum(1 for s in self.servers if s.busy)
        queue_length = len(self.queue)
        
        if busy_count < self.num_servers and queue_length == 0:
            return 0.0
        
        # 基于 Little's Law 的估计
        # L = λW, 所以 W = L / λ
        # 这里用当前队列状态估计
        effective_service_rate = self.num_servers * self.service_rate
        total_jobs = queue_length + busy_count
        
        return total_jobs / effective_service_rate if effective_service_rate > 0 else float('inf')
    
    def is_empty(self) -> bool:
        """检查队列是否为空（兼容优先级队列）"""
        return len(self.queue) == 0
    
    def get_queue_state(self) -> Dict:
        """获取当前队列状态"""
        busy_count = sum(1 for s in self.servers if s.busy)
        return {
            "queue_length": len(self.queue),
            "busy_servers": busy_count,
            "idle_servers": self.num_servers - busy_count,
            "expected_wait": self.get_expected_waiting_time(),
            "utilization": busy_count / self.num_servers,
        }
    
    def enqueue(self, request: Request, current_time: float) -> bool:
        """
        将请求加入队列
        Returns: 是否成功加入
        """
        self.current_time = current_time
        
        # 检查是否有空闲服务器可以立即服务
        idle_server = self.get_idle_server()
        if idle_server:
            # 立即开始服务
            self._start_service(request, idle_server, current_time)
            return True
        
        # 检查队列是否已满
        if len(self.queue) >= self.max_queue_length:
            self.stats["total_rejections"] += 1
            return False
        
        # 加入等待队列
        self.queue.append(request)
        self.stats["total_arrivals"] += 1
        return True
    
    def _start_service(self, request: Request, server: Server, current_time: float):
        """开始服务"""
        server.busy = True
        server.current_request = request
        
        # 基于请求ID生成固定的服务时间（控制变量）
        # 使用截断正态分布，服务时间集中在120秒附近（60-180秒）
        np.random.seed(request.id + 42)  # +42确保与仿真随机种子不冲突
        
        # 均值120秒，标准差20秒，范围60-180秒
        mean_service_time = 120.0
        std_service_time = 20.0
        min_service_time = 60.0
        max_service_time = 180.0
        
        # 生成截断正态分布
        service_time = np.random.normal(mean_service_time, std_service_time)
        service_time = np.clip(service_time, min_service_time, max_service_time)
        
        np.random.seed()  # 重置随机种子
        
        server.service_completion_time = current_time + service_time
        
        request.start_service_time = current_time
        request.service_time = service_time
        
        if request.assigned_to == "human":
            request.waiting_time = current_time - request.arrival_time
            self.stats["total_waiting_time"] += request.waiting_time
            self.stats["waiting_time_history"].append(request.waiting_time)
            # print(request.waiting_time,",",self.stats["total_waiting_time"])
        
        self.stats["total_service_time"] += service_time
    
    def process_completions(self, current_time: float) -> List[Request]:
        """
        处理在当前时间之前完成的服务
        Returns: 完成的请求列表
        """
        self.current_time = current_time
        completed = []
        
        for server in self.servers:
            if server.busy and server.service_completion_time <= current_time:
                # 服务完成
                request = server.current_request
                request.completion_time = server.service_completion_time
                completed.append(request)
                
                # 释放服务器
                server.busy = False
                server.current_request = None
                
                self.stats["total_completions"] += 1
                
                # 从队列中取出下一个请求
                if self.queue:
                    next_request = self.queue.popleft()
                    self._start_service(next_request, server, server.service_completion_time)
        
        return completed
    
    def get_average_waiting_time(self) -> float:
        """平均等待时间"""
        if self.stats["total_completions"] == 0:
            return 0.0
        print(self.stats["total_waiting_time"])
        return self.stats["total_waiting_time"] / self.stats["total_completions"]
    
    def get_server_utilization(self) -> float:
        """服务器平均利用率"""
        busy_time = sum(
            s.service_completion_time - max(0, s.service_completion_time - 1.0/self.service_rate)
            for s in self.servers if s.busy
        )
        return busy_time / (self.num_servers * self.current_time) if self.current_time > 0 else 0.0


class TrafficGenerator:
    """流量生成器 - 模拟时变到达率"""
    
    def __init__(self, traffic_pattern: Dict):
        """
        Args:
            traffic_pattern: 流量模式定义
                {
                    "normal": {"lambda": 2.0, "duration": 3600},  # 1小时
                    "morning_peak": {"lambda": 10.0, "duration": 1800},  # 30分钟
                    ...
                }
        """
        self.traffic_pattern = traffic_pattern
        self.current_phase_idx = 0
        self.phase_start_time = 0.0
        
    def get_current_lambda(self, current_time: float) -> float:
        """获取当前时刻的到达率 λ"""
        phases = list(self.traffic_pattern.keys())
        
        # 简化为循环模式
        total_duration = sum(p["duration"] for p in self.traffic_pattern.values())
        time_in_cycle = current_time % total_duration
        
        accumulated = 0.0
        for phase_name, phase_config in self.traffic_pattern.items():
            if accumulated <= time_in_cycle < accumulated + phase_config["duration"]:
                return phase_config["lambda"]
            accumulated += phase_config["duration"]
        
        return phases[0]["lambda"]
    
    def generate_next_arrival_time(self, current_time: float) -> float:
        """
        生成下一个到达时间（泊松过程）
        间隔时间服从指数分布
        """
        lambda_t = self.get_current_lambda(current_time)
        if lambda_t <= 0:
            return float('inf')
        
        # 指数分布：P(T > t) = exp(-λt)
        inter_arrival = np.random.exponential(1.0 / lambda_t)
        return current_time + inter_arrival
