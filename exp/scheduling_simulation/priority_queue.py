"""
非抢占式优先级队列模型 (Non-preemptive Priority Queue)

队列结构：
- High优先级队列: H, H, H, ... (新来的H排在现有H后面)
- Mid优先级队列: M, M, M, ...
- Low优先级队列: L, L, L, ...

调度规则：
1. 优先级: High > Mid > Low
2. 非抢占: 正在处理的任务不会被中断
3. 同级别FIFO: 同优先级内，先来的先处理
4. 空闲时: 服务器按优先级从队列取任务

数学期望:
E[delay_H] = (当前服务剩余时间) + (前面已有H的处理时间总和)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field

from .queue_model import MMcQueue, Request, Server


@dataclass 
class PriorityRequest(Request):
    """带优先级标记的请求"""
    priority: int = 0  # 0=Low, 1=Mid, 2=High
    

class NonPreemptivePriorityQueue(MMcQueue):
    """
    非抢占式优先级队列
    
    特点:
    - 三个独立队列：High > Mid > Low
    - 非抢占：不会中断正在处理的任务
    - 服务器空闲时按优先级取任务
    """
    
    def __init__(
        self,
        num_servers: int,
        service_rate: float,
        max_queue_length: int = 100,
        name: str = "PriorityQueue"
    ):
        """
        Args:
            num_servers: 服务器数量
            service_rate: 服务率 μ
            max_queue_length: 每个队列的最大长度
            name: 队列名称
        """
        super().__init__(num_servers, service_rate, max_queue_length * 3, name)
        
        # 三个优先级队列
        self.high_queue: deque = deque()  # 高危风险
        self.mid_queue: deque = deque()   # 中危风险
        self.low_queue: deque = deque()   # 低危风险
        
        # 优先级映射
        self.priority_map = {
            "high": 2,
            "mid": 1,
            "low": 0
        }
        
    def _get_priority(self, level: str) -> int:
        """获取优先级数值"""
        return self.priority_map.get(level, 0)
    
    def enqueue(self, request: Request, current_time: float) -> bool:
        """
        将请求加入对应的优先级队列
        
        High风险排在现有High后面，但在所有Mid/Low前面
        """
        self.current_time = current_time
        
        # 检查是否有空闲服务器可以立即服务
        idle_server = self.get_idle_server()
        if idle_server:
            # 立即开始服务
            self._start_service(request, idle_server, current_time)
            return True
        
        # 根据风险等级选择队列
        priority = self._get_priority(request.true_risk_level)
        
        if priority == 2:  # High
            if len(self.high_queue) >= self.max_queue_length // 3:
                self.stats["total_rejections"] += 1
                return False
            self.high_queue.append(request)
        elif priority == 1:  # Mid
            if len(self.mid_queue) >= self.max_queue_length // 3:
                self.stats["total_rejections"] += 1
                return False
            self.mid_queue.append(request)
        else:  # Low
            if len(self.low_queue) >= self.max_queue_length // 3:
                self.stats["total_rejections"] += 1
                return False
            self.low_queue.append(request)
        
        self.stats["total_arrivals"] += 1
        return True
    
    def _get_next_request_by_priority(self) -> Optional[Request]:
        """
        按优先级获取下一个请求
        
        顺序: High > Mid > Low
        """
        if self.high_queue:
            return self.high_queue.popleft()
        elif self.mid_queue:
            return self.mid_queue.popleft()
        elif self.low_queue:
            return self.low_queue.popleft()
        return None
    
    def get_expected_waiting_time(self, risk_level: str = "high") -> float:
        """
        计算特定风险等级的预期等待时间
        
        使用Little's Law改进版：考虑并行处理能力，避免线性增长
        E[delay] = (该优先级及更高优先级的任务数) / (有效服务率)
        
        Args:
            risk_level: 风险等级 (high/mid/low)
        """
        priority = self._get_priority(risk_level)
        
        # 统计各队列长度
        high_len = len(self.high_queue)
        mid_len = len(self.mid_queue)
        low_len = len(self.low_queue)
        
        # 正在服务的任务数
        busy_count = sum(1 for s in self.servers if s.busy)
        
        # 总服务率（所有专家并行处理）
        total_service_rate = self.num_servers * self.service_rate
        
        if priority == 2:  # High
            # High优先级最高，所有专家都可以处理High任务
            # 总任务 = High队列 + 正在处理的所有任务（因为非抢占）
            total_jobs = high_len + busy_count
            
            # 有效服务率：所有专家都可以处理High
            return total_jobs / total_service_rate if total_service_rate > 0 else float('inf')
            
        elif priority == 1:  # Mid
            # Mid需要等待：所有High任务 + 前面所有Mid任务 + 正在处理的任务
            # 但使用并行计算而非串行累加
            total_jobs = high_len + mid_len + busy_count
            
            # 有效服务率：所有专家
            return total_jobs / total_service_rate if total_service_rate > 0 else float('inf')
            
        else:  # Low
            # Low需要等待所有前面的任务
            total_jobs = high_len + mid_len + low_len + busy_count
            effective_service_rate = self.num_servers * self.service_rate
            
            return total_jobs / effective_service_rate if effective_service_rate > 0 else float('inf')
    
    def is_empty(self) -> bool:
        """检查所有优先级队列是否都为空"""
        return len(self.high_queue) == 0 and len(self.mid_queue) == 0 and len(self.low_queue) == 0
    
    def get_queue_state(self) -> Dict:
        """获取当前队列状态"""
        busy_count = sum(1 for s in self.servers if s.busy)
        
        return {
            "high_queue_length": len(self.high_queue),
            "mid_queue_length": len(self.mid_queue),
            "low_queue_length": len(self.low_queue),
            "total_queue_length": len(self.high_queue) + len(self.mid_queue) + len(self.low_queue),
            "busy_servers": busy_count,
            "idle_servers": self.num_servers - busy_count,
            "expected_wait_high": self.get_expected_waiting_time("high"),
            "expected_wait_mid": self.get_expected_waiting_time("mid"),
            "expected_wait_low": self.get_expected_waiting_time("low"),
            "utilization": busy_count / self.num_servers,
        }
    
    def process_completions(self, current_time: float) -> List[Request]:
        """
        处理完成的任务，并按优先级分配新任务
        
        非抢占式：任务完成后，按优先级从队列取新任务
        """
        self.current_time = current_time
        completed = []
        
        for server in self.servers:
            if server.busy and server.service_completion_time <= current_time:
                # 任务完成
                request = server.current_request
                request.completion_time = server.service_completion_time
                completed.append(request)
                
                # 释放服务器
                server.busy = False
                server.current_request = None
                
                self.stats["total_completions"] += 1
                
                # 按优先级从队列中取出下一个请求
                next_request = self._get_next_request_by_priority()
                if next_request:
                    self._start_service(next_request, server, server.service_completion_time)
        
        return completed
    
    def get_average_waiting_time_by_priority(self) -> Dict[str, float]:
        """按优先级统计平均等待时间"""
        wait_times = {"high": [], "mid": [], "low": []}
        
        # 从完成的请求中统计
        for request in getattr(self, '_completed_requests', []):
            if hasattr(request, 'waiting_time') and request.assigned_to == "human":
                level = request.true_risk_level
                if level in wait_times:
                    wait_times[level].append(request.waiting_time)
        
        return {
            level: np.mean(times) if times else 0.0
            for level, times in wait_times.items()
        }
