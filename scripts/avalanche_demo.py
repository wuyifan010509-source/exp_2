#!/usr/bin/env python3
"""
排队论雪崩效应演示

场景设计：
- 高强度流量冲击（突发峰值）
- 有限的人工坐席（3个）
- 较长的服务时间（平均120秒）
- 所有请求都是High风险（都会尝试转人工）

预期结果：
- Static Cost: 盲转人工 → 队列爆仓 → 后期请求等待500+秒
- Dynamic: 队列感知 → 智能分流 → 延迟控制在合理范围
"""

import sys
sys.path.insert(0, '/home/iilab9/scholar-papers/experiments/intention/exp-2')

import numpy as np
import json
from dataclasses import dataclass

# 设置全局种子确保可复现
np.random.seed(42)

@dataclass
class Request:
    """简化版请求对象"""
    id: int
    arrival_time: float
    risk_level: str = "high"
    llm_error_prob: float = 0.8
    true_cost: float = 2000
    
    # 路由结果
    assigned_to: str = ""
    start_service_time: float = 0.0
    service_time: float = 0.0
    waiting_time: float = 0.0


def generate_avalanche_requests(n_requests=100, peak_rate=2.0, seed=42):
    """
    生成雪崩场景请求流
    
    前50个请求：高到达率（2 req/s）模拟突发流量
    后50个请求：低到达率（0.1 req/s）模拟流量回落
    
    预期：前50个请求快速填满队列，后50个请求被迫长时间等待
    """
    np.random.seed(seed)
    requests = []
    current_time = 0.0
    
    print("=== 生成雪崩场景流量 ===")
    print(f"前50请求：高到达率 {peak_rate} req/s（突发峰值）")
    print(f"后50请求：低到达率 0.1 req/s（流量回落）")
    print()
    
    for i in range(n_requests):
        if i < 50:
            # 高峰期：2 req/s = 平均间隔0.5秒
            inter_arrival = np.random.exponential(1.0 / peak_rate)
        else:
            # 低谷期：0.1 req/s = 平均间隔10秒
            inter_arrival = np.random.exponential(10.0)
        
        current_time += inter_arrival
        
        # 生成固定的服务时间（基于ID，确保一致性）
        np.random.seed(i + 1000)
        service_time = np.random.normal(120, 20)  # 均值120s，标准差20s
        service_time = max(60, min(180, service_time))  # 限制在60-180s
        np.random.seed()
        
        req = Request(
            id=i,
            arrival_time=current_time,
            service_time=service_time
        )
        requests.append(req)
    
    return requests


def static_cost_strategy(requests, num_servers=3, cost_threshold=500):
    """
    Static Cost策略：只看错误代价，不看队列状态
    
    决策逻辑：
    - P(error)×Cost > 阈值 → 转人工
    - 否则 → 给LLM
    
    问题：队列已爆仓还继续转人工！
    """
    print("=== Static Cost策略（无队列感知）===")
    
    servers_busy_until = [0.0] * num_servers  # 每个坐席忙到什么时候
    queue = []  # 等待队列
    results = []
    
    for req in requests:
        # 决策：只看错误代价
        expected_error_cost = req.llm_error_prob * req.true_cost  # 0.8 × 2000 = 1600元
        
        if expected_error_cost > cost_threshold:
            # 转人工（不管队列多长！）
            req.assigned_to = "human"
            
            # 找最早空闲的坐席
            earliest_server = min(servers_busy_until)
            server_idx = servers_busy_until.index(earliest_server)
            
            # 计算开始服务时间
            if earliest_server > req.arrival_time:
                # 需要等待
                req.start_service_time = earliest_server
                req.waiting_time = earliest_server - req.arrival_time
            else:
                # 立即服务
                req.start_service_time = req.arrival_time
                req.waiting_time = 0
            
            # 更新坐席状态
            servers_busy_until[server_idx] = req.start_service_time + req.service_time
            
        else:
            # 给LLM
            req.assigned_to = "llm"
            req.waiting_time = 0
        
        results.append(req)
    
    return results


def dynamic_strategy(requests, num_servers=3, max_queue_length=5):
    """
    Dynamic策略：实时感知队列长度，队列拥堵时分流给LLM
    
    核心机制：
    - 跟踪当前排队中的请求数
    - 当排队数 >= 阈值时，新请求转给LLM
    - 这样可以防止队列无限增长，保护后期请求
    """
    print(f"=== Dynamic策略（队列感知，限流阈值={max_queue_length}）===")
    
    from heapq import heappush, heappop
    
    # 优先队列：(完成时间, 请求ID) - 正在占用坐席的请求
    busy_servers = []
    
    # 等待队列：存储还未开始服务的请求
    waiting_queue = []  # (预计开始时间, service_time, request_id)
    
    results = []
    
    for req in requests:
        current_time = req.arrival_time
        
        # 步骤1：清理已完成服务的请求
        while busy_servers and busy_servers[0][0] <= current_time:
            heappop(busy_servers)
        
        # 步骤2：将等待队列中可以开始服务的请求移入坐席
        # 如果当前有空闲坐席，从等待队列取请求
        while waiting_queue and len(busy_servers) < num_servers:
            # 从等待队列取出最早应该开始的请求
            start_time, svc_time, req_id = waiting_queue.pop(0)
            completion_time = start_time + svc_time
            heappush(busy_servers, (completion_time, req_id))
        
        # 步骤3：计算当前队列状态（等待中的请求数）
        queue_length = len(waiting_queue)
        
        # 步骤4：决策
        # 如果加入当前请求后队列将超过阈值，给LLM
        if queue_length >= max_queue_length:
            req.assigned_to = "llm"
            req.waiting_time = 0
        else:
            # 转人工
            req.assigned_to = "human"
            
            # 判断是立即服务还是需要排队
            if len(busy_servers) < num_servers:
                # 有空闲坐席，立即服务
                req.start_service_time = current_time
                req.waiting_time = 0
                completion_time = current_time + req.service_time
                heappush(busy_servers, (completion_time, req.id))
            else:
                # 需要排队，计算预计开始时间
                # 取最早完成的坐席
                earliest_completion = busy_servers[0][0]
                req.start_service_time = earliest_completion
                req.waiting_time = earliest_completion - current_time
                # 加入等待队列
                waiting_queue.append((req.start_service_time, req.service_time, req.id))
        
        results.append(req)
    
    return results


def analyze_results(name, results):
    """分析结果"""
    human_reqs = [r for r in results if r.assigned_to == "human"]
    llm_reqs = [r for r in results if r.assigned_to == "llm"]
    
    if human_reqs:
        avg_wait = sum(r.waiting_time for r in human_reqs) / len(human_reqs)
        max_wait = max(r.waiting_time for r in human_reqs)
        late_requests = [r for r in human_reqs if r.id >= 50]  # 后50个请求
        avg_late_wait = sum(r.waiting_time for r in late_requests) / len(late_requests) if late_requests else 0
    else:
        avg_wait = max_wait = avg_late_wait = 0
    
    print(f"\n{name} 结果:")
    print(f"  总请求: {len(results)}")
    print(f"  转人工: {len(human_reqs)} ({len(human_reqs)/len(results)*100:.1f}%)")
    print(f"  给LLM: {len(llm_reqs)} ({len(llm_reqs)/len(results)*100:.1f}%)")
    print(f"  平均等待: {avg_wait:.1f}秒")
    print(f"  最大等待: {max_wait:.1f}秒")
    print(f"  后50请求平均等待: {avg_late_wait:.1f}秒 ⚠️")
    
    # 展示前5个和后5个请求的等待时间
    print(f"\n  等待时间分布:")
    print(f"    前5请求: {[f'{r.waiting_time:.1f}s' for r in results[:5] if r.assigned_to == 'human']}")
    print(f"    后5请求: {[f'{r.waiting_time:.1f}s' for r in results[-5:] if r.assigned_to == 'human']}")
    
    return {
        'human_count': len(human_reqs),
        'avg_wait': avg_wait,
        'max_wait': max_wait,
        'late_avg_wait': avg_late_wait
    }


def main():
    print("="*80)
    print("排队论雪崩效应演示")
    print("="*80)
    print()
    print("场景设置:")
    print("  - 总请求: 100个（全部High风险）")
    print("  - 人工坐席: 3个")
    print("  - 服务时间: 平均120秒")
    print("  - 流量模式: 前50个突发高峰(2 req/s)，后50个低谷(0.1 req/s)")
    print()
    print("预期雪崩效应:")
    print("  - 前50请求快速填满3个坐席")
    print("  - 队列开始堆积")
    print("  - 后50请求到达时，前面已有长队")
    print("  - Static Cost继续盲转人工 → 等待时间指数增长")
    print("  - Dynamic队列感知 → 自动分流给LLM")
    print("="*80)
    print()
    
    # 生成雪崩场景请求
    requests = generate_avalanche_requests()
    
    # 运行两种策略
    static_results = static_cost_strategy([Request(id=r.id, arrival_time=r.arrival_time, service_time=r.service_time, llm_error_prob=r.llm_error_prob, true_cost=r.true_cost) for r in requests], num_servers=3)
    dynamic_results = dynamic_strategy([Request(id=r.id, arrival_time=r.arrival_time, service_time=r.service_time, llm_error_prob=r.llm_error_prob, true_cost=r.true_cost) for r in requests], num_servers=3, max_queue_length=3)
    
    # 分析结果
    static_stats = analyze_results("Static Cost", static_results)
    dynamic_stats = analyze_results("Dynamic", dynamic_results)
    
    # 对比
    print("\n" + "="*80)
    print("对比总结")
    print("="*80)
    print(f"{'指标':<30} {'Static Cost':<20} {'Dynamic':<20} {'改进':<15}")
    print("-"*80)
    
    hc_diff = static_stats['human_count'] - dynamic_stats['human_count']
    print(f"{'转人工数量':<30} {static_stats['human_count']:<20} {dynamic_stats['human_count']:<20} {f'-{hc_diff}':<15}")
    
    wait_diff = static_stats['avg_wait'] - dynamic_stats['avg_wait']
    improvement = (wait_diff / static_stats['avg_wait'] * 100) if static_stats['avg_wait'] > 0 else 0
    print(f"{'平均等待时间(秒)':<30} {static_stats['avg_wait']:<20.1f} {dynamic_stats['avg_wait']:<20.1f} {f'-{improvement:.1f}%':<15}")
    
    max_diff = static_stats['max_wait'] - dynamic_stats['max_wait']
    print(f"{'最大等待时间(秒)':<30} {static_stats['max_wait']:<20.1f} {dynamic_stats['max_wait']:<20.1f} {f'-{max_diff:.1f}':<15}")
    
    late_diff = static_stats['late_avg_wait'] - dynamic_stats['late_avg_wait']
    print(f"{'后50请求平均等待(秒)':<30} {static_stats['late_avg_wait']:<20.1f} {dynamic_stats['late_avg_wait']:<20.1f} {f'-{late_diff:.1f}':<15} ⭐")
    
    print("="*80)
    print()
    print("结论:")
    print("  1. Static Cost无队列感知，高峰期继续转人工 → 队列雪崩")
    print("  2. Dynamic实时感知队列，拥堵时分流给LLM → 避免雪崩")
    print("  3. 后50个请求的等待时间差异最明显（雪崩的核心体现）")
    print()


if __name__ == "__main__":
    main()
