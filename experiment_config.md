# 比较实验完整配置文档

> 实验文件：`scripts/run_priority_comparison.py`
> 生成时间：2026年3月10日（更新版）

## 1. 实验概述

**实验目的**：验证优先级队列能提高High风险转接率，同时保持较低延迟

**对比策略（5种）**：
1. LLM-Only (基线)
2. Vanilla L2D (传统阈值)
3. Static Cost (静态代价)
4. Dynamic (原版动态调度)
5. Dynamic Priority (优先级队列版本)

---

## 2. 数据配置

### 2.1 测试数据集
```python
test_data = "./data/validation/validation_set_1000_eval_with_prob.jsonl"
num_requests = 300  # 使用300条请求
```

### 2.2 风险等级分布（300条样本）

| 等级 | 样本数 | 占比 | 错误代价 | LLM错误概率（均值±标准差） |
|------|--------|------|----------|---------------------------|
| **High** | 54 | 18% | 5000元 | 0.798 ± 0.211 |
| **Mid** | 190 | 63.3% | 50元 | 0.420 ± 0.162 |
| **Low** | 56 | 18.7% | 1元 | 0.251 ± 0.090 |

### 2.3 代价倍数关系
- High/Mid = 100倍（5000 vs 50）
- High/Low = 5000倍（5000 vs 1）
- Mid/Low = 50倍（50 vs 1）

---

## 3. 路由策略参数

### 3.1 LLM-Only
- **策略**：所有请求全部由LLM处理
- **参数**：无

### 3.2 Vanilla L2D
```python
threshold = 0.2  # 置信度阈值
# 当 P(LLM错) > 0.2 时转人工
```

### 3.3 Static Cost
```python
human_cost_threshold = 100.0  # 静态成本阈值
# 当 LLM期望损失 > 100元 时转人工
```

### 3.4 Dynamic (原版)
```python
# 动态计算：比较 LLM期望损失 vs 人类期望损失
# 人类成本 = 30元 + 实时等待时间 × 0.0005元/ms
```

### 3.5 Dynamic Priority (优先级队列)
```python
# 使用优先级队列，High/Mid/Low分别计算预期等待时间
# 优先级: High > Mid > Low（非抢占式）
```

---

## 4. 人工队列配置（QUEUE_CONFIG）

```python
QUEUE_CONFIG = {
    "service_rate_mu": 1/120,    # 服务率：每120秒（2分钟）处理1个请求
    "num_servers_c": 5,           # 5个专家
    "max_queue_length": 10000,    # 最大队列长度
}
```

**处理能力**：
- 单个专家：30 请求/小时
- 5个专家：150 请求/小时

---

## 5. 流量模式（4小时仿真周期）

| 时间段 | 持续时间 | 到达率 (req/min) | 到达率 (req/s) | 说明 |
|--------|----------|-----------------|---------------|------|
| 0-30min | 30分钟 | **10.0** | 0.167 | 早盘高峰 |
| 30-120min | 90分钟 | **2.0** | 0.033 | 平时 |
| 120-150min | 30分钟 | **6.0** | 0.100 | 午盘高峰 |
| 150-240min | 90分钟 | **2.0** | 0.033 | 平时 |

**总仿真时长**：4小时（240分钟 = 14400秒）

**流量生成**：泊松过程，指数分布到达间隔
```python
inter_arrival = np.random.exponential(1.0 / lambda_rate)
```

---

## 6. 成本配置（COST_CONFIG）

```python
COST_CONFIG = {
    # 错误代价（业务损失）
    "error_cost": {
        "low": 1,      # 低风险问题答错损失
        "mid": 10,     # 中风险问题答错损失（实际数据使用50）
        "high": 1000,  # 高风险问题答错损失（实际数据使用5000）
    },
    
    # 延迟代价（每毫秒）
    "latency_cost_per_ms": 0.0005,  # 0.5元/秒 = 30元/分钟
    
    # 人工固定成本
    "human_fixed_cost": 30,  # 每次人工服务成本
}
```

**延迟成本说明**：
- 当前值：0.0005 元/ms = **0.5元/秒** = **30元/分钟**
- 调整位置：`exp/scheduling_simulation/config.py` 第21行
- 影响：值越大，排队延迟成本越高，策略越倾向于给LLM（避免排队）

---

## 7. LLM延迟配置

```python
LLM_LATENCY = {
    "slm": 16,        # 小模型（Qwen2.5-0.5B）：16ms
    "llm_32b": 95,    # 大模型（Qwen2.5-32B）：95ms
}
```

**注意**：实验中主要使用SLM（16ms）计算延迟代价

---

## 8. SLA约束

```python
SLA_CONSTRAINTS = {
    "high_risk_max_wait_ms": 30000,  # 高危请求30秒内必须响应
}
```

---

## 9. CER（灾难性错误率）计算规则

**统计逻辑**（`simulator.py:616-618`）：
```python
# 灾难性错误期望：所有High风险（无论P(error)多大）
if r.true_risk_level == "high":
    catastrophic_errors += r.llm_error_prob  # 期望错误数（小数）
```

**计入条件**：
1. 分配给LLM的请求
2. **High风险等级**（无论P(error)多大，全部计入）

**不计入的情况**：
- 转人工处理的High风险请求
- Mid/Low风险请求（无论P(error)多大）

**计算方式**：
```
CER = Σ P(error) / 总请求数
```

---

## 10. 随机种子

```python
np.random.seed(42)  # 确保实验可重复
```

---

## 11. 输出配置

```python
output_dir = "./output/scheduling_simulation_priority/"
output_file = "comparison_results.json"
```

---

## 12. 关键公式

### 12.1 LLM期望损失
```
Cost_LLM = P(error) × C_error + Latency_LLM × C_latency
```

### 12.2 人类期望损失
```
Cost_Human = C_human_fixed + Expected_Wait × C_latency
```

### 12.3 决策规则
- 当 `Cost_LLM > Cost_Human` 时，转人工
- 当 `Cost_LLM ≤ Cost_Human` 时，给LLM

---

## 13. 评估指标

1. **ETC** (Expected Total Cost)：总期望成本
2. **CER** (Catastrophic Error Rate)：灾难性错误率
3. **Avg Queueing Delay**：平均排队延迟（ms）
4. **Deferral Rate**：人工转交率
5. **High Risk Deferral Rate**：High风险转接率（核心指标）

---

## 14. 文件依赖关系

```
scripts/run_priority_comparison.py
  ├── exp/scheduling_simulation/__init__.py
  │     ├── queue_model.py (MMcQueue, Request, TrafficGenerator)
  │     ├── routers.py (四种基础路由策略)
  │     ├── priority_router.py (优先级队列路由)
  │     ├── priority_queue.py (非抢占式优先队列)
  │     ├── simulator.py (调度仿真器)
  │     ├── config.py (配置参数)
  │     └── evaluation.py (评估指标)
  └── data/validation/validation_set_1000_eval_with_prob.jsonl
```

---

## 15. 运行命令

```bash
uv run python scripts/run_priority_comparison.py
```

---

*文档更新：已修改CER计算逻辑，现在所有High风险都计入（不限制P(error) > 0.5）*
