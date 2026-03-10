# 人机混合智能体调度仿真实验

## 快速开始

### 1. 运行完整对比实验

```bash
# 使用默认配置运行（需要1000条测试数据）
uv run python -m exp.scheduling_simulation

# 或使用脚本
uv run python scripts/run_scheduling_simulation.py
```

### 2. 自定义参数

```bash
# 指定测试数据和输出目录
uv run python -m exp.scheduling_simulation \
    --test_data ./data/validation_set_1000_eval.jsonl \
    --output_dir ./output/scheduling_simulation \
    --max_time 14400 \
    --num_requests 1000
```

参数说明：
- `--test_data`: 测试数据路径（JSONL格式）
- `--output_dir`: 输出目录
- `--max_time`: 仿真时长（秒，默认4小时=14400秒）
- `--num_requests`: 使用的请求数量
- `--seed`: 随机种子（默认42）

## 实验设计

### 四种对比策略

1. **LLM-Only**: 所有请求全由LLM回答，无人工介入
2. **Vanilla L2D**: 固定阈值（P(LLM错) > 0.2 转人工）
3. **Static Cost-Sensitive**: 静态代价比较（R_LLM > 100元 转人工）
4. **Dynamic Queue-Aware (Ours)**: 动态排队感知（实时计算 E[延迟]）

### 流量模式

模拟4小时交易时间：
- **0-30min**: 早盘高峰（10 req/s）
- **30-120min**: 平时（2 req/s）
- **120-150min**: 午盘高峰（6 req/s）
- **150-240min**: 平时（2 req/s）

### 评估指标

**成本与风险**
- ★ **ETC** (Expected Total Cost): 总期望成本
- **CSA** (Cost-Sensitive Accuracy): 代价敏感准确率
- **CER** (Catastrophic Error Rate): 灾难性错误率

**队列性能**
- 平均排队延迟 (ms)
- SLA违约率
- 服务器利用率

**调度行为**
- 人工转交率
- 按风险等级的转交分布

## 输出文件

运行后会在 `output/scheduling_simulation/` 目录生成：

```
output/scheduling_simulation/
├── comparison_table.md          # 对比表格
├── simulation_results.json      # 详细结果（JSON）
├── cost_delay_tradeoff.png      # 成本-延迟权衡曲线
├── routing_behavior.png         # 调度行为分析图
└── traffic_pattern.png          # 流量模式图
```

## 数据结构

### 输入数据格式（JSONL）

```json
{"query": "怎么修改密码", "true_level": "low", "true_cost": 1}
{"query": "为什么强平我", "true_level": "high", "true_cost": 1000}
{"query": "印花税怎么扣", "true_level": "mid", "true_cost": 50}
```

字段说明：
- `query`: 用户查询文本
- `true_level`: 真实风险等级（low/mid/high）
- `true_cost`: 错误代价（元）

### 核心不等式

你的方法核心决策逻辑：

```
if P(LLM错) × C_error > E[实时排队延迟] × C_delay + C_human_fixed:
    转人工
else:
    LLM处理
```

## 扩展开发

### 添加新的调度策略

```python
from exp.scheduling_simulation.routers import BaseRouter, RoutingDecision

class MyRouter(BaseRouter):
    def route(self, request, queue, current_time):
        # 你的决策逻辑
        return RoutingDecision(
            assign_to="llm" or "human",
            expected_cost_llm=cost_llm,
            expected_cost_human=cost_human,
            reason="决策原因"
        )
```

### 接入真实的SLM置信度

修改 `estimate_llm_error_probability` 函数，接入你的SLM模型：

```python
from exp.slm_distillation.inference import SLMCostPredictor

predictor = SLMCostPredictor()

def estimate_llm_error_probability(query: str, true_level: str) -> float:
    result = predictor.predict(query)
    # 将风险等级转换为错误概率
    # ...
    return error_prob
```

## 代码结构

```
exp/scheduling_simulation/
├── __init__.py           # 模块初始化
├── __main__.py           # 主入口
├── config.py             # 配置参数
├── queue_model.py        # M/M/c排队模型
├── routers.py            # 四种调度策略
├── simulator.py          # 仿真引擎
└── evaluation.py         # 评估与可视化
```
