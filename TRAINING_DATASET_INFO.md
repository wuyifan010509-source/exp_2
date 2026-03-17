# EXP-2 高危/中危/低危训练数据集说明

## 数据集位置

**主训练集**: `/home/iilab9/scholar-papers/experiments/intention/exp-2/data/processed/final_labeled_data.jsonl`

**样本总数**: 2,435 条

## 风险等级定义

EXP-2 使用 **3级风险分类** 体系：

| 风险等级 | 代价 (Cost) | 典型场景 | 业务含义 |
|---------|------------|---------|---------|
| **Low (低危)** | 1 | 闲聊、问候、天气 | 答错几乎无损失 |
| **Mid (中危)** | 50 | 交易规则、操作流程 | 答错有一定损失 |
| **High (高危)** | 1000-2000 | 强平、自杀倾向、资金异常 | 答错造成严重损失 |

## 数据类别映射

### Category → Cost 映射关系

```
Category  样本数    Cost        风险等级    场景描述
─────────────────────────────────────────────────────────────
B1        196       1          Low        闲聊、天气、问候
B2        101       1          Low        通用对话
A1        250       50         Mid        基础交易操作
A2        598       50         Mid        账户管理
A3        250       500        High       资金安全、异常
C1        127       2000       High       极端情绪、轻生
C2        120       1000       High       资金风险、投诉
D1        100       100        Mid-High   复杂业务咨询
D2        100       200        Mid-High   纠纷处理
D3        100       1000       High       监管相关
OOS       493       -          OOS        域外问题（无代价）
```

### 风险等级详细分布

**Low (低危)** - 总代价 = 1
- **cost_label=1**: 547 条
  - B1 (闲聊/天气): 196 条
  - B2 (通用对话): 101 条
  - 其他: 250 条

**Mid (中危)** - 总代价 = 50
- **cost_label=50**: 1,091 条
  - A1 (基础交易): 250 条
  - A2 (账户管理): 598 条
  - 其他: 243 条

**High (高危)** - 总代价 = 1000-2000
- **cost_label=2000**: 127 条 (C1 - 极端情绪)
- **cost_label=1000**: 250 条 (C2, D3 - 资金风险)
- **cost_label=500**: 220 条 (A3 - 资金安全)
- **cost_label=200**: 100 条 (D2 - 纠纷)
- **cost_label=100**: 100 条 (D1 - 复杂咨询)

## 锚点示例数据

**文件**: `exp-semantic_routing/anchors.py`

### Low (低风险) - Cost=1
```python
[
    "你好", "在吗", "谢谢", "再见",
    "今天天气怎么样", "讲个笑话", "你是机器人吗",
    "周末快乐", "早上好", "节日快乐",
    "你们公司在哪里", "客服工作时间",
    "怎么联系人工", "APP怎么下载", "网站打不开了"
]
```

### Mid (中风险) - Cost=50
```python
[
    "怎么开通创业板", "科创板怎么开通", "港股通开通条件",
    "基金赎回几天到账", "银证转账时间", "交易密码忘了怎么办",
    "怎么修改绑定银行卡", "风险测评过期了", "新股申购条件",
    "可转债怎么打新", "分红方式怎么修改", "怎么查看交割单",
    "交易手续费多少", "印花税怎么算", "怎么导出对账单"
]
```

### High (高风险) - Cost=1000
```python
[
    "我要跳楼", "不想活了", "亏光了活不下去",
    "准备自杀", "死给你们看",
    "我要销户退钱", "全部清仓不玩了",
    "融资爆仓了怎么办", "担保比例低于130", "强制平仓了",
    "穿仓了要补钱吗", "期货公司追保", "期权到期忘记行权",
    "我要举报你们", "找记者曝光"
]
```

## 训练数据格式

**JSONL 格式示例**:
```json
{"query": "今天天气怎么样", "category": "B1", "cost_label": 1}
{"query": "我要跳楼", "category": "C1", "cost_label": 2000}
{"query": "怎么开通创业板", "category": "A1", "cost_label": 50}
{"query": "融资爆仓了怎么办", "category": "C2", "cost_label": 1000}
```

**字段说明**:
- `query`: 用户问题文本
- `category`: 问题类别代码 (A1/A2/A3/B1/B2/C1/C2/D1/D2/D3/OOS)
- `cost_label`: 错误代价标签 (1, 50, 100, 200, 500, 1000, 2000)

## 数据用途

### 1. 语义路由训练
**文件**: `exp/semantic_routing/router.py`
- 使用锚点计算余弦相似度
- 实现非对称代价惩罚（τ_high=0.7 强制拦截高危）

### 2. SLM蒸馏训练
**文件**: `scripts/train_slm_classifier.py`
- 将 cost_label 映射为 3 类: low(0), mid(1), high(2)
- 使用非对称损失函数 (α=10 惩罚漏报高危)

**映射代码**:
```python
cost_to_class = {1: 0, 50: 1, 100: 1, 200: 1, 500: 2, 1000: 2, 2000: 2}
# class 0=low, class 1=mid, class 2=high
```

### 3. 调度仿真实验
**文件**: `exp/scheduling_simulation/simulator.py`
- 为每个查询附加 true_level 和 true_cost
- 测试不同调度策略（Dynamic vs Priority Queue）

## 关键业务约束

**非对称代价设计**:
- **漏报高危**（High→Low）: 惩罚 = 10倍
- **误报低危**（Low→High）: 惩罚 = 1倍
- **设计哲学**: "宁可误报，绝不漏报高危"

**核心指标**:
- 高危Recall > 95%（C1/A3类不能漏判）
- 总期望代价最小化
- 延迟 < 100ms（实时交互要求）

## 文件清单

**训练数据**:
- `data/processed/final_labeled_data.jsonl` (2,435条) - 主训练集
- `data/processed/training_data.jsonl` (旧版本)
- `data/checkpoints/expanded_*.jsonl` - 数据增强中间结果

**验证数据**:
- `data/validation/validation_set_1000.jsonl` (1,000条)
- `data/validation/validation_set_OOS.jsonl` (OOS验证集)
- `data/processed/test_split.jsonl` (测试集)

**锚点定义**:
- `exp/semantic_routing/anchors.py` - 45条锚点Query
- `exp/scheduling_simulation/config.py` - 代价配置

**实验结果**:
- `output/slm_distillation/` - SLM蒸馏模型输出
- `exp/reports/EXPERIMENT_REPORT.md` - 实验报告
