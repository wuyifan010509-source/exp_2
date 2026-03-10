# 日志文件索引

**生成时间**: 2026-03-06

---

## 已保存的文件

### 训练相关

| 文件 | 大小 | 描述 | 路径 |
|------|--------|--------|--------|
| `training_log.txt` | 19K | 完整训练日志（所有 epoch） | `../training_log.txt` |
| `slm_training_summary.md` | 2.2K | 训练摘要和问题分析 | `../slm_training_summary.md` |
| `slm_inference_log.json` | 2.0K | 高危样本推理的 logits 和概率 | `../slm_inference_log.json` |
| `inference_validation_log.txt` | 774B | 推理验证的终端输出 | `../inference_validation_log.txt` |

### 评估相关

| 文件 | 大小 | 描述 | 路径 |
|------|--------|--------|--------|
| `EVALUATION_REPORT.md` | 6.5K | 最终评估报告（完整对比分析） | `EVALUATION_REPORT.md` |
| `evaluation_checkpoint31.log` | 5.1K | checkpoint-31 完整评估日志 | `evaluation_checkpoint31.log` |
| `evaluation_validation_1000.log` | 4.8K | validation_set_1000 评估日志 | `evaluation_validation_1000.log` |
| `method_comparison_analysis.log` | 4.3K | 方法对比详细分析 | `method_comparison_analysis.log` |

### 数据转换脚本

| 文件 | 描述 |
|------|------|
| `convert_validation_set.py` | 验证集格式转换（11类→3级） |
| `analyze_confusion.py` | 混淆矩阵分析工具 |
| `compare_methods.py` | 方法对比详细分析 |

---

## 训练关键发现

### 各 Epoch 高危 Recall (Validation Set)

```
Epoch 1: 96.15%
Epoch 2: 98.08% ← 最佳 (checkpoint-31)
Epoch 3: 94.23%
Epoch 4: 90.38%
Epoch 5: 94.23% ← 保存的 epoch
```

### 最终测试集指标 (Test Set)

```
High Recall: 98.08% ✅
Accuracy: 96.71%
```

---

## 评估关键发现

### 整体对比结果 (validation_set_1000)

| 指标 | Semantic Routing | SLM Distillation | 优胜者 |
|------|-----------------|------------------|--------|
| **总期望代价** | **923.28** | 1117.81 | ✅ Semantic Routing |
| 准确率 | 51.3% | 27.0% | Semantic Routing |
| High Recall | 65.6% | 71.1% | SLM Distillation |
| 漏报数 | 173 | 123 | SLM Distillation |
| 误报数 | **314** | 607 | ✅ Semantic Routing |
| 平均延迟 | **1.0 ms** | 14.9 ms | ✅ Semantic Routing |

### SLM 混淆矩阵 (True → Predicted)

```
LOW:   16( 8.9%)  38(21.1%) 126(70.0%)  ← 严重过度预测
MID:   71(11.1%) 126(19.7%) 443(69.2%)  ← 严重过度预测
HIGH:  32(17.8%)  20(11.1%) 128(71.1%)  ← 尚可
```

### 根因分析

**SLM 过度保守的原因**:
- 非对称损失训练: α_underpredict = 20.0 (漏报惩罚是误报的 20 倍)
- 模型策略: "宁可错杀，不可放过"
- 结果: 70% 的 Low 样本被预测为 High

**Semantic Routing 优势**:
- 超低延迟 (1ms vs 14.9ms)
- 误报更少 (314 vs 607)
- 各级别更平衡

---

## 结论

在 Total Expected Cost 框架下，**Semantic Routing 整体性能更优**。

**原因**:
1. 总期望代价更低 (923.28 vs 1117.81)
2. 延迟优势显著 (1ms vs 14.9ms)
3. 资源浪费更少

**SLM 适用场景**: 如对高风险样本 Recall 要求极高 (>=90%)，且可接受较高资源成本的场景。

---

## 下一步

可选改进方向:
1. **降低 SLM 的 α 值**: 从 20.0 降至 5.0-10.0，减少过度预测
2. **调整代价权重**: 根据实际业务场景重新校准损失函数
3. **后处理校准**: 使用阈值调整减少过度预测
4. **改进数据平衡**: 增加低/中风险样本的训练比重

详细分析见: `EVALUATION_REPORT.md`
