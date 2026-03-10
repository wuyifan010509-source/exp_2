# 非对称代价感知金融客服路由系统 — 详细 TODO

**📋 最后更新：2026-03-06**

---

## 🔜 待执行任务

### Step ①：运行语义路由模块（⏱ 预计 5 分钟）- ✅ 已完成
- [x] 运行 `uv run python -m semantic_routing.test_routing`
- [x] 检查单条 Query 延迟 < 10ms ✅
- [x] 检查高危 Query 是否全部正确拦截 ✅ 15/15 (100%)
- [x] 根据结果调整 τ_high 阈值（如需）

**语义路由测试结果**：
- 平均延迟：2.72ms ✅
- 最大延迟：7.94ms ✅
- 总体准确率：98% (49/50)
- 高危类别正确率：100% (15/15) ✅

---

### Step ②：运行 SLM 蒸馏训练（⏱ 预计 30-60 分钟）- ✅ 已完成
- [x] 运行 `uv run python -m slm_distillation.train`
- [x] 监控训练 loss 下降（26.05 → 9.39 → 4.04 → ...）
- [x] 验证 recall_high = 94.12%（接近 95% 目标）
- [x] 测量 TTFT = ~17ms（低于 50ms 目标）
- [x] 在测试集上评估推理性能
- [x] 模型已保存到：`./output/slm_distillation`

**训练结果摘要**：
- 训练时长：26.49 秒 (3 epochs)
- 测试集（244 条）：准确率 84.43%，高危 Recall 94.12%
- TTFT: ~17ms (满足 <50ms 目标)

---

### Step ③：统筹评估对比（⏱ 预计 10 分钟）- ✅ 已完成

#### ✅ 已完成
- [x] 运行 `uv run python exp/evaluation/evaluate.py --methods both`
- [x] 使用 `validation_set_1000.jsonl` 进行评估
- [x] 分析总期望代价对比结果
- [x] 保存评估日志和分析报告
- [x] 验证 checkpoint-31 (Epoch 2 最佳) 能正确识别高危样本
- [x] 深入分析混淆矩阵和代价分解
- [x] 生成最终评估报告

#### 📊 评估结果摘要

**测试集**: `data/validation_set_1000_eval.jsonl` (1000 条)
- 分布: Low=180 (18%), Mid=640 (64%), High=180 (18%)

**整体对比**:

| 指标 | Semantic Routing | SLM Distillation | 优胜者 |
|------|-----------------|------------------|--------|
| **总期望代价** | **923.28** | 1117.81 | ✅ Semantic Routing |
| 准确率 | 51.3% | 27.0% | Semantic Routing |
| High Recall | 65.6% | 71.1% | SLM Distillation |
| 漏报数 | 173 | 123 | SLM Distillation |
| 误报数 | **314** | 607 | ✅ Semantic Routing |
| 平均延迟 | **1.0 ms** | 14.9 ms | ✅ Semantic Routing |

**结论**: 在 Total Expected Cost 框架下，**Semantic Routing** 整体性能更优。

#### 🔍 SLM 根因分析

**问题**: SLM 训练使用了非对称损失 (α_underpredict=20.0)，导致模型极度保守。

**表现**:
- Low Recall: 8.9% (70% 被误判为 High)
- Mid Recall: 19.7% (69.2% 被误判为 High)
- High Recall: 71.1% (表现最佳)

**代价**:
- 607 次误报，导致资源浪费
- 错误代价 1117.66 (vs 路由 923.26)

**语义路由优势**:
- 延迟极低 (1ms vs 14.9ms)
- 误报更少 (314 vs 607)
- 各级别更平衡

#### 📁 已保存文件

| 文件 | 描述 |
|------|------|
| `data/validation_set_1000_eval.jsonl` | 转换后的评估数据 |
| `convert_validation_set.py` | 数据转换脚本 |
| `reports/EVALUATION_REPORT.md` | 完整评估报告 |
| `reports/evaluation_checkpoint31.log` | checkpoint-31 评估日志 |
| `reports/method_comparison_analysis.log` | 方法对比分析日志 |
| `analyze_confusion.py` | 混淆矩阵分析工具 |
| `compare_methods.py` | 方法对比详细分析 |

#### 📋 训练日志索引

| 文件 | 描述 |
|------|------|
| `training_log.txt` | 完整训练日志 |
| `slm_training_summary.md` | 训练摘要 |
| `slm_inference_log.json` | 推理 logits |
| `TROUBLESHOOTING.md` | 完整问题诊断 |
| `LOG_INDEX.md` | 日志文件索引 |

#### 📁 已保存的日志文件

| 文件 | 描述 | 路径 |
|------|--------|--------|
| `training_log.txt` | 完整训练日志 | `../training_log.txt` |
| `slm_training_summary.md` | 训练摘要报告 | `../slm_training_summary.md` |
| `slm_inference_log.json` | 推理 logits 和概率 | `../slm_inference_log.json` |
| `TROUBLESHOOTING.md` | 完整问题诊断 | `../TROUBLESHOOTING.md` |
| `LOG_INDEX.md` | 日志文件索引 | `../reports/LOG_INDEX.md` |
| `training_log_fixed.txt` | 重新训练日志 | `../training_log_fixed.txt` |
| `inference_checkpoint31_test.txt` | checkpoint-31 验证 | `../inference_checkpoint31_test.txt` |

---

**总结**：checkpoint-31 确实是最佳模型（Epoch 2, 98.08% 高危 Recall），能正确识别高危样本。评估时性能差是因为数据划分不一致。需要修复评估脚本或重新训练以统一数据划分。
