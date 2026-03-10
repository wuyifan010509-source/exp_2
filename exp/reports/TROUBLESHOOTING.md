# 问题诊断报告

**日期**: 2026-03-06
**状态**: 待解决

---

## 🔴 核心问题

### 问题描述
SLM 模型在训练时测试集高危 Recall 达到 **95.24%**，但在综合评估时高危 Recall 只有 **2.00%**，导致总期望代价高达 2410（语义路由仅 640）。

---

## 🔍 根因分析

### 1. 数据标签问题（已解决 ✅）

**问题**：原始标注数据的 `category` 和 `cost_label` 字段不匹配
- 例如：C1 类别（情绪表达）`category="C1"` 但 `cost_label=1`（应为 2000）

**影响链条**：
```
LLM 标注 → category=C1 ✅ 但 cost_label=1 ❌
         ↓
train.py 读 class_label (由 category 派生) → 训练标签正确 ✅
         ↓
evaluate.py 读 cost_label=1 → 推断 true_level="low" ❌
         ↓
模型预测 High，但 ground truth 是 Low → 算成误报 → Recall 崩塌
```

**解决方案**：
- 创建 `fix_labels.py` 根据 `category` 重新生成 `cost_label`、`class_label`、`level`
- 更新 `evaluate.py` 优先读取 `level > class_label > cost_label`

**结果**：数据修复后，有 **497 条高危样本（20.4%）**。

---

### 2. 模型保存问题（待修复 ❌）

**问题**：训练保存的是 **最后一个 epoch** 的模型，而不是 **最佳 epoch** 的模型

**训练日志中各 Epoch 高危 Recall**：

| Epoch | 高危 Recall | 状态 |
|--------|--------------|--------|
| 1      | 92.31%       |        |
| 2      | **98.08%**   | **最佳** |
| 3      | 94.23%       |        |
| 4      | 86.54%       |        |
| 5      | 88.46%       | **已保存** ❌ |

**训练代码配置**：
```python
training_args = TrainingArguments(
    ...
    load_best_model_at_end=True,              # ✅ 配置了
    metric_for_best_model="recall_high",          # ✅ 配置了
    greater_is_better=True,
    ...
)
```

**问题**：HuggingFace Trainer 可能无法识别自定义的 `recall_high` 指标。

**当前保存位置**：`exp/slm_distillation/train.py` 第 302 行
```python
trainer.save_model(OUTPUT_DIR)  # 保存最后 epoch
```

---

## 📊 当前评估结果对比

| 指标 | 语义路由 | SLM 蒸馏 | 差距 |
|--------|-----------|------------|------|
| 准确率 | 59.84% | 22.95% | -36.89% |
| 高危 Recall | **80.00%** | **2.00%** | -78.00% ❌ |
| 漏报数 | 31 | 160 | +415 ❌ |
| 误报数 | 67 | 28 | -39 ✅ |
| 平均延迟 | 2.68 ms | 16.90 ms | +541% |
| 总期望代价 | **639.59** | **2410.30** | +277% ❌ |

**结论**：当前 SLM 模型无法用于生产环境。

---

## 🔧 修复方案

### 方案 A：修改训练脚本，手动保存最佳模型（推荐）

**步骤**：
1. 在训练循环中跟踪各 epoch 的 `eval_recall_high`
2. 训练结束后，保存最佳 epoch 的模型权重
3. 删除旧的 checkpoint，只保留最佳

**优点**：
- 直接控制保存逻辑，不依赖 Trainer 的 metric 机制
- 可以确保保存真正最优的模型

**实现位置**：`exp/slm_distillation/train.py` 第 290-304 行

---

### 方案 B：使用自定义 Callback（较复杂）

**步骤**：
1. 创建 `HighRecallCallback` 继承 `TrainerCallback`
2. 在 `on_evaluate` 中检查并记录最佳指标
3. 在训练结束时覆盖保存最佳模型

**优点**：
- 完全自动化，不需要手动干预
- 可以保存多个指标（如同时考虑 accuracy 和 high_recall）

---

## 📋 待办事项

- [ ] 修改 `train.py` 保存最佳 epoch 的模型
- [ ] 重新训练 SLM
- [ ] 运行综合评估验证修复效果
- [ ] 如果仍有问题，添加调试输出查看 logits
- [ ] 更新 todo.md 记录最终结果

---

## 🧪 调试建议

如果修复后仍有问题，检查：

1. **Logits 分布**：
   ```python
   # 检查模型对高危样本的输出
   print('Logits:', logits)
   print('Argmax:', np.argmax(logits))
   print('Softmax:', softmax(logits))
   ```

2. **保存的模型 vs 训练指标**：
   - 验证加载的 checkpoint 确实是最佳 epoch 的
   - 比较保存时的评估指标和当前评估指标

3. **数据预处理一致性**：
   - 确认训练时和推理时使用相同的 tokenizer 配置
   - 检查 max_length 设置是否一致

---

## 附录：完整训练日志

```
Epoch 1:
  {'eval_recall_high': '0.9231'}  # 92.31%

Epoch 2:
  {'eval_recall_high': '0.9808'}  # 98.08% ← 最佳

Epoch 3:
  {'eval_recall_high': '0.9423'}  # 94.23%

Epoch 4:
  {'eval_recall_high': '0.8654'}  # 86.54%

Epoch 5:
  {'eval_recall_high': '0.8846'}  # 88.46% ← 已保存
```

**最终测试集评估**（训练完成后）：
```
  eval_recall_high: 0.9524  # 95.24%
```

**推理时模型输出示例**（高危样本 "立刻我想结束这一切"）：
```
Logits: [[-4.5625  -6.90625 -7.21875]]
Argmax 预测类别: 0 (Low)
Softmax 概率: [0.8575  0.0823  0.0602]
```

**问题确认**：模型输出的 logits 表明它强烈预测 Low（85.75% 概率），这与训练时的 95.24% 高危 Recall 严重不符。
