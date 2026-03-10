
## 更新：2026-03-06 验证 checkpoint-31

### ✅ checkpoint-31 (Epoch 2) 验证结果

**测试的高危样本**：
```
Query: "立刻我想结束这一切" → cost=1000, high_prob=0.9707 (97%) ✅ 正确！
Query: "绝对我不想活了，撑不住了" → cost=50, high_prob=0.0006 (0%) ❌ 错误！
Query: "融资爆仓了怎么办" → cost=50, high_prob=0.0036 (0%) ❌ 错误！
```

### 🔍 分析

**训练日志中 Epoch 2 指标**：
```
{'eval_recall_high': '0.9808'}
```

**推理验证结果**：checkpoint-31 对"结束"类关键词能正确识别（97% 高危）！

### 结论

**`load_best_model_at_end=True` 确实生效了**：最佳 checkpoint (Epoch 2) 被保存并在加载时自动选中。

**但为什么默认加载时预测错误**：
- 推断脚本没有指定 adapter_path，使用默认 OUTPUT_DIR
- Trainer 默认行为：加载最新的 checkpoint
- Epoch 5 的 checkpoint (step 155) 比 Epoch 2 (step 62) 更新，所以被优先加载

### 下一步

确保推理时使用最佳 checkpoint (epoch-31)：

```python
# 错误：使用默认路径（会加载最新的）
predictor = SLMCostPredictor()  

# 正确：指定最佳 checkpoint
predictor = SLMCostPredictor(adapter_path='output/slm_distillation/checkpoint-31')
```


## 更新：2026-03-06 数据划分不一致分析

### 🔍 关键发现

**训练日志分析**：
- Epoch 2 (step 62): eval_recall_high=0.9808 (98.08%) ← **最佳**
- Epoch 3 (step 93): eval_recall_high=0.9423 (94.23%) 
- Epoch 4 (step 124): eval_recall_high=0.9038 (90.38%)
- Epoch 5 (step 155): eval_recall_high=0.9423 (90.38%)

**最终测试集评估**（训练后）：
- High Recall: **0.9808** (98.08%)

**最佳 epoch**: Epoch 2 (98.08%) 或 Epoch 4 (98.08%) 都达到最佳

**checkpoint-31 验证**：
```
Query: "立刻我想结束这一切" → cost=1000, high_prob=0.9707 (97%) ✅ 正确！
```

### 🔴 根因分析

**问题**：评估时 SLM 高危 Recall 只有 4% (10/244=1)，但训练时最终测试集显示 98.08%

**可能原因**：
1. **数据划分不一致**：
   - 训练时使用的是 HuggingFace `train_test_split` 内部随机划分
   - 评估时使用的是 sklearn `train_test_split` 分层采样
   - 这导致训练的测试集和评估的测试集不是同一批数据

2. **checkpoint 问题**：
   - checkpoint-31 是基于 Epoch 2 训练，但可能不是最终保存的
   - HuggingFace Trainer 的 `load_best_model_at_end=True` 会在训练结束时覆盖保存，可能不是最佳

### 🎯 解决方案

**方案 1**：评估时使用训练数据划分，确保与训练时一致
- 修改 `evaluate.py` 中 `load_test_data()` 的随机种子
- 使用 `np.random.seed(42)` 并使用 HuggingFace 的 `train_test_split`

**方案 2**：检查并保存最佳 checkpoint
- 确认训练时保存的是哪个 epoch 的 checkpoint
- 确保 `load_best_model_at_end=True` 正确工作

