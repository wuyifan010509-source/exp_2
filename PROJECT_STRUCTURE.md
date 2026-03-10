# 项目目录结构说明

本文档说明项目整理后的目录结构。

## 根目录结构

```
├── CLAUDE.md                    # AI助手指南
├── README.md                    # 项目说明
├── data/                        # 数据文件（整理后）
│   ├── checkpoints/            # 数据生成检查点
│   ├── processed/              # 处理后的数据
│   ├── raw/                    # 原始数据
│   └── validation/             # 验证集
├── docs/                        # 文档（待创建）
├── exp/                         # 实验代码
│   ├── data_labeling/          # 数据标注模块
│   ├── evaluation/             # 评估模块
│   ├── output/                 # 模型输出
│   ├── reports/                # 实验报告
│   ├── semantic_routing/       # 语义路由模块
│   └── slm_distillation/       # SLM蒸馏模块
├── logs/                        # 日志文件（整理后）
├── output/                      # 模型输出（主目录）
├── scripts/                     # 脚本工具（整理后）
└── baseline/                    # 基线方法
```

## 详细说明

### data/ - 数据目录

**checkpoints/**: 数据生成过程中的中间文件
- `expanded_*.jsonl` - 各类别扩充数据

**processed/**: 处理后的训练数据
- `final_labeled_data.jsonl` - 最终标注数据（原 data_labeling/labeled_data.jsonl）
- `training_data.jsonl` - 训练数据
- `expanded_all_ready.jsonl` - 待标注数据

**raw/**: 原始数据文件
- `oos_*.csv` - OOS原始数据

**validation/**: 验证集
- `validation_set_*.jsonl` - 各类验证集

### exp/ - 实验代码

**data_labeling/**: 数据标注模块
- `config.py` - 标注配置
- `expand_data.py` - 数据扩充
- `label_data.py` - 数据标注
- `labeling_guide.md` - 标注指南
- `checkpoints/` - 标注检查点

**evaluation/**: 评估模块
- `evaluate.py` - 评估脚本

**semantic_routing/**: 语义路由模块
- `router.py` - 路由器
- `anchors.py` - 锚点定义
- `test_routing.py` - 测试脚本
- `test_full_dataset.py` - 完整数据集测试

**slm_distillation/**: SLM蒸馏模块
- `train.py` - 训练脚本
- `inference.py` - 推理脚本
- `config.py` - 训练配置
- `asymmetric_loss.py` - 非对称损失

**reports/**: 实验报告
- `EVALUATION_REPORT.md` - 评估报告
- `semantic_routing_test_report.md` - 语义路由测试报告

### logs/ - 日志目录（新创建）

- `training_log*.txt` - 训练日志
- `inference_*.txt` - 推理日志
- `c1_audit_details.txt` - C1类别审计
- `qa_audit_report_C1.txt` - QA审计报告
- `slm_training_summary.md` - 训练摘要

### scripts/ - 脚本目录（新创建）

数据生成和处理的独立脚本：
- `generate_validation_set.py` - 验证集生成（模板版）
- `generate_validation_set_llm.py` - 验证集生成（LLM版）
- `generate_expanded_data.py` - 数据扩充
- `generate_oos_data.py` - OOS数据生成
- `regenerate_training_data.py` - 重新生成训练数据
- `fix_class_labels.py` - 修复类别标签
- `fix_labels.py` - 修复标签
- `expand_data.py` - 数据扩充脚本
- `gap_filler.py` - 数据缺口填充
- `generate_critical_data.py` - 关键数据生成
- `generate_a2_data.py` - A2类别数据生成
- `merge_final_data.py` - 合并最终数据
- `offline_expand_data.py` - 离线数据扩充
- `offline_generate_C1.py` - 离线生成C1数据
- `analyze_confusion.py` - 混淆矩阵分析
- `compare_methods.py` - 方法对比
- `convert_validation_set.py` - 验证集转换
- `data_config.py` - 数据配置（原 data/data_labeling/config.py）
- `data_config_old.py` - 旧数据配置
- `data_fix_labels.py` - 数据标签修复

## 常用命令更新

### 运行语义路由测试
```bash
uv run python -m exp.semantic_routing.test_full_dataset \
  --file data/validation/validation_set_llm_1000.jsonl
```

### 运行评估
```bash
uv run python -m exp.evaluation.evaluate \
  --methods routing \
  --test_file data/validation/validation_set_llm_1000.jsonl
```

### 使用训练数据
```bash
# 原路径：data_labeling/labeled_data.jsonl
# 新路径：data/processed/final_labeled_data.jsonl
uv run python -m exp.slm_distillation.inference \
  --test_file data/processed/final_labeled_data.jsonl
```

## 注意事项

1. **原 `data_labeling/` 目录已移除**，相关脚本移至 `scripts/`
2. **原 `exp/data/data_labeling/` 目录已移除**，相关脚本移至 `scripts/`
3. **日志文件统一放在 `logs/` 目录**，不再散落在各处
4. **验证集统一放在 `data/validation/` 目录**
5. **检查点文件统一放在 `data/checkpoints/` 目录**

## 路径变更对照表

| 原路径 | 新路径 | 说明 |
|--------|--------|------|
| `data_labeling/labeled_data.jsonl` | `data/processed/final_labeled_data.jsonl` | 标注数据 |
| `data_labeling/checkpoints/` | `data/checkpoints/` | 检查点 |
| `exp/data/data_labeling/` | `scripts/` | 数据脚本 |
| `exp/data/validation_set*.jsonl` | `data/validation/` | 验证集 |
| `exp/data/oos*.csv` | `data/raw/` | 原始数据 |
| `exp/training_log*.txt` | `logs/` | 训练日志 |
| `exp/inference_*.txt` | `logs/` | 推理日志 |
| `exp/analyze_confusion.py` | `scripts/` | 分析脚本 |
| `exp/compare_methods.py` | `scripts/` | 对比脚本 |
| `root/*.py` | `scripts/` | 根目录脚本 |

---

最后更新: 2025-03-06
