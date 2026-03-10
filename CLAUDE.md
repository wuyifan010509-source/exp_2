# AGENTS.md - 项目指南

## ⚠️ 关键提醒


### UV 项目管理

**添加依赖（不要使用 pip）：**
```bash
uv add <package>                    # 生产依赖
uv add --dev <package>              # 开发依赖
```

**运行脚本（必须使用 uv run）：**
```bash
uv run python script.py
uv run python -m module_name
uv run llamafactory-cli train config.yaml
```

**常见操作：**
```bash
uv sync                             # 同步依赖
uv run python -m src.synthetic_data # 数据生成
uv run python scripts/convert_to_user_simulator.py  # 数据转换
uv run python scripts/train_and_monitor.py attention_only  # 训练
```

### 项目目录结构

```
├── data/                        # 数据文件
│   ├── checkpoints/            # 数据生成检查点
│   ├── processed/              # 处理后的数据 (final_labeled_data.jsonl)
│   ├── raw/                    # 原始数据 (oos_*.csv)
│   └── validation/             # 验证集 (validation_set_*.jsonl)
├── exp/                         # 实验代码
│   ├── data_labeling/          # 数据标注模块
│   ├── evaluation/             # 评估模块
│   ├── semantic_routing/       # 语义路由模块
│   ├── slm_distillation/       # SLM蒸馏模块
│   └── reports/                # 实验报告
├── logs/                        # 日志文件
├── scripts/                     # 独立脚本工具
└── output/                      # 模型输出
```

**完整目录说明见:** `PROJECT_STRUCTURE.md`

### 运行命令
```bash
# 允许操作，自动化执行
claude --permission-mode bypassPermissions
```
