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