"""
自定义 Trainer
重写 compute_loss 方法，注入非对称损失函数。
"""

import torch
from transformers import Trainer
from typing import Optional, Dict, Union
from torch import nn

from .asymmetric_loss import get_asymmetric_loss
from .config import (
    ALPHA_UNDERPREDICT,
    ALPHA_OVERPREDICT,
    TASK_TYPE,
    NUM_CLASSES,
)


class AsymmetricCostTrainer(Trainer):
    """
    继承 HuggingFace Trainer，重写 compute_loss 使用非对称损失函数。

    关键行为：
    - 分类任务：使用 AsymmetricClassificationLoss（带代价矩阵的加权 CE）
    - 回归任务：使用 AsymmetricRegressionLoss（非对称 MSE）
    - 漏报（把高危预测为低危）惩罚权重 = α = 10
    - 误报（把低危预测为高危）惩罚权重 = 1
    """

    def __init__(self, *args, task_type: str = TASK_TYPE, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = task_type

        # 初始化非对称损失函数
        self.asymmetric_loss_fn = get_asymmetric_loss(
            task_type=task_type,
            alpha=ALPHA_UNDERPREDICT,
            num_classes=NUM_CLASSES,
        )
        print(f"[AsymmetricCostTrainer] 任务类型: {task_type}")
        print(f"[AsymmetricCostTrainer] 漏报惩罚 α = {ALPHA_UNDERPREDICT}")

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, any]],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        重写 compute_loss，使用非对称损失函数替代标准损失。

        Args:
            model: 模型
            inputs: 输入字典，包含 input_ids, attention_mask, labels
            return_outputs: 是否返回模型输出

        Returns:
            loss 或 (loss, outputs)
        """
        labels = inputs.pop("labels")

        # 前向传播
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if logits is None:
            raise ValueError("模型输出中未找到 logits，请检查模型头部配置")

        # 计算非对称损失
        if self.task_type == "classification":
            loss = self.asymmetric_loss_fn(logits, labels)
        elif self.task_type == "regression":
            # 回归任务：logits 通常为 (batch, 1)，squeeze 后与 labels 匹配
            predictions = logits.squeeze(-1)
            loss = self.asymmetric_loss_fn(predictions, labels.float())
        else:
            raise ValueError(f"不支持的任务类型: {self.task_type}")

        return (loss, outputs) if return_outputs else loss
