"""
非对称损失函数 (Asymmetric Loss)

核心创新点：对漏报（把高危预测为低危）施加极高惩罚 α，
实现 "宁可误报，绝不漏报" 的金融保守性。

公式：
    L = (1/N) * Σ [ α * max(0, y_i - ŷ_i)² + max(0, ŷ_i - y_i)² ]

其中：
    - y_i: 真实标签
    - ŷ_i: 模型预测
    - α: 漏报惩罚系数（默认 10）
    - 漏报 (ŷ < y): 权重 α
    - 误报 (ŷ ≥ y): 权重 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AsymmetricRegressionLoss(nn.Module):
    """
    非对称回归损失函数。
    用于回归任务（连续 Cost 值 1-1000）。
    """

    def __init__(self, alpha: float = 10.0):
        """
        Args:
            alpha: 漏报惩罚系数。α > 1 表示更严厉地惩罚漏报。
        """
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算非对称回归损失。

        Args:
            predictions: 模型预测值 (batch_size,)
            targets: 真实标签值 (batch_size,)

        Returns:
            标量损失值
        """
        # 漏报部分：ŷ < y → y - ŷ > 0
        underpredict = torch.clamp(targets - predictions, min=0)
        # 误报部分：ŷ ≥ y → ŷ - y > 0
        overpredict = torch.clamp(predictions - targets, min=0)

        # 非对称加权
        loss = self.alpha * underpredict.pow(2) + overpredict.pow(2)

        return loss.mean()


class AsymmetricClassificationLoss(nn.Module):
    """
    非对称分类损失函数。
    用于分类任务（类别 0=Low, 1=Mid, 2=High）。

    实现思路：
    - 在标准 CrossEntropy 基础上，对不同错误类型施加不同权重。
    - 漏报（预测类别 < 真实类别，尤其是将 High 预测为 Low/Mid）受更大惩罚。
    - 通过构造样本级 (sample-level) 的动态权重实现。
    """

    def __init__(
        self,
        alpha: float = 10.0,
        num_classes: int = 3,
    ):
        """
        Args:
            alpha: 漏报惩罚系数
            num_classes: 类别数量
        """
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes

        # 构造代价矩阵 (cost_matrix[true_class, pred_class])
        # - 对角线为 0（正确分类无惩罚）
        # - 漏报（pred < true）权重为 α
        # - 误报（pred > true）权重为 1
        cost_matrix = torch.ones(num_classes, num_classes)
        for i in range(num_classes):
            cost_matrix[i, i] = 0.0  # 正确分类
            for j in range(i):
                # pred_class j < true_class i → 漏报
                cost_matrix[i, j] = alpha
        self.register_buffer("cost_matrix", cost_matrix)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算非对称分类损失。

        Args:
            logits: 模型输出 logits (batch_size, num_classes)
            targets: 真实类别标签 (batch_size,) long

        Returns:
            标量损失值
        """
        # 标准 CrossEntropy（逐样本，不 reduce）
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # 计算每个样本的预测类别
        pred_classes = logits.argmax(dim=-1)

        # 确保 cost_matrix 在正确的设备上
        cost_matrix = self.cost_matrix.to(targets.device)

        # 查表获取代价权重
        weights = cost_matrix[targets, pred_classes]

        # 保证正确分类的样本也有基础梯度（最低权重为 1.0）
        weights = torch.clamp(weights, min=1.0)

        # 加权损失
        weighted_loss = ce_loss * weights

        return weighted_loss.mean()


def get_asymmetric_loss(
    task_type: str = "classification",
    alpha: float = 10.0,
    num_classes: int = 3,
) -> nn.Module:
    """
    工厂函数：根据任务类型返回对应的非对称损失函数。

    Args:
        task_type: "regression" 或 "classification"
        alpha: 漏报惩罚系数
        num_classes: 类别数（仅分类有效）

    Returns:
        损失函数实例
    """
    if task_type == "regression":
        return AsymmetricRegressionLoss(alpha=alpha)
    elif task_type == "classification":
        return AsymmetricClassificationLoss(alpha=alpha, num_classes=num_classes)
    else:
        raise ValueError(f"不支持的任务类型: {task_type}，请使用 'regression' 或 'classification'")
