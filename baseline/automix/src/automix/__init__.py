"""
Automix - Self-verification based LLM routing

This module provides the Automix router for cost-effective routing
between small and large language models based on verification confidence.

Key components:
- AutomixRouter: Main router class
- AutomixRouterTrainer: Training class for parameter optimization
- AutomixModel: PyTorch model wrapper
- Routing methods: Threshold, POMDP, SelfConsistency
"""

from .router import AutomixRouter
from .trainer import AutomixRouterTrainer
from .model import AutomixModel
from .methods import POMDP, Threshold, SelfConsistency

__all__ = [
    "AutomixRouter",
    "AutomixRouterTrainer",
    "AutomixModel",
    "POMDP",
    "Threshold",
    "SelfConsistency",
]

__version__ = "0.1.0"
