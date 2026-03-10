"""
Automix Router Trainer
----------------------
Training implementation for AutomixRouter.

Original source: automix/colabs/Step3_MetaVerify.py
Adapted for standalone Automix module.
"""

import torch
import pandas as pd
from typing import Any, Dict, Optional

from .model import AutomixModel


class AutomixRouterTrainer:
    """
    AutomixRouterTrainer
    -------------------
    Trainer implementation for AutomixRouter.

    Unlike typical neural network training with gradient descent,
    Automix training involves:
    1. Searching over candidate routing parameters
    2. Evaluating each on the training data
    3. Selecting the parameter with best IBC (Incremental Benefit over Cost) lift
    """

    def __init__(
        self,
        model: AutomixModel,
        train_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
        device: str = "cpu",
        cost_constraint: Optional[tuple] = None,
        verbose: bool = False,
    ):
        """
        Initialize AutomixRouterTrainer.

        Args:
            model: An AutomixModel instance
            train_df: Training dataframe
            test_df: Test dataframe
            device (str): Device for computation (default: "cpu")
            cost_constraint: Optional tuple of (min_cost, max_cost)
            verbose: Print debug information
        """
        self.model = model
        self.device = device
        self.cost_constraint = cost_constraint
        self.verbose = verbose
        self.train_df = train_df
        self.test_df = test_df

    def train(self) -> Dict[str, Any]:
        """
        Train the AutomixRouter.

        For Automix, "training" means:
        1. Search over candidate parameters on training data
        2. Select best parameter based on IBC lift
        3. Evaluate on test data

        Returns:
            Dictionary containing training and test metrics
        """
        if self.train_df is None or len(self.train_df) == 0:
            raise ValueError("Training data is required")

        # Perform parameter search on training data
        best_param = self.model.train_routing(
            self.train_df,
            cost_constraint=self.cost_constraint
        )

        if self.verbose:
            print(f"Best parameter found: {best_param}")

        # Evaluate on training data
        train_metrics = self.model.evaluate(self.train_df, return_dict=True)

        if self.verbose:
            print(f"Training metrics: {train_metrics}")

        results = {
            "train": {
                "best_param": best_param,
                "metrics": train_metrics,
            },
        }

        # Evaluate on test data if available
        if self.test_df is not None and len(self.test_df) > 0:
            test_metrics = self.model.evaluate(
                self.test_df,
                return_dict=True,
                return_decisions=True
            )

            if self.verbose:
                print(f"Test metrics: {test_metrics}")

            results["test"] = test_metrics

        return results

    def train_and_evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Train and evaluate with provided dataframes.

        Args:
            train_df: Training dataframe
            test_df: Test dataframe

        Returns:
            Dictionary containing training and test metrics
        """
        self.train_df = train_df
        self.test_df = test_df
        return self.train()
