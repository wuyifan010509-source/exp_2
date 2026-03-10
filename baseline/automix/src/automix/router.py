"""
Automix Router
--------------
Automix router that provides routing decisions based on self-verification.

This module provides the main AutomixRouter class for routing queries
between small and large language models based on verification confidence.

Original source: automix/colabs/
Adapted for standalone Automix module.
"""

import os
import re
import yaml
import json
import pandas as pd
import copy
from typing import Any, Dict, List, Tuple, Optional

from .model import AutomixModel
from .methods import Threshold, POMDP, SelfConsistency
from .data_pipeline import (
    init_providers,
    run_solver_job,
    prepare_row,
    run_verification,
    compute_fraction_correct,
    calculate_f1_for_models,
    categorize_rows,
)


def parse_size(size_str: str) -> float:
    """
    Parse a model size string (e.g., '7B', '13B', '512M') into
    a numeric value in billions.

    Supported suffixes:
        - K: thousands
        - M: millions
        - B: billions
        - T: trillions

    If parsing fails, this function returns 0.0.
    """
    size_str = str(size_str).strip().upper()
    try:
        if size_str.endswith("K"):
            return float(size_str[:-1]) / 1e6
        elif size_str.endswith("M"):
            return float(size_str[:-1]) / 1e3
        elif size_str.endswith("B"):
            return float(size_str[:-1])
        elif size_str.endswith("T"):
            return float(size_str[:-1]) * 1e3
        else:
            return float(size_str)
    except Exception:
        return 0.0


def detect_small_large_models(llm_data_path: str) -> Tuple[str, str]:
    """
    Automatically detect the smallest and largest models from llm_data file.

    Args:
        llm_data_path: Path to the LLM data JSON file

    Returns:
        Tuple of (smallest_model_engine, largest_model_engine)

    Raises:
        ValueError: If llm_data is empty or no valid models found
    """
    with open(llm_data_path, 'r') as f:
        llm_data = json.load(f)

    if not llm_data:
        raise ValueError("llm_data is empty, cannot detect models")

    # Parse sizes and find smallest and largest
    model_info = []
    for name, info in llm_data.items():
        if isinstance(info.get("size", ""), str) and info["size"].strip():
            size = parse_size(info["size"])
            if size > 0:
                engine = info.get("model", name)
                model_info.append((name, engine, size))

    if not model_info:
        raise ValueError("No models with valid size information found in llm_data")

    # Sort by size
    model_info.sort(key=lambda x: x[2])

    smallest_name, smallest_engine, smallest_size = model_info[0]
    largest_name, largest_engine, largest_size = model_info[-1]

    print(f"✅ Auto-detected small model: {smallest_name} ({smallest_size}B) -> {smallest_engine}")
    print(f"✅ Auto-detected large model: {largest_name} ({largest_size}B) -> {largest_engine}")

    return smallest_engine, largest_engine


class AutomixRouter:
    """
    AutomixRouter
    -------------
    Router that uses self-verification to decide when to route queries
    from a small language model to a larger, more capable model.

    Key features:
    - Cost-effective routing based on verification confidence
    - Multiple routing methods: Threshold, POMDP, SelfConsistency
    - Automatic data preparation and preprocessing
    """

    def __init__(self, yaml_path: str = None, config: dict = None):
        """
        Initialize AutomixRouter.

        Args:
            yaml_path (str): Path to YAML config file
            config (dict): Configuration dictionary (alternative to yaml_path)
        """
        # Load configuration
        if yaml_path is not None:
            with open(yaml_path, 'r') as f:
                self.cfg = yaml.safe_load(f)
        elif config is not None:
            self.cfg = config
        else:
            # Use default configuration
            self.cfg = self._default_config()

        # Resolve project root
        if yaml_path:
            self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(yaml_path)))
        else:
            self.project_root = os.getcwd()

        # Get hyperparameters
        hparam = self.cfg.get("hparam", {})

        # Check if inference mode (skip data preprocessing)
        inference_mode = hparam.get("inference_mode", False)

        # Get model engines
        self.engine_small = hparam.get("engine_small")
        self.engine_large = hparam.get("engine_large")

        # Auto-detect small and large models from llm_data if not specified
        data_cfg = self.cfg.get("data_path", {})
        llm_data_path = data_cfg.get("llm_data")

        if not self.engine_small or not self.engine_large:
            if llm_data_path:
                if not os.path.isabs(llm_data_path):
                    llm_data_path = os.path.join(self.project_root, llm_data_path)
                if os.path.exists(llm_data_path):
                    self.engine_small, self.engine_large = detect_small_large_models(llm_data_path)

        if not self.engine_small:
            self.engine_small = "gpt-3.5-turbo"
        if not self.engine_large:
            self.engine_large = "gpt-4"

        # Use generic column names for small/large models
        self.slm_column = hparam.get("slm_column", "slm_f1")
        self.llm_column = hparam.get("llm_column", "llm_f1")
        self.verifier_column = hparam.get("verifier_column", "p_ver_slm")

        if inference_mode:
            # Inference mode: skip data loading and preprocessing
            print("✅ Inference mode enabled - skipping data preprocessing")
            self.train_df = pd.DataFrame()
            self.test_df = pd.DataFrame()
        else:
            # Training mode: load and prepare data
            self.train_df, self.test_df = self._prepare_data()

        # Create routing method
        method = self._build_method(
            hparam.get("routing_method", "POMDP"),
            hparam.get("num_bins", 8)
        )

        # Extract costs
        small_model_cost = hparam.get("small_model_cost", 1)
        large_model_cost = hparam.get("large_model_cost", 50)
        verifier_cost = hparam.get("verifier_cost", 1)

        # Extract verbose setting
        verbose = hparam.get("verbose", False)

        # Create AutomixModel
        self.model = AutomixModel(
            method=method,
            slm_column=self.slm_column,
            llm_column=self.llm_column,
            verifier_column=self.verifier_column,
            costs=[small_model_cost, large_model_cost],
            verifier_cost=verifier_cost,
            verbose=verbose,
        )

    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            "hparam": {
                "routing_method": "POMDP",
                "num_bins": 8,
                "small_model_cost": 1,
                "large_model_cost": 50,
                "verifier_cost": 1,
                "device": "cpu",
                "verbose": True,
                "inference_mode": True,
            }
        }

    def _prepare_data(self):
        """
        Prepare and preprocess data using intermediate variables.

        Returns:
            tuple: (train_df, test_df) - Training and test DataFrames with all required columns
        """
        data_cfg = self.cfg.get("data_path", {})
        hparam = self.cfg.get("hparam", {})

        # Get paths for routing data
        train_path = data_cfg.get("routing_data_train")
        test_path = data_cfg.get("routing_data_test")

        if not train_path or not test_path:
            raise ValueError("Config must specify 'routing_data_train' and 'routing_data_test' in data_path section")

        # Resolve to absolute paths
        if not os.path.isabs(train_path):
            train_path = os.path.join(self.project_root, train_path)
        if not os.path.isabs(test_path):
            test_path = os.path.join(self.project_root, test_path)

        # Check if files exist
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data not found: {test_path}")

        # Load data into memory
        train_df = pd.read_json(train_path, lines=True, orient="records")
        test_df = pd.read_json(test_path, lines=True, orient="records")

        # Add split column if not present
        if "split" not in train_df.columns:
            train_df["split"] = "train"
        if "split" not in test_df.columns:
            test_df["split"] = "test"

        # Determine ground truth column
        ground_truth_col = None
        for col in ["gt", "ground_truth", "answer", "output"]:
            if col in train_df.columns and col in test_df.columns:
                ground_truth_col = col
                break

        # Check if data already has required columns
        required_cols = [self.slm_column, self.llm_column, self.verifier_column]
        train_has_cols = all(col in train_df.columns for col in required_cols)
        test_has_cols = all(col in test_df.columns for col in required_cols)

        if train_has_cols and test_has_cols:
            return train_df, test_df

        # If not, need to generate predictions and verification scores
        print("⚠️ Data does not have required columns. Running data preparation pipeline...")

        # Merge train and test into a single DataFrame for processing
        merged_df = pd.concat([test_df, train_df], ignore_index=True)

        # Initialize API providers
        init_providers()

        # Get max_workers from config
        max_workers = hparam.get("max_workers", 1)

        # Step 1: Solve queries with small and large model
        results_slm = run_solver_job(merged_df, prepare_row, self.engine_small, max_workers=max_workers)
        results_llm = run_solver_job(merged_df, prepare_row, self.engine_large, max_workers=max_workers)

        merged_df["slm_pred_ans"] = results_slm
        merged_df["llm_pred_ans"] = results_llm

        # Calculate F1 scores
        if ground_truth_col:
            merged_df = calculate_f1_for_models(merged_df, ["slm", "llm"], ground_truth_col=ground_truth_col)
        else:
            # Default F1 to 0 if no ground truth
            merged_df[self.slm_column] = 0.0
            merged_df[self.llm_column] = 0.0

        # Step 2: Self-verification
        ver_results = run_verification(
            merged_df,
            ans_col="slm_pred_ans",
            engine_name=self.engine_small,
            temperature=1.0,
            n=2,
            stop="---",
            max_tokens=250,
            max_workers=max_workers,
        )

        merged_df["slm_ver"] = ver_results
        merged_df[self.verifier_column] = merged_df["slm_ver"].apply(compute_fraction_correct)

        # Categorize rows
        merged_df = categorize_rows(merged_df, slm_column=self.slm_column, llm_column=self.llm_column)

        # Split back into train and test
        train_df = merged_df[merged_df["split"] == "train"].copy().reset_index(drop=True)
        test_df = merged_df[merged_df["split"] == "test"].copy().reset_index(drop=True)

        # Verify required columns are present
        for col in required_cols:
            if col not in train_df.columns:
                raise ValueError(f"Training data missing required column after processing: {col}")
            if col not in test_df.columns:
                raise ValueError(f"Test data missing required column after processing: {col}")

        return train_df, test_df

    @staticmethod
    def _build_method(name: str, num_bins: int):
        """Build routing method from name."""
        mapping = {
            "POMDP": POMDP,
            "Threshold": Threshold,
            "SelfConsistency": SelfConsistency,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported routing method: {name}. Available: {list(mapping.keys())}")
        return mapping[name](num_bins=num_bins)

    def route_batch(self, batch: Optional[Any] = None, task_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Route a batch of queries with full Automix inference pipeline.

        This method performs end-to-end processing:
        1. Calls small model for initial answer
        2. Performs self-verification
        3. Routes to large model if verification confidence is low
        4. Returns final answers and routing decisions

        Args:
            batch (Any, optional): Batch of queries to route
            task_name (str, optional): Task name for prompt formatting

        Returns:
            list of dict: Routing results with model_name, response, and routing decision
        """
        # Determine which data to use
        if batch is not None:
            query_data = batch if isinstance(batch, list) else [batch]
        else:
            if hasattr(self, "test_df") and self.test_df is not None and len(self.test_df) > 0:
                query_data = self.test_df.to_dict('records')
            else:
                print("Warning: No batch provided and no test data available.")
                return []

        query_data_output = []
        max_workers = self.cfg.get("hparam", {}).get("max_workers", 1)

        for row in query_data:
            # Handle both dict and non-dict inputs
            if isinstance(row, dict):
                row_copy = copy.copy(row)
                original_query = row_copy.get("query", "")
            else:
                row_copy = {"query": str(row)}
                original_query = str(row)

            # Step 1: Call small model
            query_df = pd.DataFrame([{'query': original_query}])
            results_slm = run_solver_job(
                query_df, prepare_row, self.engine_small, max_workers=max_workers
            )
            slm_answer = results_slm[0] if results_slm else ""
            query_df['slm_pred_ans'] = slm_answer

            # Step 2: Self-verification
            ver_results = run_verification(
                query_df,
                ans_col='slm_pred_ans',
                engine_name=self.engine_small,
                temperature=1.0,
                n=2,
                stop="---",
                max_tokens=250,
                max_workers=max_workers,
            )
            verification_score = compute_fraction_correct(ver_results[0]) if ver_results else 0.0

            # Step 3: Make routing decision
            decision_df = pd.DataFrame([{self.verifier_column: verification_score}])
            decision_df[self.slm_column] = 0.0
            decision_df[self.llm_column] = 0.0
            batch_dict = {"data": decision_df, "mode": "infer"}
            outputs = self.model(batch_dict)
            route_to_llm = bool(outputs["decisions"].item())

            # Step 4: Route to large model if needed
            if route_to_llm:
                results_llm = run_solver_job(query_df, prepare_row, self.engine_large, max_workers=max_workers)
                response = results_llm[0] if results_llm else slm_answer
                model_name = self.engine_large
            else:
                response = slm_answer
                model_name = self.engine_small

            row_copy["model_name"] = model_name
            row_copy["response"] = response
            row_copy["verification_score"] = verification_score
            row_copy["route_to_llm"] = route_to_llm
            row_copy["slm_answer"] = slm_answer
            row_copy["success"] = bool(response)

            query_data_output.append(row_copy)

        return query_data_output

    def route_single(self, sample: Any) -> Dict[str, Any]:
        """
        Route a single query with full Automix inference pipeline.

        Args:
            sample: Single sample (dict with 'query' field)

        Returns:
            dict: Routing result with model_name, response, and routing decision
        """
        # Extract query
        if isinstance(sample, dict) and 'query' in sample:
            query = sample['query']
        else:
            raise ValueError("sample must be a dict with 'query' field")

        # Use route_batch for single query
        results = self.route_batch([sample])
        return results[0] if results else None

    def route(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for route_batch for backward compatibility."""
        return self.route_batch(batch)
