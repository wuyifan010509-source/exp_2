#!/usr/bin/env python3
"""
Automix Router - Complete Usage Example
========================================

This script demonstrates how to use the Automix router for complete training and inference workflows.

Usage:
    uv run python main_automix.py --config configs/automix_train.yaml
    uv run python main_automix.py --config configs/automix_test.yaml --inference
"""

import os
import sys
import argparse
import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from automix import (
    AutomixRouter,
    AutomixRouterTrainer,
    AutomixModel,
    POMDP,
    Threshold,
    SelfConsistency,
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_routing_method(method_name: str, num_bins: int):
    """Create routing method instance based on method name."""
    method_map = {
        "Threshold": Threshold,
        "SelfConsistency": SelfConsistency,
        "POMDP": POMDP,
    }

    if method_name not in method_map:
        raise ValueError(
            f"Unknown routing method: {method_name}. "
            f"Available methods: {list(method_map.keys())}"
        )

    return method_map[method_name](num_bins=num_bins)


def train_and_evaluate(config: dict):
    """Train and evaluate using configuration."""
    hparam = config.get("hparam", {})
    data_cfg = config.get("data_path", {})

    # Check for required data
    train_path = data_cfg.get("routing_data_train")
    test_path = data_cfg.get("routing_data_test")

    if not train_path or not test_path:
        print("Error: Config must specify routing_data_train and routing_data_test")
        return None

    # Resolve paths
    project_root = os.path.dirname(__file__)
    if not os.path.isabs(train_path):
        train_path = os.path.join(project_root, train_path)
    if not os.path.isabs(test_path):
        test_path = os.path.join(project_root, test_path)

    if not os.path.exists(train_path):
        print(f"Error: Training data not found: {train_path}")
        return None
    if not os.path.exists(test_path):
        print(f"Error: Test data not found: {test_path}")
        return None

    # Load data
    print(f"Loading training data from: {train_path}")
    print(f"Loading test data from: {test_path}")

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    # Check if data has required columns
    required_cols = ["slm_f1", "llm_f1", "p_ver_slm"]
    has_cols = all(col in train_df.columns for col in required_cols)

    if not has_cols:
        print("\n⚠️ Data does not have required columns (slm_f1, llm_f1, p_ver_slm)")
        print("Please run data preparation pipeline first or provide pre-processed data.")
        return None

    # Create routing method
    method = get_routing_method(
        hparam.get("routing_method", "POMDP"),
        hparam.get("num_bins", 8)
    )

    # Create model
    model = AutomixModel(
        method=method,
        slm_column=hparam.get("slm_column", "slm_f1"),
        llm_column=hparam.get("llm_column", "llm_f1"),
        verifier_column=hparam.get("verifier_column", "p_ver_slm"),
        costs=[hparam.get("small_model_cost", 1), hparam.get("large_model_cost", 50)],
        verifier_cost=hparam.get("verifier_cost", 1),
        verbose=hparam.get("verbose", True),
    )

    # Create trainer
    trainer = AutomixRouterTrainer(
        model=model,
        train_df=train_df,
        test_df=test_df,
        device=hparam.get("device", "cpu"),
        cost_constraint=hparam.get("cost_constraint"),
        verbose=hparam.get("verbose", True),
    )

    # Train and evaluate
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)

    results = trainer.train()

    # Print results
    print("\n" + "="*50)
    print("Results")
    print("="*50)

    if "train" in results:
        train_res = results["train"]
        print(f"\nBest parameter: {train_res['best_param']}")
        print(f"Training metrics: {train_res['metrics']}")

    if "test" in results:
        test_res = results["test"]
        print(f"\nTest metrics:")
        print(f"  IBC Lift: {test_res.get('ibc_lift', 'N/A')}")
        print(f"  Avg Performance: {test_res.get('avg_performance', 'N/A'):.4f}")
        print(f"  Avg Cost: {test_res.get('avg_cost', 'N/A'):.4f}")

        if "route_to_llm" in test_res:
            route_pct = test_res["route_to_llm"].mean() * 100
            print(f"  Routing to LLM: {route_pct:.1f}%")

    return results


def run_inference(config: dict, queries: list = None):
    """Run inference using trained router."""
    hparam = config.get("hparam", {})

    # Set inference mode
    hparam["inference_mode"] = True
    config["hparam"] = hparam

    # Create router
    print("Initializing router in inference mode...")
    router = AutomixRouter(config=config)

    # If no queries provided, use sample queries
    if queries is None:
        queries = [
            {"query": "What is 2 + 2?"},
            {"query": "What is the capital of France?"},
            {"query": "Explain quantum computing in simple terms."},
        ]

    print(f"\nProcessing {len(queries)} queries...")

    results = router.route_batch(queries)

    # Print results
    print("\n" + "="*50)
    print("Inference Results")
    print("="*50)

    for i, result in enumerate(results):
        print(f"\nQuery {i+1}: {result.get('query', 'N/A')[:50]}...")
        print(f"  Routed to: {result.get('model_name', 'N/A')}")
        print(f"  Route to LLM: {result.get('route_to_llm', 'N/A')}")
        print(f"  Verification score: {result.get('verification_score', 'N/A'):.2f}")
        print(f"  Response: {result.get('response', 'N/A')[:100]}...")

    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Automix Router Training and Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/automix_train.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run in inference mode instead of training",
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if args.inference:
        run_inference(config)
    else:
        train_and_evaluate(config)


if __name__ == "__main__":
    main()
