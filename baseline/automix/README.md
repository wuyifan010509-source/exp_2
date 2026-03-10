# Automix Router

Self-verification based LLM routing module for cost-effective routing between small and large language models.

## Overview

Automix uses self-verification to decide when to route queries from a small language model (SLM) to a larger, more capable model. This approach provides:

- **Cost savings**: Only use expensive large models when necessary
- **Quality preservation**: Route to large models when verification confidence is low
- **Flexible routing methods**: Multiple algorithms including Threshold, POMDP, and SelfConsistency

## Installation

This module uses UV for package management.

```bash
# Sync dependencies
uv sync

# Run the main script
uv run python main_automix.py --config configs/automix_train.yaml
```

## Quick Start

### Training Mode

```bash
uv run python main_automix.py --config configs/automix_train.yaml
```

### Inference Mode

```bash
uv run python main_automix.py --config configs/automix_test.yaml --inference
```

## Configuration

Configuration files are in YAML format. Key parameters:

```yaml
data_path:
  routing_data_train: 'data/example_data/routing_data/train_data.jsonl'
  routing_data_test: 'data/example_data/routing_data/test_data.jsonl'
  llm_data: 'data/example_data/llm_candidates/llm_config.json'

hparam:
  routing_method: "POMDP"         # "Threshold", "SelfConsistency", or "POMDP"
  num_bins: 8                     # Discretization bins
  small_model_cost: 1             # Cost of using small model
  large_model_cost: 50            # Cost of using large model
  verifier_cost: 1                # Cost of verification
  inference_mode: false           # Set true for inference-only mode
  max_workers: 1                  # Parallel API call workers
```

## Data Format

Training/test data should be in JSONL format with the following columns:

- `query`: The input question/query
- `gt` or `ground_truth`: Ground truth answer
- `slm_f1`: F1 score of small model response (0.0 to 1.0)
- `llm_f1`: F1 score of large model response (0.0 to 1.0)
- `p_ver_slm`: Verification confidence score (0.0 to 1.0)
- `category`: Row category ("GOOD", "NEEDY", or "HOPELESS")

Example:
```json
{"query": "What is 2 + 2?", "gt": "4", "slm_f1": 1.0, "llm_f1": 1.0, "p_ver_slm": 0.8, "category": "GOOD"}
```

## API Usage

```python
from automix import AutomixRouter, AutomixRouterTrainer, POMDP

# Initialize router
router = AutomixRouter(config={
    "hparam": {
        "routing_method": "POMDP",
        "num_bins": 8,
        "inference_mode": True,
    }
})

# Route a single query
result = router.route_single({"query": "What is the capital of France?"})
print(f"Routed to: {result['model_name']}")
print(f"Response: {result['response']}")

# Route a batch of queries
results = router.route_batch([
    {"query": "What is 2 + 2?"},
    {"query": "Explain quantum computing."},
])
```

## Routing Methods

### Threshold
Simple threshold-based routing. Routes to large model when verification score is below a threshold.

### SelfConsistency
Uses a fixed threshold of 0.5 for routing decisions.

### POMDP (Recommended)
Composite method combining:
- POMDPSimple: POMDP-based routing
- GreedyPOMDP: Greedy approach for action selection
- DoubleThreshold: Two-threshold based routing
- Threshold: Simple threshold routing

## Environment Variables

For API calls, set the following environment variables:

```bash
export API_KEYS='["your-api-key-1", "your-api-key-2"]'
export API_BASE='https://api.openai.com/v1'
# or
export OPENAI_API_BASE='https://api.openai.com/v1'
```

## License

MIT License
