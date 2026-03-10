"""
Automix Data Pipeline
---------------------
Data processing utilities for Automix router.

This module provides functions for:
- Initializing API providers
- Running solver jobs with LLMs
- Running self-verification
- Computing verification scores
- F1 score calculation
- Data categorization

Original source: automix/colabs/Step3_MetaVerify.py
Adapted for standalone Automix module.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

# Try to import litellm for API calls
try:
    from litellm import completion
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False
    completion = None


# Global API configuration
_api_keys: List[str] = []
_api_base: str = ""
_initialized = False


def init_providers():
    """Initialize API providers from environment variables."""
    global _api_keys, _api_base, _initialized

    if _initialized:
        return

    # Get API keys
    api_keys_env = os.environ.get('API_KEYS', '')
    if api_keys_env:
        try:
            parsed = json.loads(api_keys_env)
            if isinstance(parsed, list):
                _api_keys = [str(key) for key in parsed if key]
            elif isinstance(parsed, str):
                _api_keys = [parsed] if parsed else []
        except (json.JSONDecodeError, TypeError):
            if api_keys_env.strip():
                _api_keys = [api_keys_env.strip()]

    # Get API base URL
    _api_base = (
        os.environ.get('OPENAI_API_BASE')
        or os.environ.get('NVIDIA_API_BASE')
        or os.environ.get('API_BASE', '')
    ).strip()

    if not _api_keys:
        print("Warning: API_KEYS environment variable is not set")
    if not _api_base:
        print("Warning: API_BASE/OPENAI_API_BASE environment variable is not set")

    _initialized = True


def run_solver_job(
    df: pd.DataFrame,
    prepare_func,
    engine_name: str,
    max_workers: int = 5,
) -> List[str]:
    """
    Run solver job in parallel.

    Args:
        df: Input dataframe
        prepare_func: Function to prepare each row
        engine_name: LLM engine name
        max_workers: Number of parallel workers

    Returns:
        List of model responses
    """
    init_providers()

    results = [None] * len(df)

    def process_row(idx, row):
        try:
            return idx, prepare_func(row, engine_name)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            return idx, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, row in df.iterrows():
            future = executor.submit(process_row, idx, row)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Running solver"):
            idx, result = future.result()
            results[idx] = result

    return results


def prepare_row(row: pd.Series, engine_name: str) -> str:
    """
    Prepare and solve a single row.

    Args:
        row: Dataframe row
        engine_name: LLM engine name

    Returns:
        Model response string
    """
    if not HAS_LITELLM:
        raise ImportError("litellm is required for API calls. Install with: pip install litellm")

    query = row.get("query", "")
    if not query:
        return ""

    # Build messages
    messages = [{"role": "user", "content": query}]

    # Add system prompt if available
    system_prompt = row.get("system_prompt")
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    try:
        # Get API key (round-robin)
        api_key = _api_keys[0] if _api_keys else ""
        if len(_api_keys) > 1:
            # Simple round-robin based on query hash
            idx = hash(query) % len(_api_keys)
            api_key = _api_keys[idx]

        response = completion(
            model=f"openai/{engine_name}",
            messages=messages,
            api_key=api_key,
            api_base=_api_base,
            max_tokens=512,
            temperature=0.1,
            timeout=60
        )

        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"API call error: {e}")
        return ""


def run_verification(
    df: pd.DataFrame,
    ans_col: str,
    engine_name: str,
    temperature: float = 1.0,
    n: int = 2,
    stop: str = "---",
    max_tokens: int = 250,
    max_workers: int = 5,
) -> List[List[str]]:
    """
    Run self-verification on model answers.

    Args:
        df: Input dataframe
        ans_col: Column name containing answers to verify
        engine_name: LLM engine name
        temperature: Sampling temperature
        n: Number of verification samples
        stop: Stop sequence
        max_tokens: Maximum tokens to generate
        max_workers: Number of parallel workers

    Returns:
        List of verification results (list of strings for each row)
    """
    if not HAS_LITELLM:
        raise ImportError("litellm is required for API calls. Install with: pip install litellm")

    init_providers()

    results = [None] * len(df)

    def verify_row(idx, row):
        try:
            query = row.get("query", "")
            answer = row.get(ans_col, "")

            if not query or not answer:
                return idx, []

            # Build verification prompt
            verify_prompt = f"""Given the question and answer below, determine if the answer is correct.
Question: {query}
Answer: {answer}

Is this answer correct? Answer with "yes" or "no" and explain briefly."""

            messages = [{"role": "user", "content": verify_prompt}]

            verifications = []
            for _ in range(n):
                try:
                    api_key = _api_keys[0] if _api_keys else ""
                    if len(_api_keys) > 1:
                        idx_key = hash(query + str(_)) % len(_api_keys)
                        api_key = _api_keys[idx_key]

                    response = completion(
                        model=f"openai/{engine_name}",
                        messages=messages,
                        api_key=api_key,
                        api_base=_api_base,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=stop,
                        timeout=60
                    )
                    verifications.append(response.choices[0].message.content or "")
                except Exception as e:
                    verifications.append("")

            return idx, verifications
        except Exception as e:
            print(f"Verification error for row {idx}: {e}")
            return idx, []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, row in df.iterrows():
            future = executor.submit(verify_row, idx, row)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Running verification"):
            idx, result = future.result()
            results[idx] = result

    return results


def compute_fraction_correct(verification_results: List[str]) -> float:
    """
    Compute the fraction of verifications that indicate the answer is correct.

    Args:
        verification_results: List of verification strings

    Returns:
        Fraction of "correct" verifications (0.0 to 1.0)
    """
    if not verification_results:
        return 0.0

    correct_count = 0
    for result in verification_results:
        if not result:
            continue
        result_lower = result.lower()
        # Check for positive indicators
        if "yes" in result_lower or "correct" in result_lower:
            correct_count += 1
        # Check for negative indicators (overrides positive)
        if "no" in result_lower or "incorrect" in result_lower or "wrong" in result_lower:
            correct_count -= 1

    return correct_count / len(verification_results) if verification_results else 0.0


def clean_answer(answer: str) -> str:
    """
    Clean answer string by removing quotes and extra whitespace.

    Args:
        answer: Raw answer string

    Returns:
        Cleaned answer string
    """
    if answer is None:
        return None
    if isinstance(answer, (float, type(pd.NA))) and pd.isna(answer):
        return None
    ans_str = str(answer).strip()
    if not ans_str:
        return None
    return ans_str.replace("'", "").replace('"', '')


def normalize_answer(s: str) -> str:
    """
    Normalize text for F1 calculation.

    Args:
        s: String to normalize

    Returns:
        Normalized string
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> tuple:
    """
    Calculate F1 score between prediction and ground truth.

    Args:
        prediction: Predicted text
        ground_truth: Ground truth text

    Returns:
        Tuple of (f1, precision, recall)
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall


def calculate_f1_for_models(
    df: pd.DataFrame,
    model_names: List[str],
    ground_truth_col: str = "ground_truth"
) -> pd.DataFrame:
    """
    Calculate F1 scores for multiple models.

    Args:
        df: Input dataframe
        model_names: List of model names (e.g., ["slm", "llm"])
        ground_truth_col: Column name for ground truth

    Returns:
        Dataframe with F1 score columns added
    """
    for model_name in model_names:
        pred_col = f"{model_name}_pred_ans"
        f1_col = f"{model_name}_f1"

        def calc_f1(row):
            pred = row.get(pred_col)
            gt = row.get(ground_truth_col)
            if pred is None or gt is None:
                return 0.0
            return f1_score(str(pred), str(gt))[0]

        df[f1_col] = df.apply(calc_f1, axis=1)

    return df


def categorize_rows(
    df: pd.DataFrame,
    slm_column: str = "slm_f1",
    llm_column: str = "llm_f1"
) -> pd.DataFrame:
    """
    Categorize rows based on model performance.

    Categories:
    - NEEDY: Small model fails, large model succeeds
    - GOOD: Small model succeeds
    - HOPELESS: Both models fail

    Args:
        df: Input dataframe
        slm_column: Column name for small model scores
        llm_column: Column name for large model scores

    Returns:
        Dataframe with category column added
    """
    def categorize(row):
        slm_score = row.get(slm_column, 0)
        llm_score = row.get(llm_column, 0)

        # Threshold for success (F1 > 0.5 or exact match)
        slm_success = slm_score > 0.5
        llm_success = llm_score > 0.5

        if slm_success:
            return "GOOD"
        elif llm_success:
            return "NEEDY"
        else:
            return "HOPELESS"

    df["category"] = df.apply(categorize, axis=1)
    return df
