"""
数据扩充脚本 (Data Expansion)
利用 LLM API 基于种子 Query 为每个类别生成更多变体。

使用方式：
    python -m data_labeling.expand_data
    python -m data_labeling.expand_data --categories A1 A2 C1
    python -m data_labeling.expand_data --count 50
"""

import os
import sys
import csv
import json
import time
import argparse
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from typing import List, Dict, Optional

from openai import OpenAI

from .config import (
    API_KEY,
    API_BASE_URL,
    MODEL_NAME,
    MAX_WORKERS,
    REQUESTS_PER_MINUTE,
    RETRY_MAX,
    RETRY_BACKOFF,
    CATEGORY_DEFINITIONS,
    EXPANSION_TARGETS,
    SEED_QUERIES,
    EXPANSION_PROMPT_TEMPLATE,
    OOS_CSV_PATH,
    EXPANDED_OUTPUT_PATH,
    CHECKPOINT_DIR,
)


def load_oos_csv(csv_path: str) -> List[str]:
    """从 CSV 加载已有 OOS 数据"""
    queries = []
    if not os.path.exists(csv_path):
        print(f"[warn] OOS CSV 不存在: {csv_path}")
        return queries

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", "").strip()
            if q and q != ".":
                queries.append(q)

    print(f"[load] 从 {csv_path} 加载了 {len(queries)} 条 OOS 数据")
    return queries


def load_checkpoint(category: str) -> List[str]:
    """加载断点数据"""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{category}.jsonl")
    if not os.path.exists(ckpt_path):
        return []

    queries = []
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                queries.append(data["query"])
    print(f"[checkpoint] {category}: 已有 {len(queries)} 条")
    return queries


def save_checkpoint(category: str, queries: List[str]):
    """保存断点数据"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{category}.jsonl")
    with open(ckpt_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps({"query": q, "category": category}, ensure_ascii=False) + "\n")


def call_llm_expand(
    client: OpenAI,
    category_code: str,
    count: int,
    existing_queries: List[str],
    rate_limiter: Semaphore,
) -> List[str]:
    """
    调用 LLM 生成指定类别的扩充数据。

    Args:
        client: OpenAI client
        category_code: 类别代码
        count: 需要生成的数量
        existing_queries: 已有 queries（用于去重）
        rate_limiter: 限流信号量

    Returns:
        新生成的 queries 列表
    """
    cat_def = CATEGORY_DEFINITIONS[category_code]
    seeds = SEED_QUERIES.get(category_code, [])

    # 如果 OOS 类，将已有 CSV 数据也加入种子
    seed_text = "\n".join(f"- {s}" for s in seeds[:15])

    prompt = EXPANSION_PROMPT_TEMPLATE.format(
        count=count,
        category_code=category_code,
        category_name=cat_def["name"],
        category_desc=cat_def["desc"],
        seed_queries=seed_text,
    )

    for attempt in range(RETRY_MAX):
        try:
            rate_limiter.acquire()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=4096,
            )

            text = response.choices[0].message.content.strip()
            lines = [
                line.strip().lstrip("0123456789.-) 、")
                for line in text.split("\n")
                if line.strip() and len(line.strip()) > 2
            ]

            # 去重
            existing_set = set(existing_queries)
            new_queries = [q for q in lines if q not in existing_set]

            print(f"[expand] {category_code}: 生成 {len(lines)} 条, "
                  f"去重后 {len(new_queries)} 条")
            return new_queries

        except Exception as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"[error] {category_code} attempt {attempt+1}: {e}, "
                  f"等待 {wait:.1f}s 后重试")
            time.sleep(wait)

    print(f"[fail] {category_code}: 重试耗尽")
    return []


def expand_category(
    client: OpenAI,
    category_code: str,
    target_count: int,
    existing_queries: List[str],
    rate_limiter: Semaphore,
    batch_size: int = 50,
) -> List[str]:
    """
    分批扩充单个类别到目标数量。
    """
    all_queries = list(existing_queries)

    while len(all_queries) < target_count:
        remaining = target_count - len(all_queries)
        batch = min(batch_size, remaining)

        print(f"\n[expand] {category_code}: 当前 {len(all_queries)}/{target_count}, "
              f"本轮生成 {batch} 条")

        new = call_llm_expand(
            client, category_code, batch, all_queries, rate_limiter
        )

        if not new:
            print(f"[warn] {category_code}: 本轮无新数据，跳过")
            break

        all_queries.extend(new)
        save_checkpoint(category_code, all_queries)
        print(f"[expand] {category_code}: 累计 {len(all_queries)} 条 (checkpoint已保存)")

        # 短暂等待
        time.sleep(1)

    return all_queries[:target_count]


def merge_and_save(
    all_data: Dict[str, List[str]],
    output_path: str,
):
    """合并所有类别数据并保存为 JSONL"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for category, queries in all_data.items():
            cost = CATEGORY_DEFINITIONS[category]["cost"]
            for q in queries:
                record = {
                    "query": q,
                    "category": category,
                    "cost_label": cost,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1

    print(f"\n[save] 共 {total} 条数据已保存到 {output_path}")

    # 打印分布
    print("\n类别分布:")
    for cat, queries in sorted(all_data.items()):
        pct = len(queries) / total * 100 if total > 0 else 0
        cost = CATEGORY_DEFINITIONS[cat]["cost"]
        print(f"  {cat} ({CATEGORY_DEFINITIONS[cat]['name']}): "
              f"{len(queries)} 条 ({pct:.1f}%), Cost={cost}")


def main():
    parser = argparse.ArgumentParser(description="数据扩充脚本")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="指定要扩充的类别（默认扩充全部）",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="每个类别的目标数量（覆盖 config 中的设定）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=EXPANDED_OUTPUT_PATH,
        help="输出 JSONL 路径",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅打印计划，不实际调用 API",
    )
    args = parser.parse_args()

    # 验证 API 配置
    if not args.dry_run:
        if not API_KEY or not API_BASE_URL or not MODEL_NAME:
            print("错误：请先在 data_labeling/config.py 中填入 API_KEY, API_BASE_URL, MODEL_NAME")
            sys.exit(1)

    categories = args.categories or list(EXPANSION_TARGETS.keys())

    # 加载已有 OOS 数据
    oos_queries = load_oos_csv(OOS_CSV_PATH)

    # 计划
    print("\n" + "=" * 60)
    print("数据扩充计划")
    print("=" * 60)

    for cat in categories:
        target = args.count or EXPANSION_TARGETS.get(cat, 100)
        existing = load_checkpoint(cat)
        if cat == "OOS":
            existing = list(set(existing + oos_queries))

        remaining = max(0, target - len(existing))
        print(f"  {cat} ({CATEGORY_DEFINITIONS[cat]['name']}): "
              f"已有 {len(existing)}, 目标 {target}, 需生成 {remaining}")

    if args.dry_run:
        print("\n[dry_run] 仅打印计划，未调用 API")
        return

    # 初始化 client
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    rate_limiter = Semaphore(REQUESTS_PER_MINUTE)

    # 释放限流器的定时器
    import threading
    def release_rate_limiter():
        while True:
            time.sleep(60.0 / REQUESTS_PER_MINUTE)
            try:
                rate_limiter.release()
            except ValueError:
                pass

    timer_thread = threading.Thread(target=release_rate_limiter, daemon=True)
    timer_thread.start()

    # 逐类别扩充
    all_data = {}
    for cat in categories:
        target = args.count or EXPANSION_TARGETS.get(cat, 100)
        existing = load_checkpoint(cat)
        if cat == "OOS":
            existing = list(set(existing + oos_queries))

        if len(existing) >= target:
            print(f"\n[skip] {cat}: 已达目标 ({len(existing)}/{target})")
            all_data[cat] = existing[:target]
            continue

        all_data[cat] = expand_category(
            client, cat, target, existing, rate_limiter
        )

    # 保存
    merge_and_save(all_data, args.output)


if __name__ == "__main__":
    main()
