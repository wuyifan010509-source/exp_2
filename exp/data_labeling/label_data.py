"""
数据标注脚本 (Data Labeling)
对已扩充的 Query 数据进行批量标注，输出带 category + cost_label 的 JSONL。

使用方式：
    python -m data_labeling.label_data
    python -m data_labeling.label_data --input data.jsonl --output labeled.jsonl
    python -m data_labeling.label_data --input raw_queries.txt  # 纯文本，每行一条
"""

import os
import sys
import json
import time
import argparse
import re
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore, Lock

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
    LABELING_PROMPT_TEMPLATE,
    EXPANDED_OUTPUT_PATH,
    LABELED_OUTPUT_PATH,
    CHECKPOINT_DIR,
)


def load_queries_from_jsonl(path: str) -> List[Dict]:
    """加载 JSONL 格式数据"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_queries_from_txt(path: str) -> List[Dict]:
    """加载纯文本格式，每行一条 Query"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if q:
                records.append({"query": q})
    return records


def load_labeled_checkpoint() -> Dict[str, Dict]:
    """加载已标注的断点数据，返回 {query: record}"""
    ckpt_path = os.path.join(CHECKPOINT_DIR, "labeled_checkpoint.jsonl")
    labeled = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    labeled[record["query"]] = record
        print(f"[checkpoint] 已标注 {len(labeled)} 条")
    return labeled


def save_labeled_checkpoint(records: List[Dict]):
    """保存标注断点"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "labeled_checkpoint.jsonl")
    with open(ckpt_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_llm_response(response_text: str) -> List[Dict]:
    """
    解析 LLM 返回的 JSON 标注结果。
    处理潜在的格式问题（多余文字、markdown 包裹等）。
    """
    results = []

    # 尝试逐行解析 JSON
    for line in response_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # 去掉 markdown 代码块标记
        line = line.strip("`")
        if line.startswith("json"):
            continue
        if line in ("```", "```json", "[", "]"):
            continue

        # 尝试解析 JSON
        try:
            record = json.loads(line)
            # 验证必需字段
            if "query" in record and "category" in record and "cost_label" in record:
                # 验证 category 合法性
                if record["category"] in CATEGORY_DEFINITIONS:
                    results.append(record)
                else:
                    print(f"[warn] 未知类别 '{record['category']}': {record['query'][:30]}")
        except json.JSONDecodeError:
            # 尝试找出行内的 JSON 对象
            json_match = re.search(r'\{[^}]+\}', line)
            if json_match:
                try:
                    record = json.loads(json_match.group())
                    if "query" in record and "category" in record:
                        results.append(record)
                except json.JSONDecodeError:
                    pass

    return results


def label_batch(
    client: OpenAI,
    queries: List[str],
    rate_limiter: Semaphore,
) -> List[Dict]:
    """
    对一批 Query 进行标注。

    Args:
        client: OpenAI client
        queries: 待标注的 query 列表
        rate_limiter: 限流

    Returns:
        标注结果列表
    """
    queries_text = "\n".join(f"- {q}" for q in queries)

    prompt = LABELING_PROMPT_TEMPLATE.format(queries=queries_text)

    for attempt in range(RETRY_MAX):
        try:
            rate_limiter.acquire()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # 标注任务用低温度确保一致性
                max_tokens=4096,
            )

            text = response.choices[0].message.content.strip()
            results = parse_llm_response(text)

            if len(results) < len(queries) * 0.5:
                print(f"[warn] 返回结果过少 ({len(results)}/{len(queries)}), 重试")
                time.sleep(RETRY_BACKOFF)
                continue

            return results

        except Exception as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"[error] 标注 attempt {attempt+1}: {e}, "
                  f"等待 {wait:.1f}s 后重试")
            time.sleep(wait)

    print(f"[fail] 标注批次失败: {queries[0][:30]}...")
    return []


def label_all(
    client: OpenAI,
    queries: List[str],
    batch_size: int = 10,
) -> List[Dict]:
    """
    批量标注全部 queries。

    Args:
        client: OpenAI client
        queries: 全部待标注 queries
        batch_size: 每批发送给 LLM 的数量

    Returns:
        标注结果列表
    """
    # 加载断点
    labeled_map = load_labeled_checkpoint()
    all_results = list(labeled_map.values())

    # 过滤已标注的
    unlabeled = [q for q in queries if q not in labeled_map]
    print(f"\n[label] 总计 {len(queries)} 条, 已标注 {len(labeled_map)} 条, "
          f"待标注 {len(unlabeled)} 条")

    if not unlabeled:
        return all_results

    # 限流
    rate_limiter = Semaphore(REQUESTS_PER_MINUTE)
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

    # 分批标注
    batches = [
        unlabeled[i:i + batch_size]
        for i in range(0, len(unlabeled), batch_size)
    ]

    for batch_idx, batch in enumerate(batches):
        print(f"\n[label] 批次 {batch_idx+1}/{len(batches)} ({len(batch)} 条)")

        results = label_batch(client, batch, rate_limiter)

        for r in results:
            if r["query"] not in labeled_map:
                labeled_map[r["query"]] = r
                all_results.append(r)

        # 保存断点
        save_labeled_checkpoint(all_results)
        print(f"[label] 累计 {len(all_results)} 条 (checkpoint已保存)")

        # 短暂等待
        time.sleep(0.5)

    return all_results


def save_final_output(records: List[Dict], output_path: str):
    """保存最终标注结果"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计
    from collections import Counter
    cat_counts = Counter(r["category"] for r in records)
    cost_counts = Counter(r["cost_label"] for r in records)

    print(f"\n{'='*60}")
    print(f"标注完成! 共 {len(records)} 条")
    print(f"保存到: {output_path}")
    print(f"\n类别分布:")
    for cat, count in sorted(cat_counts.items()):
        name = CATEGORY_DEFINITIONS.get(cat, {}).get("name", "未知")
        pct = count / len(records) * 100
        print(f"  {cat} ({name}): {count} ({pct:.1f}%)")

    print(f"\nCost 分布:")
    for cost, count in sorted(cost_counts.items()):
        pct = count / len(records) * 100
        print(f"  Cost={cost}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="数据标注脚本")
    parser.add_argument(
        "--input",
        type=str,
        default=EXPANDED_OUTPUT_PATH,
        help="输入文件路径 (JSONL 或 TXT)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=LABELED_OUTPUT_PATH,
        help="输出 JSONL 路径",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="每批标注数量",
    )
    parser.add_argument(
        "--relabel",
        action="store_true",
        help="忽略断点，重新标注全部数据",
    )
    args = parser.parse_args()

    # 验证 API 配置
    if not API_KEY or not API_BASE_URL or not MODEL_NAME:
        print("错误：请先在 data_labeling/config.py 中填入 API_KEY, API_BASE_URL, MODEL_NAME")
        sys.exit(1)

    # 加载数据
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"错误：输入文件不存在: {input_path}")
        sys.exit(1)

    if input_path.endswith(".jsonl"):
        records = load_queries_from_jsonl(input_path)
    elif input_path.endswith(".txt"):
        records = load_queries_from_txt(input_path)
    else:
        # 尝试 JSONL
        try:
            records = load_queries_from_jsonl(input_path)
        except:
            records = load_queries_from_txt(input_path)

    queries = [r["query"] for r in records]
    print(f"[load] 加载了 {len(queries)} 条待标注数据")

    # 清除断点（如果 relabel）
    if args.relabel:
        ckpt_path = os.path.join(CHECKPOINT_DIR, "labeled_checkpoint.jsonl")
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
            print("[relabel] 已清除标注断点")

    # 初始化 client
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    # 标注
    results = label_all(client, queries, batch_size=args.batch_size)

    # 保存
    save_final_output(results, args.output)


if __name__ == "__main__":
    main()
