"""
统筹评估 Pipeline (Evaluation)
将语义路由和 SLM 蒸馏模型放入统一评测框架，计算总期望代价并对比。

评测逻辑：
1. 准备极端分布测试集（20% Low + 70% Mid + 10% High）
2. 分别用两种方法跑测试集
3. 记录处理耗时和分发错误
4. 计算总期望代价，得出最优方案
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 评估配置
# ============================================================

# 代价参数
COST_MAP = {
    "low": 1,
    "mid": 50,
    "high": 1000,
}

# 人工转接代价（模拟：分发错误后的人工介入成本）
C_HUMAN = 200

# 延迟代价系数（每毫秒延迟带来的代价）
LATENCY_COST_COEFF = 0.01

# 错误代价倍率
# 漏报（把高危判为低危）比误报（把低危判为高危）更危险
ERROR_COST_MULTIPLIER = {
    "underpredict": 10.0,   # 漏报系数
    "overpredict": 1.0,     # 误报系数
}


# ============================================================
# 评估器基类
# ============================================================

class BaseEvaluator:
    """评估器基类，定义统一的评估接口"""

    def predict(self, query: str) -> Dict:
        """
        预测单条 Query。

        Returns:
            {"predicted_level": str, "predicted_cost": int/float, "latency_ms": float}
        """
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError


class SemanticRoutingEvaluator(BaseEvaluator):
    """语义路由评估器"""

    def __init__(self):
        from semantic_routing.router import SemanticRouter
        self.router = SemanticRouter()

    def predict(self, query: str) -> Dict:
        result = self.router.route(query)
        # 兼容旧版本 router.py 返回 tuple (level, sim) 或 (level, cost, sim...) 的情况
        if isinstance(result, tuple):
            level = result[0]
            # 如果能拿到 cost 就拿，拿不到就按字典推断
            from slm_distillation.config import COST_LABEL_MAP
            cost_map = {"low": 1, "mid": 50, "high": 1000}
            cost = cost_map.get(level, 50)
            return {
                "predicted_level": level,
                "predicted_cost": cost,
                "latency_ms": 1.0, # 旧版没存 latency
            }
        
        return {
            "predicted_level": result.get("predicted_level", "mid"),
            "predicted_cost": result.get("predicted_cost", 50),
            "latency_ms": result.get("latency_ms", 1.0),
        }

    def get_name(self) -> str:
        return "Semantic Routing (非对称代价语义路由)"


class SLMDistillationEvaluator(BaseEvaluator):
    """SLM 蒸馏模型评估器"""

    def __init__(self, adapter_path: str = None):
        from slm_distillation.inference import SLMCostPredictor
        from slm_distillation.config import OUTPUT_DIR, COST_LABEL_MAP

        self.predictor = SLMCostPredictor(
            adapter_path=adapter_path or OUTPUT_DIR
        )
        self.cost_label_map = COST_LABEL_MAP
        # 反向映射: cost_value → level_name
        self.cost_to_level = {1: "low", 50: "mid", 1000: "high"}

    def predict(self, query: str) -> Dict:
        result = self.predictor.predict(query)
        cost = result["predicted_cost"]
        level = self.cost_to_level.get(cost, "mid")
        return {
            "predicted_level": level,
            "predicted_cost": cost,
            "latency_ms": result["ttft_ms"],
        }

    def get_name(self) -> str:
        return "SLM Distillation (非对称代价蒸馏模型)"



class LLMBaselineEvaluator(BaseEvaluator):
    """32B 大模型基线评估器"""
    
    def __init__(self):
        from openai import OpenAI
        
        # 从环境变量加载配置
        self.base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
        self.model_name = os.getenv("OPENAI_MODEL_BASE", "Qwen2.5-32B-Instruct")
        self.api_key = os.getenv("OPENAI_API_KEY", "sk-xxx")
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        self.prompt_template = """你是一个金融客服风险分类专家。请判断以下用户query的风险等级：

query: {query}

风险等级定义：
- Low: 闲聊、日常咨询（如天气、问候）
- Mid: 普通业务咨询（如密码修改、费率查询）
- High: 高危风险（如强平、自杀倾向、资金异常）

请只输出一个单词：Low 或 Mid 或 High"""
        
        print(f"[LLMBaselineEvaluator] 初始化完成")
        print(f"[LLMBaselineEvaluator] 模型: {self.model_name}")
        print(f"[LLMBaselineEvaluator] API地址: {self.base_url}")
    
    def predict(self, query: str) -> Dict:
        """使用 32B 大模型预测风险等级"""
        import time
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的金融客服风险分类助手。"},
                    {"role": "user", "content": self.prompt_template.format(query=query)}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            # 解析响应
            raw_output = response.choices[0].message.content.strip()
            latency_ms = (time.time() - start_time) * 1000
            
            # 标准化输出
            raw_upper = raw_output.upper()
            if "HIGH" in raw_upper or "高危" in raw_output:
                level = "high"
                cost = 1000
            elif "MID" in raw_upper or "中危" in raw_output or "中等" in raw_output:
                level = "mid"
                cost = 50
            elif "LOW" in raw_upper or "低危" in raw_output or "低" in raw_output:
                level = "low"
                cost = 1
            else:
                # 默认 mid
                level = "mid"
                cost = 50
            
            return {
                "predicted_level": level,
                "predicted_cost": cost,
                "latency_ms": latency_ms,
                "raw_output": raw_output,
                "confidence": 1.0
            }
            
        except Exception as e:
            print(f"[LLMBaselineEvaluator] 错误: {e}")
            # 出错时返回默认值
            return {
                "predicted_level": "mid",
                "predicted_cost": 50,
                "latency_ms": (time.time() - start_time) * 1000,
                "error": str(e)
            }
    
    def get_name(self) -> str:
        return f"LLM Baseline (32B大模型-{self.model_name})"


# ============================================================
# 评估指标计算
# ============================================================

def compute_error_cost(
    true_level: str,
    pred_level: str,
) -> float:
    """
    计算单条预测的分发错误代价。

    逻辑：
    - 正确分类 → 代价 0
    - 漏报（pred 级别 < true 级别）→ 高惩罚
    - 误报（pred 级别 > true 级别）→ 低惩罚
    """
    level_order = {"low": 0, "mid": 1, "high": 2}
    true_ord = level_order[true_level]
    pred_ord = level_order[pred_level]

    if pred_ord == true_ord:
        return 0.0

    # 代价差
    cost_diff = abs(COST_MAP[true_level] - COST_MAP[pred_level])

    if pred_ord < true_ord:
        # 漏报：预测低于实际 → 高危被遗漏
        return cost_diff * ERROR_COST_MULTIPLIER["underpredict"] + C_HUMAN
    else:
        # 误报：预测高于实际 → 浪费资源但安全
        return cost_diff * ERROR_COST_MULTIPLIER["overpredict"]


def evaluate_method(
    evaluator: BaseEvaluator,
    test_data: List[Dict],
) -> Dict:
    """
    对单个方法进行完整评估。

    Args:
        evaluator: 评估器
        test_data: [{"query": str, "true_level": str, "true_cost": int}]

    Returns:
        评估结果字典
    """
    print(f"\n{'='*70}")
    print(f"评估: {evaluator.get_name()}")
    print(f"{'='*70}")

    results = []
    total_latency = 0
    total_error_cost = 0
    confusion = defaultdict(int)  # (true, pred) → count
    level_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, item in enumerate(test_data):
        pred = evaluator.predict(item["query"])
        true_level = item["true_level"]
        pred_level = pred["predicted_level"]
        latency = pred["latency_ms"]

        # 错误代价
        error_cost = compute_error_cost(true_level, pred_level)

        # 延迟代价
        latency_cost = latency * LATENCY_COST_COEFF

        total_latency += latency
        total_error_cost += error_cost

        confusion[(true_level, pred_level)] += 1
        level_stats[true_level]["total"] += 1
        if pred_level == true_level:
            level_stats[true_level]["correct"] += 1

        results.append({
            "query": item["query"],
            "true_level": true_level,
            "pred_level": pred_level,
            "latency_ms": latency,
            "error_cost": error_cost,
            "latency_cost": latency_cost,
        })

    n = len(test_data)

    # 总期望代价
    avg_latency_cost = (total_latency * LATENCY_COST_COEFF) / n
    avg_error_cost = total_error_cost / n
    total_expected_cost = avg_error_cost + avg_latency_cost

    # 各级别 Recall
    recall_by_level = {}
    for level in ["low", "mid", "high"]:
        stats = level_stats[level]
        recall_by_level[level] = (
            stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        )

    # 漏报 / 误报统计
    underpredict_count = sum(
        v for (t, p), v in confusion.items()
        if {"low": 0, "mid": 1, "high": 2}[p] < {"low": 0, "mid": 1, "high": 2}[t]
    )
    overpredict_count = sum(
        v for (t, p), v in confusion.items()
        if {"low": 0, "mid": 1, "high": 2}[p] > {"low": 0, "mid": 1, "high": 2}[t]
    )

    latencies = [r["latency_ms"] for r in results]

    summary = {
        "method": evaluator.get_name(),
        "total_queries": n,
        "accuracy": sum(1 for r in results if r["true_level"] == r["pred_level"]) / n,
        "recall_by_level": recall_by_level,
        "underpredict_count": underpredict_count,
        "overpredict_count": overpredict_count,
        "avg_latency_ms": np.mean(latencies),
        "p99_latency_ms": np.percentile(latencies, 99),
        "max_latency_ms": np.max(latencies),
        "avg_error_cost": avg_error_cost,
        "avg_latency_cost": avg_latency_cost,
        "total_expected_cost": total_expected_cost,
        "confusion_matrix": dict(confusion),
    }

    # 打印摘要
    print(f"\n--- 结果摘要 ---")
    print(f"  总 Query 数: {n}")
    print(f"  总准确率:    {summary['accuracy']:.4f}")
    print(f"  Recall(Low):  {recall_by_level['low']:.4f}")
    print(f"  Recall(Mid):  {recall_by_level['mid']:.4f}")
    print(f"  Recall(High): {recall_by_level['high']:.4f}")
    print(f"  漏报数:   {underpredict_count}")
    print(f"  误报数:   {overpredict_count}")
    print(f"  平均延迟: {summary['avg_latency_ms']:.3f} ms")
    print(f"  P99延迟:  {summary['p99_latency_ms']:.3f} ms")
    print(f"\n--- 期望代价分解 ---")
    print(f"  平均错误代价:   {avg_error_cost:.2f}")
    print(f"  平均延迟代价:   {avg_latency_cost:.4f}")
    print(f"  ★ 总期望代价:  {total_expected_cost:.4f}")

    return summary


# ============================================================
# 对比评估
# ============================================================

def comparative_evaluation(
    test_data: List[Dict],
    methods: List[BaseEvaluator],
) -> Dict:
    """
    多方法对比评估。

    Returns:
        对比结果字典
    """
    print("\n" + "=" * 70)
    print("统筹评估 — 巅峰对决")
    print("=" * 70)
    print(f"测试集: {len(test_data)} 条")
    print(f"分布: Low={sum(1 for d in test_data if d['true_level']=='low')}, "
          f"Mid={sum(1 for d in test_data if d['true_level']=='mid')}, "
          f"High={sum(1 for d in test_data if d['true_level']=='high')}")
    print(f"对比方法: {len(methods)}")
    print(f"参数: C_human={C_HUMAN}, 延迟代价系数={LATENCY_COST_COEFF}")

    all_results = []
    for method in methods:
        result = evaluate_method(method, test_data)
        all_results.append(result)

    # 对比表
    print("\n\n" + "=" * 70)
    print("★ 最终对比 ★")
    print("=" * 70)
    print(f"{'指标':<20}", end="")
    for r in all_results:
        name_short = r["method"][:20]
        print(f"  {name_short:<25}", end="")
    print()
    print("-" * (20 + 27 * len(all_results)))

    metrics_display = [
        ("准确率", "accuracy", ".4f"),
        ("高危 Recall", None, ".4f"),  # 特殊处理
        ("漏报数", "underpredict_count", "d"),
        ("误报数", "overpredict_count", "d"),
        ("平均延迟(ms)", "avg_latency_ms", ".3f"),
        ("错误代价", "avg_error_cost", ".2f"),
        ("延迟代价", "avg_latency_cost", ".4f"),
        ("★总期望代价", "total_expected_cost", ".4f"),
    ]

    for label, key, fmt in metrics_display:
        print(f"{label:<20}", end="")
        for r in all_results:
            if key is None:
                # 高危 Recall 特殊取值
                val = r["recall_by_level"]["high"]
            else:
                val = r[key]
            print(f"  {val:<25{fmt}}", end="")
        print()

    # 结论
    best_idx = min(range(len(all_results)),
                   key=lambda i: all_results[i]["total_expected_cost"])
    print(f"\n{'='*70}")
    print(f"★ 结论: {all_results[best_idx]['method']}")
    print(f"  在不等式模型下，该方法的整体总损失最小 "
          f"(Total Expected Cost = {all_results[best_idx]['total_expected_cost']:.4f})")
    print(f"{'='*70}")

    return {
        "results": all_results,
        "best_method": all_results[best_idx]["method"],
        "best_cost": all_results[best_idx]["total_expected_cost"],
    }


# ============================================================
# 测试集加载
# ============================================================

def load_test_data(test_path: str) -> List[Dict]:
    """
    加载测试集 JSONL。

    支持多种字段格式，优先级：
    1. level 字段（fix_labels.py 输出）
    2. class_label 字段 (0/1/2)
    3. true_level 字段
    4. cost_label 字段（阈值推断，不推荐）
    """
    CLASS_TO_LEVEL = {0: "low", 1: "mid", 2: "high"}
    LEVEL_TO_COST = {"low": 1, "mid": 50, "high": 1000}

    data = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            # 确定 true_level
            if "level" in record:
                record["true_level"] = record["level"]
            elif "class_label" in record:
                record["true_level"] = CLASS_TO_LEVEL.get(
                    record["class_label"], "mid"
                )
            elif "true_level" not in record and "cost_label" in record:
                cost = record["cost_label"]
                if cost <= 10:
                    record["true_level"] = "low"
                elif cost <= 100:
                    record["true_level"] = "mid"
                else:
                    record["true_level"] = "high"

            # 确定 true_cost
            if "true_cost" not in record:
                record["true_cost"] = LEVEL_TO_COST.get(
                    record.get("true_level", "mid"), 50
                )

            data.append(record)
    return data


def generate_demo_test_data() -> List[Dict]:
    """
    生成 Demo 测试集，遵循极端分布要求：
    20% 闲聊(Low) + 70% 基础查询(Mid) + 10% 极高危(High)

    注意：这是演示用简化数据集，正式评估请使用真实标注数据。
    """
    low_queries = [
        "怎么修改密码", "APP闪退了", "你好", "客服电话是多少",
        "怎么绑定手机", "交易时间是什么", "怎么看行情",
        "开户流程是什么", "怎么注销", "软件更新了吗",
    ]
    mid_queries = [
        "印花税怎么扣", "科创板条件", "可转债怎么买", "ETF和股票区别",
        "两融利率多少", "分红什么时候到", "逆回购怎么操作", "打新股条件",
        "佣金费率怎么查", "T+1是什么", "北交所怎么开", "港股通规则",
        "期权是什么", "基金定投怎么设", "限价和市价区别", "做空怎么操作",
        "转融通是什么", "大宗交易折价", "涨停还能买吗", "回购怎么做",
        "融券怎么还", "质押回购条件", "可转债打新流程", "创业板开通",
        "沪深港通区别", "权证和期权", "场内基金交易", "信用账户转普通",
        "盘后交易时间", "龙虎榜怎么看", "集合竞价规则", "配股怎么操作",
        "要约收购影响", "停牌复牌规则", "新三板交易门槛",
    ]
    high_queries = [
        "为什么强平我", "入金没到账", "担保比不足", "仓位被平了",
        "可用资金为负",
    ]

    test_data = []
    for q in low_queries:
        test_data.append({"query": q, "true_level": "low", "true_cost": 1})
    for q in mid_queries:
        test_data.append({"query": q, "true_level": "mid", "true_cost": 50})
    for q in high_queries:
        test_data.append({"query": q, "true_level": "high", "true_cost": 1000})

    # 验证分布
    total = len(test_data)
    print(f"[Demo测试集] 总计: {total} 条")
    print(f"  Low:  {len(low_queries)} ({len(low_queries)/total*100:.0f}%)")
    print(f"  Mid:  {len(mid_queries)} ({len(mid_queries)/total*100:.0f}%)")
    print(f"  High: {len(high_queries)} ({len(high_queries)/total*100:.0f}%)")

    return test_data


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="统筹评估 Pipeline")
    parser.add_argument(
        "--test_file",
        type=str,
        default="slm_distillation/test_split.jsonl",
        help="JSONL 测试集路径",
    )
    parser.add_argument(
        "--slm_adapter",
        type=str,
        default=None,
        help="SLM LoRA adapter 路径",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=["routing", "slm", "llm", "both", "all"],
        default=["routing"],
        help="评估方法: routing=语义路由, slm=SLM蒸馏, llm=32B大模型基线, both=语义路由+SLM, all=全部三种",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="大模型测试时最大测试数量（默认100条，避免API费用过高）",
    )
    args = parser.parse_args()

    # 加载测试数据
    if args.test_file and os.path.exists(args.test_file):
        test_data = load_test_data(args.test_file)
    else:
        print(f"[评估] 警告: 未找到 {args.test_file}，使用 25 条 Demo 假数据")
        test_data = generate_demo_test_data()

    # 构建评估器列表
    methods = []
    method_set = set(args.methods)
    if "both" in method_set:
        method_set = {"routing", "slm"}
    if "all" in method_set:
        method_set = {"routing", "slm", "llm"}

    if "routing" in method_set:
        methods.append(SemanticRoutingEvaluator())

    if "slm" in method_set:
        methods.append(SLMDistillationEvaluator(adapter_path=args.slm_adapter))
    
    if "llm" in method_set:
        methods.append(LLMBaselineEvaluator())
        # 如果是大模型测试，限制测试数量
        if args.limit and len(test_data) > args.limit:
            print(f"\n[LLM测试] 限制测试数量为 {args.limit} 条（原始 {len(test_data)} 条）")
            # 按类别分层采样
            low_data = [d for d in test_data if d['true_level'] == 'low'][:args.limit//3]
            mid_data = [d for d in test_data if d['true_level'] == 'mid'][:args.limit//3]
            high_data = [d for d in test_data if d['true_level'] == 'high'][:args.limit//3]
            test_data = low_data + mid_data + high_data
            print(f"[LLM测试] 实际测试 {len(test_data)} 条")

    if len(methods) == 0:
        print("错误：未指定任何评估方法")
        return

    # 执行对比评估
    if len(methods) >= 2:
        comparative_evaluation(test_data, methods)
    else:
        evaluate_method(methods[0], test_data)


if __name__ == "__main__":
    main()

