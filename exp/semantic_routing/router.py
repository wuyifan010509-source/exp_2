"""
语义路由核心模块
实现基于余弦相似度的路由 + 非对称代价惩罚
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import time

from .anchors import ANCHORS, COST_MAP


class SemanticRouter:
    """语义路由器 - 非对称代价感知"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        tau_high: float = 0.7,  # 高危阈值
        device: str = None
    ):
        """
        Args:
            model_name: Embedding模型名称
            tau_high: 高危锚点相似度阈值，超过直接判定为高危
            device: 运行设备
        """
        self.tau_high = tau_high
        
        # 加载模型
        print(f"正在加载模型: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        
        # 预计算锚点Embedding
        self.anchor_embeddings = {}
        self._precompute_anchors()
        
    def _precompute_anchors(self):
        """预计算所有锚点的Embedding向量"""
        for level, queries in ANCHORS.items():
            embeddings = self.model.encode(queries, convert_to_numpy=True)
            self.anchor_embeddings[level] = embeddings
            print(f"  [{level}] {len(queries)} 条锚点已编码")
    
    def embed(self, text: str) -> np.ndarray:
        """将文本编码为向量"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    def route(self, query: str) -> Tuple[str, int, float, Dict]:
        """
        路由主函数
        
        Returns:
            level: 风险等级 (low/mid/high)
            cost: 代价数值
            max_sim: 最大相似度
            details: 详细信息
        """
        # 编码查询
        query_vec = self.embed(query)
        
        # 计算与各级锚点的相似度
        similarities = {}
        for level, anchor_vecs in self.anchor_embeddings.items():
            # 计算与该类所有锚点的相似度，取最大
            sims = [self.cosine_similarity(query_vec, av) for av in anchor_vecs]
            similarities[level] = max(sims)
        
        # 非对称代价惩罚逻辑
        # 如果与高危锚点相似度超过阈值，直接判定为高危
        if similarities["high"] > self.tau_high:
            level = "high"
            cost = COST_MAP["high"]
            max_sim = similarities["high"]
            flag = "ASYMMETRIC_FLAG"  # 被非对称规则拦截
        else:
            # 标准路由：取相似度最高的类别
            level = max(similarities, key=similarities.get)
            cost = COST_MAP[level]
            max_sim = similarities[level]
            flag = "BASELINE"
        
        details = {
            "similarities": similarities,
            "flag": flag,
            "tau_high": self.tau_high
        }
        
        return level, cost, max_sim, details
    
    def batch_route(self, queries: List[str]) -> List[Tuple[str, int, float, Dict]]:
        """批量路由"""
        results = []
        for q in queries:
            results.append(self.route(q))
        return results


def demo():
    """演示运行"""
    print("=" * 60)
    print("非对称代价感知语义路由系统")
    print("=" * 60)
    
    # 初始化路由器
    router = SemanticRouter(tau_high=0.7)
    
    # 测试Query
    test_queries = [
        "你好呀",
        "今天天气真好",
        "怎么开通科创板",
        "基金赎回几天到账",
        "我要跳楼",
        "不想活了",
        "融资爆仓了怎么办",
        "担保比例低于130会被平仓吗",
    ]
    
    print("\n" + "-" * 60)
    print("开始测试...")
    print("-" * 60)
    
    for query in test_queries:
        start = time.time()
        level, cost, max_sim, details = router.route(query)
        latency = (time.time() - start) * 1000  # ms
        
        flag_str = f"[{details['flag']}]" if details['flag'] == "ASYMMETRIC_FLAG" else ""
        print(f"\nQuery: {query}")
        print(f"  → Level: {level.upper()} | Cost: {cost} | Sim: {max_sim:.3f} {flag_str}")
        print(f"  → Latency: {latency:.2f} ms")
        print(f"  → All similarities: Low={details['similarities']['low']:.3f}, "
              f"Mid={details['similarities']['mid']:.3f}, "
              f"High={details['similarities']['high']:.3f}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
