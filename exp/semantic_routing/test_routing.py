"""
语义路由测试脚本
50条测试Query，覆盖三个风险等级 + 边界模糊Query
"""

import os
import sys

# 添加父目录到路径以支持相对导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_routing.router import SemanticRouter
import time


# 50条测试Query
TEST_QUERIES = [
    # === Low Cost (闲聊/基础类) ===
    "你好",
    "在吗",
    "谢谢",
    "再见",
    "今天天气怎么样",
    "讲个笑话",
    "你是机器人吗",
    "周末有什么好玩的",
    "早上好",
    "节日快乐",
    "你们公司在哪里",
    "客服工作到几点",
    "怎么联系人工客服",
    "APP在哪里下载",
    "网站打不开了怎么办",
    "你好呀",
    "在不在",
    "辛苦了",
    "拜拜",
    "外面天气热不热",
    
    # === Mid Cost (交易规则/操作类) ===
    "怎么开通创业板",
    "科创板开通条件",
    "港股通怎么开",
    "基金赎回几天到账",
    "银证转账多久到账",
    "交易密码忘了怎么办",
    "怎么修改绑定银行卡",
    "风险测评过期了怎么重新做",
    "新股申购有什么条件",
    "可转债怎么打新",
    "分红方式能改成红利再投资吗",
    "怎么查看交割单",
    "交易手续费是多少",
    "印花税怎么计算",
    "怎么导出对账单",
    
    # === High Cost (高危/资金风险类) ===
    "我要跳楼",
    "不想活了",
    "亏光了活不下去",
    "准备自杀",
    "死给你们看",
    "我要销户退钱",
    "全部清仓不玩了",
    "融资爆仓了怎么办",
    "担保比例低于130了",
    "被强制平仓了",
    "穿仓了要补钱吗",
    "期货公司追保",
    "期权到期忘记行权",
    "我要举报你们",
    "找记者曝光你们",
]


def run_tests():
    """运行全部测试"""
    print("=" * 70)
    print("语义路由测试 - 50条Query")
    print("=" * 70)
    
    # 初始化路由器
    print("\n[初始化] 加载模型...")
    router = SemanticRouter(tau_high=0.7)
    
    print("\n" + "-" * 70)
    print("开始测试 (高危阈值 τ_high=0.7)")
    print("-" * 70)
    
    results = []
    latencies = []
    
    for i, query in enumerate(TEST_QUERIES, 1):
        start = time.time()
        level, cost, max_sim, details = router.route(query)
        latency = (time.time() - start) * 1000  # ms
        
        latencies.append(latency)
        results.append({
            "id": i,
            "query": query,
            "level": level,
            "cost": cost,
            "max_sim": max_sim,
            "latency_ms": latency,
            "flag": details["flag"]
        })
        
        # 标记非对称拦截
        flag_str = "⚠️ " if details["flag"] == "ASYMMETRIC_FLAG" else "   "
        
        print(f"{flag_str}[{i:02d}] {query:25s} → {level:6s} | Cost={cost:4d} | "
              f"Sim={max_sim:.3f} | {latency:5.2f}ms")
    
    # 统计报告
    print("\n" + "=" * 70)
    print("测试报告")
    print("=" * 70)
    
    # 统计各等级分布
    level_counts = {"low": 0, "mid": 0, "high": 0}
    asymmetric_count = 0
    for r in results:
        level_counts[r["level"]] += 1
        if r["flag"] == "ASYMMETRIC_FLAG":
            asymmetric_count += 1
    
    print(f"\n分布统计:")
    print(f"  • Low  (Cost=1):    {level_counts['low']:2d} 条")
    print(f"  • Mid  (Cost=50):   {level_counts['mid']:2d} 条")
    print(f"  • High (Cost=1000): {level_counts['high']:2d} 条")
    print(f"  • 非对称拦截: {asymmetric_count} 条")
    
    # 延迟统计
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)
    
    print(f"\n延迟统计:")
    print(f"  • 平均: {avg_latency:.2f} ms")
    print(f"  • 最大: {max_latency:.2f} ms")
    print(f"  • 最小: {min_latency:.2f} ms")
    print(f"  • 满足 <10ms 要求: {'✅' if avg_latency < 10 else '❌'}")
    
    # 验证预期
    print("\n" + "-" * 70)
    print("预期验证:")
    # 前20条应该是Low
    low_correct = sum(1 for r in results[:20] if r["level"] == "low")
    # 中间15条应该是Mid  
    mid_correct = sum(1 for r in results[20:35] if r["level"] == "mid")
    # 最后15条应该是High
    high_correct = sum(1 for r in results[35:] if r["level"] == "high")
    
    print(f"  • Low 类别正确率: {low_correct}/20 ({low_correct/20*100:.0f}%)")
    print(f"  • Mid 类别正确率: {mid_correct}/15 ({mid_correct/15*100:.0f}%)")
    print(f"  • High类别正确率: {high_correct}/15 ({high_correct/15*100:.0f}%)")
    
    total_correct = low_correct + mid_correct + high_correct
    print(f"  • 总体准确率: {total_correct}/50 ({total_correct/50*100:.0f}%)")
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_tests()
