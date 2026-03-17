#!/usr/bin/env python3
"""
为验证集生成LLM答案和置信度
遍历validation_set_1000_eval_with_prob.jsonl，为每个query生成：
- answer: 模型回答
- llm_confidence: 答案置信度(0-1)
- reasoning: 置信度理由
"""

import json
import time
from pathlib import Path
from openai import OpenAI

# 配置API
client = OpenAI(
    base_url="http://172.17.160.46:8080/v1",  # 根据你的实际API地址修改
    api_key="sk-xxx"  # 根据你的实际API key修改
)

def generate_answer_with_confidence(query: str, true_level: str):
    """
    让LLM生成答案和置信度
    """
    system_prompt = """你是专业金融客服助手。请回答用户问题，并评估你答案的准确性。

要求：
1. 先给出详细、专业的回答
2. 然后给出置信度(0.0-1.0)：
   - 0.9-1.0：非常确定，有标准答案
   - 0.7-0.9：比较确定，但可能有细节偏差
   - 0.5-0.7：不太确定，建议用户核实
   - <0.5：不确定，应该转人工

输出格式（严格JSON，不要有任何其他内容）：
{
  "answer": "你的详细回答",
  "confidence": 0.85,
  "reasoning": "为什么是这个置信度的简要说明"
}"""

    user_prompt = f"用户问题：{query}\n\n请回答并给出置信度："
    
    try:
        response = client.chat.completions.create(
            model="Qwen2.5-32B-Instruct",  # 根据你的实际模型修改
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=800
        )
        
        content = response.choices[0].message.content.strip()
        
        # 尝试解析JSON
        try:
            result = json.loads(content)
            return {
                "answer": result.get("answer", ""),
                "llm_confidence": float(result.get("confidence", 0.5)),
                "reasoning": result.get("reasoning", "")
            }
        except json.JSONDecodeError:
            # 如果解析失败，尝试提取信息
            print(f"JSON解析失败，原始内容：{content[:100]}...")
            return {
                "answer": content[:200],
                "llm_confidence": 0.5,
                "reasoning": "解析失败，使用默认置信度"
            }
            
    except Exception as e:
        print(f"API调用失败: {e}")
        return {
            "answer": "",
            "llm_confidence": 0.0,
            "reasoning": f"API错误: {str(e)}"
        }

def process_dataset(input_file: str, output_file: str, limit: int = None):
    """
    处理数据集
    
    Args:
        input_file: 输入JSONL文件路径
        output_file: 输出JSONL文件路径
        limit: 限制处理数量（用于测试）
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    # 读取输入数据
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    total = len(data) if limit is None else min(limit, len(data))
    print(f"总共{len(data)}条数据，将处理{total}条")
    print(f"输出文件: {output_path}")
    
    # 处理每条数据
    results = []
    for i, item in enumerate(data[:total], 1):
        query = item['query']
        true_level = item.get('true_level', 'unknown')
        
        print(f"\n[{i}/{total}] 处理: {query[:30]}...")
        print(f"    真实风险等级: {true_level}")
        
        # 调用LLM生成答案和置信度
        start_time = time.time()
        llm_result = generate_answer_with_confidence(query, true_level)
        elapsed = time.time() - start_time
        
        # 合并结果
        new_item = item.copy()
        new_item['llm_answer'] = llm_result['answer']
        new_item['llm_confidence'] = llm_result['llm_confidence']
        new_item['llm_reasoning'] = llm_result['reasoning']
        
        results.append(new_item)
        
        print(f"    置信度: {llm_result['llm_confidence']:.2f}")
        print(f"    耗时: {elapsed:.1f}s")
        
        # 每10条保存一次
        if i % 10 == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(f"    已保存进度: {i}/{total}")
        
        # 避免API限流
        time.sleep(0.5)
    
    # 最终保存
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 完成！已处理{total}条数据")
    print(f"输出文件: {output_path}")
    
    # 显示统计
    confidences = [r['llm_confidence'] for r in results]
    print(f"\n置信度统计:")
    print(f"  平均: {sum(confidences)/len(confidences):.2f}")
    print(f"  最小: {min(confidences):.2f}")
    print(f"  最大: {max(confidences):.2f}")

if __name__ == "__main__":
    # 配置路径
    INPUT_FILE = "./data/validation/validation_set_1000_eval_with_prob.jsonl"
    OUTPUT_FILE = "./data/validation/validation_set_with_llm_confidence.jsonl"
    
    # 处理（可以设置limit参数先测试几条）
    process_dataset(INPUT_FILE, OUTPUT_FILE, limit=10)  # 先处理10条测试
    
    # 如果要处理全部数据，取消limit参数：
    # process_dataset(INPUT_FILE, OUTPUT_FILE)
