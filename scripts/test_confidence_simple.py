#!/usr/bin/env python3
"""简化版测试脚本"""
import json
from openai import OpenAI

client = OpenAI(
    base_url="http://172.17.160.46:8080/v1",
    api_key="sk-xxx"
)

# 读取前2条数据
with open("./data/validation/validation_set_1000_eval_with_prob.jsonl", 'r') as f:
    lines = f.readlines()[:2]

results = []

for i, line in enumerate(lines, 1):
    item = json.loads(line)
    query = item['query']
    true_level = item['true_level']
    
    print(f"\n{'='*60}")
    print(f"[{i}/2] Query: {query}")
    print(f"真实等级: {true_level}")
    
    system_prompt = """你是专业金融客服助手。请回答用户问题，并评估你答案的准确性。

要求：
1. 先给出详细、专业的回答
2. 然后给出置信度(0.0-1.0)

输出格式（严格JSON）：
{
  "answer": "你的详细回答",
  "confidence": 0.85,
  "reasoning": "为什么是这个置信度"
}"""

    try:
        response = client.chat.completions.create(
            model="Qwen2.5-32B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"用户问题：{query}\n\n请回答并给出置信度："}
            ],
            temperature=0.0,
            max_tokens=800
        )
        
        content = response.choices[0].message.content.strip()
        print(f"\n原始回复:\n{content}\n")
        
        # 解析JSON
        try:
            result = json.loads(content)
            item['llm_answer'] = result.get("answer", "")
            item['llm_confidence'] = float(result.get("confidence", 0.5))
            item['llm_reasoning'] = result.get("reasoning", "")
            
            print(f"✅ 解析成功!")
            print(f"置信度: {item['llm_confidence']:.2f}")
            print(f"理由: {item['llm_reasoning'][:50]}...")
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析失败: {e}")
            item['llm_answer'] = content[:200]
            item['llm_confidence'] = 0.5
            item['llm_reasoning'] = "解析失败"
            
    except Exception as e:
        print(f"❌ API调用失败: {e}")
        item['llm_answer'] = ""
        item['llm_confidence'] = 0.0
        item['llm_reasoning'] = f"错误: {str(e)}"
    
    results.append(item)

# 保存结果
output_file = "./data/validation/test_llm_confidence.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"\n{'='*60}")
print(f"✅ 完成！结果保存到: {output_file}")
print(f"\n预览:")
for r in results:
    print(f"\nQuery: {r['query']}")
    print(f"真实等级: {r['true_level']}")
    print(f"置信度: {r['llm_confidence']}")
    print(f"答案: {r['llm_answer'][:80]}...")
