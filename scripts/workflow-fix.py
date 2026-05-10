import pandas as pd
import json
import time
import inspect
import os
import io
import base64
from typing import Callable, Dict, Any, List
from docstring_parser import parse
from typing import get_type_hints
from contextlib import redirect_stdout
from openai import OpenAI 
from workflow_tool import WorkflowTool
from datetime import datetime

# API Token should be set via environment variable LLM_API_TOKEN
# Example: export LLM_API_TOKEN="your-token-here"
if "LLM_API_TOKEN" not in os.environ:
    raise ValueError("Please set LLM_API_TOKEN environment variable")

def log_message(log_file_path, message):
    """写入日志文件"""
    if log_file_path:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {message}\n")

def fix_and_validate_workflows(client, model, question, workflow, requirements, max_iterations=80, log_file_path="fix_log.txt"):
    """
    修复并验证工作流代码，并验证其可执行性
    """
    generator_tool = WorkflowTool(client, model)
    system_prompt = f"""你是一个专业的代码修复和验证助手。你的任务是：
基于给定的已有工作流代码，诊断代码存在的问题，找到原因并修复代码，确保代码能够在给定的网站上稳定获取实时性答案。

背景信息：
- 你将收到一个之前可以正常使用的工作流代码，该工作流代码用于在指定网站上稳定获取实时性问题的答案；
- 该代码现在运行出现了问题（可能是网站结构变动、网址不可访问、元素定位失效等）导致无法获取正确答案；
- 你需要找到代码无法工作的原因并修复代码使其重新正常工作。

要求：
- 代码修复后必须保持原有功能：回答的问题内容、最终输出格式都不能发生变化；
- 不允许使用硬编码数据作为兜底策略，必须确保代码能够获取真实的实时数据；
- 若网站结构发生变化，需要更新相应的元素定位和数据提取逻辑；
- 代码里使用from datetime import datetime；now = datetime.now()获取时间，不可预设一个固定时间；
- 所有码中不要包含任何try except，不要包含任何兜底策略。code_interpreter会使用exec执行代码，希望在代码不符合要求时，能直接报错；
- 工作流代码类似于爬虫代码，参考给你的代码样式，使用playwright和markdownify；
- 工作流代码必须经过测试确保可执行；
- 成功执行工作流代码后用save_generated_item保存。

工具使用技巧：
1. 使用check_html_content工具可以帮助你检查网页中内容，或者确保网页中包含所需信息；
2. 使用think工具深度分析问题原因和修复策略；
3. 使用code_interpreter工具测试代码是否可执行，如果执行成功，则调用check_output_content工具进行最终检查，否则调用think工具思考问题原因，并修改代码，直到代码可执行为止；
4. 关键结果需要return，main函数中使用result存储要return的内容，并使用print打印result；
5. 使用save_generated_item工具保存通过验证的问题和答案获取工作流代码。
6. 当你认为工作流的结果符合预期的时候，你就可以不再修改代码。而是保存
存在问题的工作流：
实时性问题：{question}

相应的工作流代码：
```python
{workflow}
```
"""
    user_prompt = f"""{requirements}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    log_message(log_file_path, f"=== 开始修复工作流 ===")
    log_message(log_file_path, f"问题: {question}")
    log_message(log_file_path, f"用户要求: {requirements}")

    # print(messages)
    # 最多尝试30轮对话
    for _ in range(max_iterations):
        print(f"第 {_ + 1} 轮对话...")
        log_message(log_file_path, f"--- 第 {_ + 1} 轮对话 ---")
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=generator_tool.get_openai_tools(),
                temperature=0.7,
                top_p=0.9,
                max_tokens=32768,
            )
            
            message = completion.choices[0].message
            assistant_content = message.content or ""
            if '<think>' in assistant_content and '</think>' in assistant_content:
                assistant_content = assistant_content.split('<think>')[1].split('</think>')[0]
            if '</think>' in assistant_content:
                assistant_content = assistant_content.split('</think>')[1]

            log_message(log_file_path, f"助手回复: {assistant_content}")

            assistant_dict = {
                "role": "assistant",
                "content": assistant_content
            }

            assistant_dict["tool_calls"] = []
            if message.tool_calls:  # 如果工具调用不为空，则将工具调用添加到assistant_dict中
                for tool_call in message.tool_calls:
                    assistant_dict["tool_calls"].append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            
            messages.append(assistant_dict)
            
            # 如果有工具调用
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # 执行对应的方法
                    if hasattr(generator_tool, function_name):
                        method = getattr(generator_tool, function_name)
                        result = method(**function_args)
                        
                        # 添加工具调用结果到对话
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": result
                        })
                        print(f"[工具] {function_name}: {result}")
                        log_message(log_file_path, f"[工具] {function_name}: {result}")
            
            # 检查是否已经成功修复
            if generator_tool.generated_items:
                # 获取最后一个成功的项目
                fixed_item = generator_tool.generated_items[-1]
                print(f"\n✓ 成功修复工作流")
                log_message(log_file_path, f"✓ 成功修复工作流")
                return {
                    'question': fixed_item['question'],
                    'workflow': fixed_item['workflow'],
                    'status': 'success'
                }
                
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            log_message(log_file_path, f"API调用失败: {str(e)}")
            break

    # 如果超过最大迭代次数仍未修复成功
    log_message(log_file_path, f"=== 修复失败：超过最大迭代次数 {max_iterations} ===")
    return {
        'question': question,
        'workflow': workflow,
        'status': 'failed'
    }

def process_error_workflows(json_file_path, error_ids, output_file_path, requirements, client, model, max_iterations=80, log_file_path="fix_log.txt"):
    """
    处理JSON文件中的错误工作流
    Args:
        json_file_path: 原始JSON文件路径
        error_ids: 需要修复的问题ID列表
        output_file_path: 输出JSON文件路径
        requirements: 修复要求
        client: OpenAI客户端
        model: 使用的模型
        max_iterations: 最大迭代次数
        log_file_path: 日志文件路径
    """
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 确保数据是列表格式
    if not isinstance(data, list):
        raise ValueError("JSON文件必须包含一个数组")

    # 验证必需字段
    required_fields = ['L1_id', '类型', 'L1_问题', '工作流']
    if data and not all(field in data[0] for field in required_fields):
        raise ValueError(f"JSON文件中的每个对象必须包含以下字段：{required_fields}")

    # 创建结果数据的副本
    result_data = [item.copy() for item in data]

    # 统计信息
    success_count = 0
    failed_ids = []

    # 处理每个错误ID
    for error_id in error_ids:
        print(f"\n处理问题 ID: {error_id}")
        log_message(log_file_path, f"\n=== 处理问题 ID: {error_id} ===")
        
        # 找到对应的项目
        item_index = None
        for i, item in enumerate(data):
            if item.get('L1_id') == error_id:
                item_index = i
                break
        
        if item_index is None:
            print(f"警告：未找到ID为 {error_id} 的问题")
            log_message(log_file_path, f"警告：未找到ID为 {error_id} 的问题")
            continue
        
        item = data[item_index]
        
        question = item['L1_问题']
        workflow = item['工作流']
        
        # 修复工作流
        fix_result = fix_and_validate_workflows(
            client=client,
            model=model,
            question=question,
            workflow=workflow,
            requirements=requirements,
            max_iterations=max_iterations,
            log_file_path=log_file_path
        )
        
        if fix_result['status'] == 'success':
            # 更新JSON数据中的工作流
            result_data[item_index]['工作流'] = fix_result['workflow']
            success_count += 1
            print(f"✓ 成功修复问题 ID: {error_id}")
            log_message(log_file_path, f"✓ 成功修复问题 ID: {error_id}")
        else:
            failed_ids.append(error_id)
            print(f"✗ 未能修复问题 ID: {error_id}")
            log_message(log_file_path, f"✗ 未能修复问题 ID: {error_id}")
        
        # 避免API调用过于频繁
        time.sleep(1)

    # 保存结果到新文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    # 输出统计信息
    print("\n=== 修复完成 ===")
    print(f"总共需要修复: {len(error_ids)} 个问题")
    print(f"成功修复: {success_count} 个问题")
    print(f"修复失败: {len(failed_ids)} 个问题")

    log_message(log_file_path, "\n=== 修复完成 ===")
    log_message(log_file_path, f"总共需要修复: {len(error_ids)} 个问题")
    log_message(log_file_path, f"成功修复: {success_count} 个问题")
    log_message(log_file_path, f"修复失败: {len(failed_ids)} 个问题")

    if failed_ids:
        print(f"\n未成功修复的问题ID: {failed_ids}")
        log_message(log_file_path, f"未成功修复的问题ID: {failed_ids}")
        for failed_id in failed_ids:
            print(f"  - 问题 ID {failed_id} 未成功修复")

    print(f"\n结果已保存到: {output_file_path}")
    log_message(log_file_path, f"结果已保存到: {output_file_path}")


if __name__ == "__main__":
    # 修改为JSON文件路径
    # 可以通过环境变量配置，例如: export WORKFLOW_INPUT_JSON="/path/to/input.json"
    json_file_path = os.environ.get("WORKFLOW_INPUT_JSON", "dataset/realtime-benchmark.json")  # 原始JSON文件路径
    output_file_path = os.environ.get("WORKFLOW_OUTPUT_JSON", "dataset/realtime-benchmark-fix.json")  # 输出JSON文件路径
    log_file_path = os.environ.get("WORKFLOW_LOG_FILE", "dataset/realtime-benchmark-fix_log.log")  # 日志文件路径
    
    # 需要修复的问题ID列表
    error_ids = ['0001'] # 替换为实际的错误ID
    
    # 修复要求
    requirements = """
    请修复工作流代码，确保能够正确获取实时数据......
    """
    
    # 初始化OpenAI客户端
    client = OpenAI(
        base_url="https://api.openai.com/v1", 
        api_key=os.environ["LLM_API_TOKEN"], 
        max_retries=3, 
        timeout=180
    )
    model = 'aws-claude-sonnet-4'
    max_iterations = 40
    
    # 处理错误工作流
    process_error_workflows(
        json_file_path=json_file_path,
        error_ids=error_ids,
        output_file_path=output_file_path,
        requirements=requirements,
        client=client,
        model=model,
        max_iterations=max_iterations,
        log_file_path=log_file_path
    )