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
# API Token should be set via environment variable LLM_API_TOKEN
# Example: export LLM_API_TOKEN="your-token-here"
if "LLM_API_TOKEN" not in os.environ:
    raise ValueError("Please set LLM_API_TOKEN environment variable")


def openai_tool(func: Callable) -> Callable:
    """
    装饰器，将带 Google 风格 docstring 的函数转换为 OpenAI 工具 schema，并附加在函数的 __tool_schema__ 属性上。
    """
    sig = inspect.signature(func)
    annotations = get_type_hints(func)

    doc = parse(func.__doc__ or "")
    description_parts = [doc.short_description or ""]
    if doc.long_description:
        description_parts.append(doc.long_description)
    description = "\n".join(p.strip() for p in description_parts if p)

    param_docs = {p.arg_name: p.description for p in doc.params}

    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object"
    }

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for name, param in sig.parameters.items():
        if name == 'self':
            continue
            
        type_hint = annotations.get(name, str)
        schema_type = type_mapping.get(type_hint, "string")

        properties[name] = {
            "type": schema_type,
            "description": param_docs.get(name, f"No description for `{name}` provided.")
        }

        if param.default == inspect.Parameter.empty:
            required.append(name)

    func.__tool_schema__ = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }

    return func

class WorkflowTool:
    def __init__(self, client, model):
        self.thought = ''
        self.namespace = self._init_namespace()
        self.generated_items = []
        self.code = ''
        self.client = client
        self.model = model
        self.image_path = './screenshot.png' #缓存screenshot的图片路径
        self.output_content = ''  #缓存code_interpreter的print内容
    
    def _init_namespace(self):
        """初始化namespace，预导入常用库"""
        namespace = {}
        # 预导入常用库
        imports = """
import requests
import json
import time
from datetime import datetime, timedelta
import re
"""
        exec(imports, namespace)
        return namespace

    @openai_tool
    def think(self, thought: str) -> str:
        """
        进行思考，思考。

        Args:
            thought (str): 对当前流程的思考。

        Returns:
            str: 提示当前思考已完成。
        """
        self.thought = thought
        return f"思考已完成，思考结果为：{thought}"

    @openai_tool
    def code_interpreter(self, code: str, is_workflow: bool = False) -> str:
        """
        执行给定 Python 代码，并返回执行结果成功的提示或执行错误的报错信息。

        Args:
            code (str): 可执行的 Python 代码字符串，重要信息使用print输出，代码请不要包含任何try except，直接让错误暴露出来。如果是用来获取问题答案的工作流代码，需要将答案信息用函数return，然后在主函数中print出来。代码里使用from datetime import；datetime now = datetime.now()获取时间，不可预设一个固定时间。
            is_workflow (bool): 是否是用来获取问题答案的工作流代码，默认False。

        Returns:
            str: 执行成功提示或详细报错信息。
        """
        try:
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                exec(code, self.namespace)
                # 获取捕获的输出
            output_content = output_buffer.getvalue()
            if is_workflow:
                self.code = code
                self.output_content = output_content
                return f"代码执行成功。这是用来获取问题答案的工作流代码，代码执行过程中产生的答案信息为：{output_content}。\n请调用think工具思考是否符合预期，如果符合预期，则调用check_output_content工具进行最终检查，否则请调用think工具思考问题原因，并修改代码。"
            return f"代码执行成功。代码执行过程中产生的重要信息：{output_content}。"
        except Exception as e:
            return f"执行代码失败，错误信息:\n{e}\n请调用`think(thought)`进行深度思考。"
        finally:
            output_buffer.close()

    @openai_tool
    def check_output_content(self, question: str, url: str) -> str:
        """
        最终检查针对问题获取答案的工作流代码执行过程中产生的答案信息是否符合预期。对url进行截图缓存，将前一步生成的答案信息和截图进行校验。

        Args:
            question (str): 问题。
            url (str): 需要截图的url。

        Returns:
            str: 检查结果。
        """
        from playwright.sync_api import sync_playwright
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(url, wait_until="domcontentloaded", timeout=60000)
                    # 适当等待，确保动态内容渲染完成
                    page.wait_for_timeout(10000)
                        # 可选：向下滚动触发懒加载（确保全页图片都加载）
                    page.evaluate("""
                    () => new Promise(resolve => {
                        let total = 0; const step = 1000;
                        const timer = setInterval(() => {
                            const sh = document.body.scrollHeight;
                            window.scrollBy(0, step);
                            total += step;
                            if (total + window.innerHeight >= sh) {
                                clearInterval(timer);
                                resolve();
                            }
                        }, 200);
                    })
                    """)

                    page.screenshot(path=self.image_path, full_page=True, timeout=120000)
                    browser.close()
                    break
            except Exception as e:
                print(f"第 {attempt + 1} 次截图尝试失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)  # 等待3秒后重试
                    continue
                else:
                    # 所有尝试都失败，返回错误信息
                    return f"截图失败，无法进行最终检查。错误: {str(e)}，建议重试一次。"

        with open(self.image_path, 'rb') as f:
            image_data = f.read()

        # 构建消息
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的答案验证助手，给其他助手提供指引。你将获得一张图片、一个针对图片问题和相应的答案，请检查答案是否能经过图片验证，如果经过验证，则提示下一个助手可以调用save_generated_item工具保存，否则提示答案与图片不符，并要求重新检查代码"
            },
            {
            "role": "user",
            "content": [
                {"type": "text", "text": f"问题：{question}\n答案：{self.output_content}\n图片如图所示"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image_data).decode()}"}}
            ]}
        ]
        
        # 调用模型
        response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        message = response.choices[0].message
        response_content = message.content or ""  # 安全访问，避免 None
        return f"最终检查完成，{response_content}"
    
    @openai_tool
    def check_html_content(self, html_check_code: str) -> str:
        """
        执行给定的Python内容检查代码，检查网页中内容或者检查网页中是否包含所需信息，并返回检查结果，重要信息使用print输出，代码不要包含任何try except。

        Args:
            html_check_code (str): 检查网页中内容或者检查网页中是否包含所需信息的Python代码字符串，重要信息使用print输出，代码不要包含任何try except。建议使用playwright库，with sync_playwright() as p:    browser = p.chromium.launch(headless=True)  page = browser.new_page()   page.goto(url, wait_until="domcontentloaded", timeout=60000)# 使用domcontentloaded防止超时    page.wait_for_timeout(3000)# 额外等待，确保动态内容渲染完成。

        Returns:
            str: 检查结果。
        """
        try:
            output_buffer = io.StringIO()
            # 重定向stdout到我们的buffer
            with redirect_stdout(output_buffer):
                exec(html_check_code, self.namespace)
            output_content = output_buffer.getvalue()
            return f"HTML 内容检查完成。重要信息为：{output_content}"
        except Exception as e:
            return f"HTML 内容检查失败，错误信息:\n{e}\n。"
        finally:
            output_buffer.close()

    @openai_tool
    def save_generated_item(self, question: str) -> str:
        """
        保存生成的问题和经过校验的可执行的工作流代码。此过程紧跟在code_interpreter工具调用之后。

        Args:
            question (str): 生成的新问题。

        Returns:
            str: 保存成功的提示。
        """
        self.generated_items.append({
            "question": question,
            "workflow": self.code
        })
        return f"已保存新问题和工作流。当前已生成 {len(self.generated_items)} 个项目。"
    
    def get_openai_tools(self):
        """获取所有工具的schema列表"""
        tools = []
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, '__tool_schema__'):
                tools.append(method.__tool_schema__)
        return tools

def generate_and_validate_workflows(client, model, sample_question, sample_workflow, requirements, max_iterations=80):
    """
    调用模型生成新的问题和工作流，并验证其可执行性
    """
    generator_tool = WorkflowTool(client, model)
    system_prompt = f"""你是一个专业的代码生成和验证助手。你的任务是：
1. 基于给定的网站，生成和时间相关的问题，以及对应的获取问题答案的工作流代码
3. 使用code_interpreter工具测试代码是否可执行，如果执行成功，则调用check_output_content工具进行最终检查，否则调用think工具思考问题原因，并修改代码，直到代码可执行为止
4. 使用save_generated_item工具保存最终可执行的问题和答案获取工作流代码
6. 关键结果需要return，main函数中使用print打印return的内容

要求：
- 问题需要是实时性问题，跟时间相关，通常和过去时间相关，除非原问题与未来时间相关
- 代码里使用from datetime import；datetime now = datetime.now()获取时间，不可预设一个固定时间
- 所有码中不要包含任何try except，不要包含任何兜底策略。code_interpreter会使用exec执行代码，希望在代码不符合要求时，能直接报错。
- 工作流代码类似于爬虫代码，尽量仿照示例的代码样式，使用playwright和markdownify。
- 工作流代码必须经过测试确保可执行
- 成功执行工作流代码后用save_generated_item保存

示例：
实时性问题：{sample_question}

相应的最终工作流代码：
```python
{sample_workflow}
```
"""
    user_prompt = f"""{requirements}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # print(messages)
    # 最多尝试30轮对话
    for _ in range(max_iterations):
        print(f"第 {_ + 1} 轮对话...")
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
            
            # 检查是否已经生成足够的项目
            if len(generator_tool.generated_items) >= 10:
                print(f"\n✓ 成功生成 {len(generator_tool.generated_items)} 个问题和工作流")
                break
                
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            break

    return generator_tool.generated_items

def process(client, model, csv_file_path, output_file_path, requirements, max_iterations=80):
    """
    使用pandas处理CSV文件，生成并验证新的问题和工作流
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    # 获取列名
    question_col = df.columns[0]
    workflow_col = df.columns[1]

    # 存储所有结果
    all_results = []

    # 处理每一行
    for idx, row in df.iterrows():
        example_question = str(row[question_col]).strip()
        example_workflow = str(row[workflow_col]).strip()
        
        # 跳过空值
        if pd.isna(example_question) or pd.isna(example_workflow):
            continue
        
        # 生成并验证新的问题和工作流
        generated_items = generate_and_validate_workflows(client, model, example_question, example_workflow, requirements, max_iterations)
        
        if generated_items:
            for item in generated_items:
                all_results.append({
                    '原始问题': example_question,
                    '新问题': item['question'],
                    '新工作流': item['workflow'],
                    '状态': '已验证可执行'
                })
                print(f"  ✓ 生成并验证的新问题: {item['question']}")
        else:
            print(f"  ✗ 未能生成有效的问题和工作流")
        # 避免API调用过于频繁
        time.sleep(1)
    # 创建结果DataFrame并保存
    if all_results:
        result_df = pd.DataFrame(all_results)
        result_df.to_csv(output_file_path, index=False, encoding='utf-8')
        print(f"\n处理完成！结果已保存到: {output_file_path}")
        print(f"共生成并验证 {len(all_results)} 个新的问题和工作流")
        
        # 显示统计信息
        print("\n统计信息:")
        print(f"- 成功验证的工作流: {len(all_results)}")
        print(f"- 平均每个原始问题生成: {len(all_results) / len(df):.1f} 个新问题")
    else:
        print("\n没有生成任何有效结果")

    return all_results

if __name__ == "__main__":
    # 可以通过环境变量配置路径，例如: export EXAMPLE_FILE="/path/to/input.csv"
    example_file_path = os.environ.get("EXAMPLE_FILE", "input.csv")
    output_file_path = os.environ.get("OUTPUT_FILE", "output.csv")
    # 初始化OpenAI客户端
    # 注意：内网地址已移除，请使用环境变量配置 API URL
    # 可以通过环境变量配置 API URL，例如: export OPENAI_API_URL="https://api.openai.com/v1"
    api_url = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1")
    client = OpenAI(base_url=api_url, api_key=os.environ["LLM_API_TOKEN"], max_retries=3, timeout=180)
    model = 'google-claude-sonnet-4'
    requirements = """
针对......信息，问几个实时性问题，
https://www.heavens-above.com/SolarEclipses.aspx?lat=31.2313&lng=121.47&loc=%e4%b8%8a%e6%b5%b7%e5%b8%82&alt=0&tz=ChST 记录了一些日蚀信息，可能可以问“根据heavens-above官网的信息，未来半年有几次日蚀”；
https://www.heavens-above.com/Asteroids.aspx?lat=31.2313&lng=121.47&loc=%e4%b8%8a%e6%b5%b7%e5%b8%82&alt=0&tz=ChST 记录了一些小行星信息，可能可以问“根据heavens-above官网的信息，目前能观测到的最亮的小行星是哪颗”；
请生成5-10个实时性问题和工作流，并确保每个工作流都是可执行的。
"""
    max_iterations = 80
    results = process(client, model, example_file_path, output_file_path, requirements, max_iterations)