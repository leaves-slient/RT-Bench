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

