import json
import json5
import os
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from datetime import datetime
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs
from prompt import *
import time
import asyncio

from tool_file import *
from tool_scholar import *
from tool_python import *
from tool_search import *
from tool_visit import *

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))

TOOL_CLASS = [
    FileParser(),
    Scholar(),
    Visit(),
    Search(),
    PythonInterpreter(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}

import random
import datetime


def current_datetime():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 **kwargs):

        llm = llm or {}
        self.llm_generate_cfg = llm.get("generate_cfg", {}) or {}
        # RT-Bench runs with OpenAI-compatible endpoints; prefer explicit args, then env.
        self.api_url = (llm.get("api_url") or os.environ.get("API_URL") or "").strip()
        self.api_key = (llm.get("api_key") or os.environ.get("API_KEY") or "").strip()
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "600"))
        # Switch between XML tool protocol and OpenAI Function Calling
        self.use_function_calling = (os.environ.get("USE_FUNCTION_CALLING", "true").strip().lower() == "true")

    def _normalize_parameters_schema(self, params):
        """
        Normalize tool parameter schema for OpenAI-compatible Function Calling.
        Supports:
        1) Standard JSON Schema object (pass-through)
        2) Legacy qwen-agent list style:
           [
             {"name":"files","type":"array","array_type":"string","required":True,...}
           ]
        """
        # Standard JSON Schema object
        if isinstance(params, dict):
            return params

        # Legacy list style -> convert to JSON Schema object
        if isinstance(params, list):
            properties = {}
            required = []
            for item in params:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if not name:
                    continue
                t = item.get("type", "string")
                schema = {"type": t}
                if t == "array":
                    schema["items"] = {"type": item.get("array_type", "string")}
                if item.get("description"):
                    schema["description"] = item["description"]
                properties[name] = schema
                if item.get("required"):
                    required.append(name)
            return {"type": "object", "properties": properties, "required": required}

        # Fallback: empty object schema
        return {"type": "object", "properties": {}}

    def _build_openai_tools(self) -> List[dict]:
        """
        Convert our tool objects to OpenAI-compatible tool schemas.
        """
        tools: List[dict] = []
        for tool in TOOL_CLASS:
            try:
                fn = getattr(tool, "function", None)
                if callable(fn):
                    fn = fn()
                if isinstance(fn, dict) and "name" in fn and "parameters" in fn:
                    normalized_fn = dict(fn)
                    normalized_fn["parameters"] = self._normalize_parameters_schema(fn.get("parameters"))
                    tools.append({"type": "function", "function": normalized_fn})
                    continue
            except Exception:
                pass

            # Fallback: use declared attributes if present
            name = getattr(tool, "name", None)
            desc = getattr(tool, "description", "") or ""
            params = getattr(tool, "parameters", None)
            if name and params is not None:
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": desc,
                            "parameters": self._normalize_parameters_schema(params),
                        },
                    }
                )
        return tools

    def _run_tool_calls_openai(self, tool_calls, round_id: int = 0) -> List[dict]:
        """
        Execute OpenAI tool_calls and return tool messages.
        """
        out: List[dict] = []
        for tc in tool_calls or []:
            try:
                fn = tc.function
                tool_name = fn.name
                raw_args = fn.arguments or "{}"
                try:
                    tool_args = json.loads(raw_args)
                except Exception:
                    tool_args = json5.loads(raw_args)
            except Exception as e:
                self._log_tool_io(round_id, "parse_error", "", f"[tool_call parse error] {e}")
                out.append({"role": "tool", "tool_call_id": getattr(tc, "id", ""), "content": f"[tool_call parse error] {e}"})
                continue

            try:
                result = self.custom_call_tool(tool_name, tool_args)
            except Exception as e:
                result = f"[tool execution error] {e}"
            self._log_tool_io(round_id, tool_name, tool_args, result)

            if not isinstance(result, str):
                result = str(result)
            out.append({"role": "tool", "tool_call_id": getattr(tc, "id", ""), "content": result})
        return out

    def _tool_calls_to_debug_text(self, tool_calls) -> str:
        """
        Serialize tool_calls for readable logs across providers.
        """
        if not tool_calls:
            return "[]"
        parts = []
        for tc in tool_calls:
            try:
                fn = tc.function
                name = getattr(fn, "name", "")
                arguments = getattr(fn, "arguments", "")
                parts.append(json.dumps({"id": getattr(tc, "id", ""), "name": name, "arguments": arguments}, ensure_ascii=False))
            except Exception:
                parts.append(str(tc))
        return "[" + ", ".join(parts) + "]"

    def _serialize_tool_calls_for_messages(self, tool_calls) -> List[dict]:
        """
        Convert SDK tool call objects to message-safe OpenAI format.
        """
        serialized: List[dict] = []
        for tc in tool_calls or []:
            try:
                fn = tc.function
                serialized.append(
                    {
                        "id": getattr(tc, "id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(fn, "name", ""),
                            "arguments": getattr(fn, "arguments", "") or "{}",
                        },
                    }
                )
            except Exception:
                continue
        return serialized

    def _has_parsable_tool_calls(self, tool_calls) -> bool:
        """
        Return True only if at least one tool call can be parsed.
        """
        if not tool_calls:
            return False
        for tc in tool_calls:
            try:
                fn = tc.function
                _ = fn.name
                raw_args = fn.arguments or "{}"
                try:
                    json.loads(raw_args)
                except Exception:
                    json5.loads(raw_args)
                return True
            except Exception:
                continue
        return False

    def _log_tool_io(self, round_id: int, tool_name: str, tool_args, tool_result) -> None:
        """
        Print full tool input/output for debugging in both XML and Function Calling modes.
        """
        try:
            args_text = tool_args if isinstance(tool_args, str) else json.dumps(tool_args, ensure_ascii=False)
        except Exception:
            args_text = str(tool_args)
        try:
            result_text = tool_result if isinstance(tool_result, str) else json.dumps(tool_result, ensure_ascii=False)
        except Exception:
            result_text = str(tool_result)

        print(f"[TOOL_CALL][round={round_id}] name={tool_name}")
        print(f"[TOOL_ARGS][round={round_id}] {args_text}")
        print(f"[TOOL_RESULT_START][round={round_id}] name={tool_name}")
        print(result_text)
        print(f"[TOOL_RESULT_END][round={round_id}] name={tool_name}")

    def _safe_json_dump(self, obj) -> str:
        """
        Best-effort JSON dump for debug logging.
        """
        try:
            return json.dumps(obj, ensure_ascii=False, default=str, indent=2)
        except Exception:
            return str(obj)

    def _log_model_input(self, round_id: int, mode: str, model_input) -> None:
        print(f"[MODEL_INPUT_START][round={round_id}][mode={mode}]")
        print(self._safe_json_dump(model_input))
        print(f"[MODEL_INPUT_END][round={round_id}][mode={mode}]")

    def _log_model_output(self, round_id: int, mode: str, model_output) -> None:
        print(f"[MODEL_OUTPUT_START][round={round_id}][mode={mode}]")
        print(self._safe_json_dump(model_output))
        print(f"[MODEL_OUTPUT_END][round={round_id}][mode={mode}]")

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content
    
    def call_server(self, msgs, planning_port=None, max_tries=10):
        """
        Call an OpenAI-compatible chat completions endpoint.
        RT-Bench provides API_URL/API_KEY via env and also passes them as CLI args.
        """
        api_key = self.api_key or os.environ.get("API_KEY", "")
        api_base = self.api_url or os.environ.get("API_URL", "")
        if not api_base:
            raise ValueError("Missing API_URL (set env API_URL or pass --api_url)")

        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=self.timeout,
            max_retries=3,
        )

        base_sleep_time = 1 
        for attempt in range(max_tries):
            try:
                print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
                create_kwargs = dict(
                    model=self.model,
                    messages=msgs,
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    max_tokens=10000,
                    presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1),
                )
                if self.use_function_calling:
                    # OpenAI tool calling (Function Calling)
                    create_kwargs["tools"] = self._build_openai_tools()
                    create_kwargs["tool_choice"] = "auto"
                else:
                    # Legacy XML tool protocol
                    create_kwargs["stop"] = ["\n<tool_response>", "<tool_response>"]

                chat_response = client.chat.completions.create(**create_kwargs)
                msg = chat_response.choices[0].message
                content = msg.content or ""

                # OpenRouter provides API calling. If you want to use OpenRouter, you need to uncomment line 89 - 90.
                # reasoning_content = "<think>\n" + chat_response.choices[0].message.reasoning.strip() + "\n</think>"
                # content = reasoning_content + content                
                
                if content and content.strip():
                    print("--- Service call successful, received a valid response ---")
                    # In function calling mode, content may be empty when tool_calls exist.
                    # We return the raw content; caller inspects tool_calls separately via last_response.
                    return content.strip()
                else:
                    print(f"Warning: Attempt {attempt + 1} received an empty response.")

            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30) 
                
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")
        
        return f"vllm server error!!!"

    def count_tokens(self, messages) -> Optional[int]:
        """
        Best-effort token count. We avoid requiring a local HF tokenizer path because
        RT-Bench typically uses remote OpenAI-compatible endpoints.
        """
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            text = ""
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "")
                text += f"{role}: {content}\n"
            return len(encoding.encode(text))
        except Exception:
            return None

    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        self.model=model
        try:
            question = data['item']['question']
        except: 
            raw_msg = data['item']['messages'][1]["content"] 
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg 

        start_time = time.time()
        answer = data['item']['answer']
        self.user_prompt = question
        # Keep XML and Function Calling prompts separated.
        system_prompt = SYSTEM_PROMPT_FC if self.use_function_calling else SYSTEM_PROMPT
        cur_date = current_datetime()
        system_prompt = system_prompt + str(cur_date)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        while num_llm_calls_available > 0:
            # Check whether time is reached
            if time.time() - start_time > 150 * 60:  # 150 minutes in seconds
                prediction = 'No answer found after 2h30mins'
                termination = 'No answer found after 2h30mins'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result
            round += 1
            num_llm_calls_available -= 1
            if self.use_function_calling:
                # Function Calling loop: run tools until model returns a normal assistant message.
                api_key = self.api_key or os.environ.get("API_KEY", "")
                api_base = self.api_url or os.environ.get("API_URL", "")
                client = OpenAI(api_key=api_key, base_url=api_base, timeout=self.timeout, max_retries=3)
                create_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "tools": self._build_openai_tools(),
                    "tool_choice": "auto",
                    "temperature": self.llm_generate_cfg.get('temperature', 0.6),
                    "top_p": self.llm_generate_cfg.get('top_p', 0.95),
                    "max_tokens": 10000,
                    "presence_penalty": self.llm_generate_cfg.get('presence_penalty', 1.1),
                }
                self._log_model_input(round_id=round, mode="function_calling", model_input=create_kwargs)
                chat_response = client.chat.completions.create(**create_kwargs)
                msg = chat_response.choices[0].message
                content = (msg.content or "").strip()
                tool_calls = getattr(msg, "tool_calls", None)
                serialized_tool_calls = self._serialize_tool_calls_for_messages(tool_calls)
                self._log_model_output(
                    round_id=round,
                    mode="function_calling",
                    model_output={
                        "message": msg.model_dump() if hasattr(msg, "model_dump") else str(msg),
                        "usage": chat_response.usage.model_dump() if getattr(chat_response, "usage", None) and hasattr(chat_response.usage, "model_dump") else str(getattr(chat_response, "usage", "")),
                    },
                )
                print(f"Round {round}: content={content!r}")
                print(f"Round {round}: tool_calls={self._tool_calls_to_debug_text(tool_calls)}")
                assistant_msg = {"role": "assistant", "content": content}
                if serialized_tool_calls:
                    # OpenAI-compatible history requires tool_calls on assistant message.
                    assistant_msg["tool_calls"] = serialized_tool_calls
                messages.append(assistant_msg)

                # Prefer tool_calls in function-calling mode. If empty/unparsable, fallback to content.
                if self._has_parsable_tool_calls(tool_calls):
                    tool_msgs = self._run_tool_calls_openai(tool_calls, round_id=round)
                    messages.extend(tool_msgs)
                    continue
            else:
                # Legacy XML tool protocol
                xml_request_payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.llm_generate_cfg.get('temperature', 0.6),
                    "top_p": self.llm_generate_cfg.get('top_p', 0.95),
                    "max_tokens": 10000,
                    "presence_penalty": self.llm_generate_cfg.get('presence_penalty', 1.1),
                    "stop": ["\n<tool_response>", "<tool_response>"],
                }
                self._log_model_input(round_id=round, mode="xml", model_input=xml_request_payload)
                content = self.call_server(messages)
                self._log_model_output(round_id=round, mode="xml", model_output={"content": content})
                tool_call_preview = ""
                if '<tool_call>' in content and '</tool_call>' in content:
                    try:
                        tool_call_preview = content.split('<tool_call>')[1].split('</tool_call>')[0].strip()
                    except Exception:
                        tool_call_preview = "<parse_error>"
                print(f"Round {round}: content={content!r}")
                print(f"Round {round}: tool_calls={tool_call_preview!r}")
                if '<tool_response>' in content:
                    pos = content.find('<tool_response>')
                    content = content[:pos]
                messages.append({"role": "assistant", "content": content.strip()})
                if '<tool_call>' in content and '</tool_call>' in content:
                    tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                    tool_name = ""
                    tool_args = {}
                    try:
                        if "python" in tool_call.lower():
                            try:
                                code_raw=content.split('<tool_call>')[1].split('</tool_call>')[0].split('<code>')[1].split('</code>')[0].strip()
                                result = TOOL_MAP['PythonInterpreter'].call(code_raw)
                                self._log_tool_io(round, "PythonInterpreter", {"code": code_raw}, result)
                            except:
                                result = "[Python Interpreter Error]: Formatting error."
                                self._log_tool_io(round, "PythonInterpreter", {"code": ""}, result)

                        else:
                            tool_call = json5.loads(tool_call)
                            tool_name = tool_call.get('name', '')
                            tool_args = tool_call.get('arguments', {})
                            result = self.custom_call_tool(tool_name, tool_args)
                            self._log_tool_io(round, tool_name, tool_args, result)

                    except:
                        result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                        self._log_tool_io(round, tool_name or "parse_error", tool_args, result)
                    result = "<tool_response>\n" + result + "\n</tool_response>"
                    messages.append({"role": "user", "content": result})
            if '<answer>' in content and '</answer>' in content:
                termination = 'answer'
                break
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            max_tokens = 110 * 1024
            token_count = self.count_tokens(messages)
            if token_count is not None:
                print(f"round: {round}, token count: {token_count}")

            if token_count is not None and token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")
                
                messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"
                content = self.call_server(messages)
                messages.append({"role": "assistant", "content": content.strip()})
                if '<answer>' in content and '</answer>' in content:
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result

        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination
        }
        return result

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in TOOL_MAP:
            tool_args["params"] = tool_args
            if "python" in tool_name.lower():
                # Function-calling passes {"code": "..."}; XML path may pass raw code string.
                if isinstance(tool_args, dict):
                    code = tool_args.get("code", "")
                else:
                    code = tool_args
                result = TOOL_MAP['PythonInterpreter'].call(code)
            elif tool_name == "parse_file":
                params = {"files": tool_args["files"]}
                
                raw_result = asyncio.run(TOOL_MAP[tool_name].call(params, file_root_path="./eval_data/file_corpus"))
                result = raw_result

                if not isinstance(raw_result, str):
                    result = str(raw_result)
            else:
                raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
                result = raw_result
            return result

        else:
            return f"Error: Tool {tool_name} not found"
