#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按「单题、多模型并发」方式运行 DeepResearch / run_multi_react.py 的控制脚本。

核心逻辑（基于用户需求）：
1. 读取 models_config.yaml，确定要运行的模型列表和 L1/L2/L3 数据集路径。
2. 先对所选模型做「可用性检查」（基于 API URL 的 TCP 连接探测），如果有不可用的模型则定期重试，直到全部可用。
3. 对每个 level （L1/L2/L3）和该 level 下的每一道题：
   - 再次确认所有模型当前都可用（如果有模型掉线则等待恢复）。
   - 并发启动多个子进程，每个模型各跑一次 run_multi_react.py，
     但通过 total_splits / worker_split 参数，保证每个子进程只处理当前这 1 道题。
   - 等所有模型都完成当前题目后，异步调用 get_standard_answers.sh 生成 / 更新标准答案文件
     （不阻塞下一题的模型推理）。
   - 然后立即进入下一道题。

说明：
- 为了尽量复用现有逻辑，本脚本不会修改 run_multi_react.py，而是通过其已有的
  total_splits / worker_split 分片逻辑来做到「单题运行」。
- 调用 get_standard_answers.sh 是逐题调用的，逻辑最简单但可能比较耗时；
  如需优化，可后续自行改成「先预先计算好标准答案，再这里按题目做匹配」。
"""

import os
import sys
import time
import json
import socket
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any

import yaml
import json as _json
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import importlib


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "models_config.yaml")
GET_STANDARD_ANSWERS_SH = os.path.join(SCRIPT_DIR, "get_standard_answers.sh")


# ========================== 工具函数 ========================== #

# 线程安全的打印锁
_print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """线程安全的打印封装。"""
    with _print_lock:
        print(*args, **kwargs, flush=True)


def load_config(config_file: str) -> Dict[str, Any]:
    """加载 YAML 配置文件。"""
    if not os.path.exists(config_file):
        safe_print(f"❌ 错误: 找不到配置文件: {config_file}")
        sys.exit(1)

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        safe_print(f"❌ 错误: 无法读取配置文件 {config_file}: {e}")
        sys.exit(1)


def validate_config(config: Dict[str, Any]) -> None:
    """验证配置文件的关键字段。"""
    required_keys = ["shared_keys", "datasets", "output_base", "models", "run_settings"]
    for k in required_keys:
        if k not in config:
            safe_print(f"❌ 错误: 配置文件中缺少必需的键: {k}")
            sys.exit(1)

    if "SERPER_KEY_ID" not in config["shared_keys"]:
        safe_print("❌ 错误: 配置文件中缺少 SERPER_KEY_ID")
        sys.exit(1)
    if "JINA_API_KEYS" not in config["shared_keys"]:
        safe_print("❌ 错误: 配置文件中缺少 JINA_API_KEYS")
        sys.exit(1)


def get_selected_models(config: Dict[str, Any]) -> List[str]:
    """获取启用并被选择的模型名称列表。"""
    models = config["models"]
    run_settings = config["run_settings"]
    selected = run_settings.get("selected_models", [])

    # 如果没有指定模型，则返回所有 enabled=true 的模型
    if not selected:
        enabled_models = [name for name, m in models.items() if m.get("enabled", False)]
        return enabled_models

    # 如果指定了模型，则只使用存在且启用的
    valid_models: List[str] = []
    for name in selected:
        if name not in models:
            safe_print(f"⚠️  警告: 模型 '{name}' 不存在，跳过")
            continue
        if not models[name].get("enabled", False):
            safe_print(f"⚠️  警告: 模型 '{name}' 未启用 (enabled: false)，跳过")
            continue
        valid_models.append(name)
    return valid_models


def get_selected_levels(config: Dict[str, Any]) -> List[str]:
    """获取要运行的问题级别列表（L1/L2/L3）。"""
    run_settings = config["run_settings"]
    selected = run_settings.get("selected_levels", ["L1", "L2", "L3"])
    datasets = config["datasets"]

    valid_levels: List[str] = []
    for level in selected:
        if level not in datasets:
            safe_print(f"⚠️  警告: 问题级别 '{level}' 在 datasets 中没有配置路径，跳过")
            continue
        ds_path = datasets[level]
        if not os.path.exists(ds_path):
            safe_print(f"⚠️  警告: 数据集文件不存在: {ds_path}，跳过级别 '{level}'")
            continue
        valid_levels.append(level)
    return valid_levels


def is_model_available(api_url: str, timeout: float = 5.0) -> bool:
    """
    通过 TCP 连接 host:port 判断模型对应服务是否可用。
    不依赖具体 HTTP 接口，只要端口能连通就视为“可用”。
    """
    try:
        # 简单解析 URL
        if "://" not in api_url:
            # 如果没有 scheme，默认 http
            api_url = "http://" + api_url
        parsed = urlparse(api_url)
        host = parsed.hostname
        port = parsed.port
        if host is None:
            return False
        if port is None:
            port = 443 if parsed.scheme == "https" else 80

        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
    except Exception:
        return False


def _build_openai_compatible_chat_endpoint(base_url: str) -> str:
    """
    构造 OpenAI 兼容的 chat completions 接口地址。
    传入的 base_url 可能以 /v1 结尾，也可能已经是完整前缀。
    """
    base = (base_url or "").rstrip("/")
    return f"{base}/chat/completions"


def _http_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
    """
    使用标准库发起 JSON POST 请求，返回解析后的 JSON。
    """
    body = _json.dumps(payload).encode("utf-8")
    req = Request(url=url, data=body, headers=headers, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        resp_body = resp.read()
        return _json.loads(resp_body.decode("utf-8"))


def probe_model_via_provider(model_conf: Dict[str, Any]) -> Dict[str, Any]:
    """
    通过 llm_provider 对应的接口发起一次最小化 Chat Completions 探测。
    使用与运行阶段（react_agent.py）完全一致的 OpenAI 客户端初始化方式。
    返回结构：
      {
        "ok": bool,
        "message_preview": str,   # 截断后的模型回复
        "raw": Dict[str, Any],    # 原始返回（可能已截断字段）
        "error": str | None
      }
    """
    api_url = model_conf.get("api_url", "")
    api_key = model_conf.get("api_key", "")
    model_name = model_conf.get("model_name", "")
    provider = (model_conf.get("llm_provider", "openai") or "openai").lower()

    if not api_url or not api_key or not model_name:
        return {
            "ok": False,
            "message_preview": "",
            "raw": {},
            "error": "missing api_url / api_key / model_name",
        }

    # 与运行阶段一致：根据 provider 选择客户端（与 react_agent.py 中的逻辑保持一致）
    if provider in ("openai", "doubao"):
        try:
            # 与运行阶段一致：根据 provider 选择客户端
            # 运行阶段从环境变量 LLM_PROVIDER 读取，这里从配置文件的 llm_provider 读取（值会通过环境变量传递给运行阶段）
            if provider == "doubao":
                # 豆包使用标准 OpenAI 客户端，完全兼容 OpenAI API
                try:
                    # 与运行阶段一致：使用 openai.OpenAI
                    OpenAI = importlib.import_module("openai").OpenAI
                except Exception as e:
                    return {
                        "ok": False,
                        "message_preview": "",
                        "raw": {},
                        "error": f"ImportError OpenAI: {e}",
                    }
                # 与运行阶段一致：使用相同的初始化参数（timeout 在探测时使用较短值，运行阶段使用 600.0）
                client = OpenAI(api_key=api_key, base_url=api_url, timeout=30.0)
                # 豆包使用基础参数（与运行阶段一致，探测时不使用 top_p, presence_penalty）
                # 使用明确的提示词进行健康检查，要求模型返回特定内容
                create_kwargs = {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": "Please reply with the word 'ok' to confirm you are working."},
                    ],
                    "temperature": 0,
                    "max_tokens": 20,
                }
            else:
                # OpenAI 或其他兼容 OpenAI API 的模型
                try:
                    # 与运行阶段一致：使用 openai.OpenAI
                    OpenAI = importlib.import_module("openai").OpenAI
                except Exception as e:
                    return {
                        "ok": False,
                        "message_preview": "",
                        "raw": {},
                        "error": f"ImportError OpenAI: {e}",
                    }
                # 与运行阶段一致：使用相同的初始化参数（timeout 在探测时使用较短值，运行阶段使用 600.0）
                client = OpenAI(api_key=api_key, base_url=api_url, timeout=30.0)
                # 健康检查使用最小参数集，运行阶段会使用更多参数（top_p, logprobs, presence_penalty）
                # 使用明确的提示词进行健康检查，要求模型返回特定内容
                create_kwargs = {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": "Please reply with the word 'ok' to confirm you are working."},
                    ],
                    "temperature": 0,
                    "max_tokens": 20,
                }

            # 与运行阶段一致：使用相同的 API 调用方式
            resp = client.chat.completions.create(**create_kwargs)
            
            # 详细检查响应结构
            content = ""
            reasoning = ""
            refusal = None
            error_detail = ""
            
            try:
                # 检查响应基本结构
                if not hasattr(resp, "choices") or not resp.choices:
                    error_detail = "response has no choices"
                else:
                    choice = resp.choices[0]
                    if not hasattr(choice, "message"):
                        error_detail = "choice has no message"
                    else:
                        msg = choice.message
                        # 尝试获取 content
                        if hasattr(msg, "content"):
                            content = (msg.content or "").strip()
                        # 尝试获取 reasoning（某些模型使用此字段）
                        if hasattr(msg, "reasoning"):
                            reasoning = (getattr(msg, "reasoning", "") or "").strip()
                        # 检查 refusal 字段（某些模型可能通过此字段表示拒绝）
                        if hasattr(msg, "refusal"):
                            refusal = getattr(msg, "refusal", None)
                        
                        # 如果 content 和 reasoning 都为空，检查是否有 refusal
                        if not content and not reasoning:
                            if refusal is not None:
                                error_detail = f"model refused request, refusal: {refusal}"
                            else:
                                # 对于健康检查：如果 API 调用成功且没有 refusal，即使 content 为空也认为可用
                                # 因为至少说明 API 端点是工作的（某些模型可能在某些情况下返回空内容但仍然是可用的）
                                # 但为了更严格，我们仍然要求有内容返回
                                msg_attrs = [attr for attr in dir(msg) if not attr.startswith("_")]
                                error_detail = f"empty content/reasoning, message attrs: {msg_attrs}, message repr: {repr(msg)}"
            except Exception as e:
                error_detail = f"error parsing response: {e}"
            
            effective = content if content else reasoning
            preview = (effective[:120] + ("..." if len(effective) > 120 else "")) if effective else ""
            # 判断模型是否可用：
            # 1. 如果有有效内容（content 或 reasoning），认为可用
            # 2. 如果 content 为空但没有 refusal，也认为可用（API 工作正常，只是返回了空内容）
            # 3. 如果有 refusal，认为不可用
            ok = bool(effective) or (not content and not reasoning and refusal is None)
            
            # 将响应尽量转换为可序列化字典（仅用于调试打印，不做严格保证）
            raw_dict = {}
            try:
                if hasattr(resp, "model_dump"):
                    raw_dict = resp.model_dump()
                elif hasattr(resp, "dict"):
                    raw_dict = resp.dict()
                else:
                    # 尝试手动提取关键信息
                    raw_dict = {
                        "id": getattr(resp, "id", None),
                        "object": getattr(resp, "object", None),
                        "created": getattr(resp, "created", None),
                        "model": getattr(resp, "model", None),
                        "choices_count": len(resp.choices) if hasattr(resp, "choices") else 0,
                        "repr": str(resp)[:500]  # 限制长度
                    }
            except Exception as e:
                raw_dict = {"repr": str(resp)[:500], "parse_error": str(e)}

            # 生成错误信息
            final_error = None
            if not ok:
                if refusal is not None:
                    final_error = f"model refused request: {refusal}"
                elif not content and not reasoning:
                    final_error = error_detail or "empty content/reasoning from provider (API call succeeded but no content returned)"
                else:
                    final_error = error_detail or "unknown error"
            
            return {
                "ok": ok,
                "message_preview": preview,
                "raw": raw_dict,
                "error": final_error,
            }
        except Exception as e:
            return {
                "ok": False,
                "message_preview": "",
                "raw": {},
                "error": str(e),
            }

    # 未知 provider：判为不可用
    return {
        "ok": False,
        "message_preview": "",
        "raw": {},
        "error": f"unsupported provider: {provider}",
    }


def wait_until_all_models_available(
    config: Dict[str, Any],
    selected_models: List[str],
    check_interval: float = 15.0,
) -> None:
    """
    轮询所有选中模型的可用性，直到全部可用才返回。
    可用性定义：配置完整且能通过 llm_provider 的最小化 Chat Completions 探测。
    """
    models_conf = config["models"]
    verbose = bool(config.get("run_settings", {}).get("verbose", False))
    while True:
        unavailable: List[str] = []
        probe_logs: List[str] = []
        for name in selected_models:
            mconf = models_conf[name]
            api_url = mconf.get("api_url", "")
            api_key = mconf.get("api_key", "")
            if not api_url or not api_key:
                unavailable.append(name)
                probe_logs.append(f" - {name}: missing api_url/api_key")
                continue

            # 基于 provider 的真实推理探测
            probe = probe_model_via_provider(mconf)
            if probe["ok"]:
                msg = probe.get("message_preview", "")
                probe_logs.append(f" - {name}: OK, reply preview: {msg!r}")
            else:
                unavailable.append(name)
                error_msg = probe.get('error', 'unknown')
                api_url = mconf.get("api_url", "")
                provider = (mconf.get("llm_provider", "openai") or "openai").lower()
                model_name = mconf.get("model_name", "")
                # 如果失败，尝试打印更多调试信息
                raw_info = probe.get('raw', {})
                if raw_info:
                    # 提取关键调试信息
                    debug_info = []
                    if 'choices_count' in raw_info:
                        debug_info.append(f"choices_count={raw_info['choices_count']}")
                    if 'model' in raw_info:
                        debug_info.append(f"model={raw_info['model']}")
                    if 'repr' in raw_info and len(str(raw_info['repr'])) < 200:
                        debug_info.append(f"resp_repr={raw_info['repr']}")
                    if debug_info:
                        error_msg = f"{error_msg} ({', '.join(debug_info)})"
                probe_logs.append(
                    f" - {name}: FAIL\n"
                    f"    provider: {provider}\n"
                    f"    model_name: {model_name}\n"
                    f"    api_url: {api_url}\n"
                    f"    error: {error_msg}"
                )

        if not unavailable:
            return

        safe_print(f"⏳ 模型不可用: {', '.join(unavailable)}，{check_interval:.0f}s后重试...")
        # Print probe details so users can see the real failure reasons.
        # Keep it always-on (or verbose) because a silent retry loop is hard to debug.
        if probe_logs and (verbose or True):
            safe_print("---- 探测详情（probe details）----")
            safe_print("\n".join(probe_logs))
            safe_print("-------------------------------")
        time.sleep(check_interval)


def load_dataset_items(path: str) -> List[Any]:
    """加载数据集（JSON 或 JSONL），返回列表。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                raise ValueError("JSON 文件内容必须是列表(list)。")
        elif path.endswith(".jsonl"):
            items = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
        else:
            raise ValueError("仅支持 .json 或 .jsonl 数据集。")
    except Exception as e:
        safe_print(f"❌ 读取数据集失败: {path}，错误: {e}")
        sys.exit(1)
    return items


def check_question_already_processed(
    config: Dict[str, Any],
    level: str,
    level_dataset_path: str,
    question_index: int,
    model_name: str,
    rollout_count: int = 1,
) -> bool:
    """
    检查某个模型是否已经完成了某道题目的处理（断点续跑功能）。
    检查当天目录下的 iter1.jsonl 或 iter1_split{worker_split}of{total_splits}.jsonl 文件是否包含该题目的答案。
    当使用split模式时（total_splits > 1），优先检查对应的split文件。
    """
    models_conf = config["models"]
    model_conf = models_conf[model_name]
    model_id_for_output = model_conf["model_name"]
    
    output_base = config["output_base"]
    model_dir_name = f"{model_id_for_output}_sglang"
    
    # 获取数据集名称（用于构建输出路径）
    dataset_name = os.path.splitext(os.path.basename(level_dataset_path))[0]
    dataset_dir = os.path.join(output_base, model_dir_name, dataset_name)
    
    # 读取数据集获取题目内容和总题目数
    items = load_dataset_items(level_dataset_path)
    total_questions = len(items)
    if question_index >= total_questions:
        return False
    
    item = items[question_index]
    question = item.get("question", "").strip()
    if not question:
        # 尝试从 messages 中提取
        try:
            user_msg = item["messages"][1]["content"]
            question = user_msg.split("User:")[1].strip() if "User:" in user_msg else user_msg
        except Exception:
            return False
    
    # 当使用split模式时（total_splits > 1），只检查对应的split文件是否存在
    # split文件格式：iter1_split{worker_split}of{total_splits}.jsonl
    # worker_split = question_index + 1（因为worker_split从1开始，而question_index从0开始）
    # 只要split文件存在，就认为推理已完成（不管是否包含答案，答案检查在process_single_question中进行）
    if total_questions > 1:
        worker_split = question_index + 1
        split_file = os.path.join(dataset_dir, f"iter1_split{worker_split}of{total_questions}.jsonl")
        
        # 如果split文件存在，返回True（推理已完成）
        if os.path.exists(split_file):
            return True
        
        # 如果split文件不存在，返回False（需要重新执行推理）
        return False
    
    # 非split模式：检查 iter1.jsonl 文件
    output_file = os.path.join(dataset_dir, "iter1.jsonl")
    
    if not os.path.exists(output_file):
        return False
    
    # 检查输出文件中是否已有该题目的答案
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    existing_question = data.get("question", "").strip()
                    if existing_question == question and "error" not in data:
                        # 找到匹配的题目且没有错误
                        return True
                except json.JSONDecodeError:
                    continue
    except Exception:
        return False
    
    return False


def process_single_question(
    config: Dict[str, Any],
    model_name: str,
    level: str,
    dataset_path: str,
    q_idx: int,
    num_q: int,
    current_question_item: Dict[str, Any],
    full_dataset_path: str,
    model_id_for_output: str,
    rollout_count: int,
) -> Dict[str, Any]:
    """
    处理单个题目（包括检查、推理、答案获取）。
    用于并发执行。
    """
    q_display = q_idx + 1
    result = {
        "success": False,
        "model": model_name,
        "level": level,
        "question_index": q_idx,
        "duration": 0.0,
        "skipped": False,
    }
    
    start_time = time.time()
    
    # 先检查题目是否已完成（检查split文件是否存在）
    question_processed = check_question_already_processed(
        config, level, dataset_path, q_idx, model_name, rollout_count
    )
    
    if question_processed:
        # 题目已完成（split文件存在），检查答案是否存在
        answer_exists = False
        if current_question_item:
            answer_exists = check_answer_already_exists(
                config, model_id_for_output, current_question_item, level
            )
        
        if answer_exists:
            # 有split文件且有答案，完全跳过（不跑推理，不获取答案）
            result["skipped"] = True
            result["duration"] = time.time() - start_time
            return result
        else:
            # 有split文件但没有答案，只获取答案（不跑推理）
            safe_print(f"[{model_name}][{level}] Q{q_display}/{num_q}: 推理已完成，获取答案...")
            run_get_standard_answers_for_model(
                config=config,
                model_name=model_name,
                question_item=current_question_item,
                full_dataset_path=full_dataset_path,
                level=level,
                async_mode=True,
            )
            result["skipped"] = True
            result["duration"] = time.time() - start_time
            return result
    
    # 题目未完成（split文件不存在），必须跑推理 + 获取答案（即使答案已存在也要获取）
    
    # 处理当前题目
    safe_print(f"[{model_name}][{level}] Q{q_display}/{num_q}: 开始处理...")
    question_result = run_single_model_for_question(
        config, level, dataset_path, q_idx, num_q, model_name
    )
    print(question_result)
    result.update(question_result)
    result["duration"] = time.time() - start_time
    
    if question_result.get("success"):
        safe_print(f"[{model_name}][{level}] Q{q_display}/{num_q}: ✅ 完成")
        # 异步获取答案
        safe_print(f"[{model_name}][{level}] Q{q_display}/{num_q}: 获取答案...")
        run_get_standard_answers_for_model(
            config=config,
            model_name=model_name,
            question_item=current_question_item,
            full_dataset_path=full_dataset_path,
            level=level,
            async_mode=True,
        )
    else:
        safe_print(f"[{model_name}][{level}] Q{q_display}/{num_q}: ❌ 失败")
    
    return result


def run_single_model_independently(
    config: Dict[str, Any],
    selected_levels: List[str],
    model_name: str,
) -> Dict[str, Any]:
    """
    单个模型独立执行所有级别的所有题目（并发模式）。
    每个模型内部使用线程池并发处理多个题目（默认并发数为3）。
    支持断点续跑：跳过已完成的题目。
    """
    models_conf = config["models"]
    model_conf = models_conf[model_name]
    model_id_for_output = model_conf["model_name"]
    rollout_count = model_conf.get("rollout_count", 1)
    
    # 获取并发数（从配置读取，默认为3）
    question_concurrency = model_conf.get("question_concurrency", 3)
    if question_concurrency < 1:
        question_concurrency = 3
    
    model_start_time = time.time()
    all_results: List[Dict[str, Any]] = []
    
    # 获取完整数据集路径（用于答案获取）
    # 可以通过环境变量配置，例如: export FULL_DATASET_PATH="/path/to/dataset.json"
    full_dataset_path = os.environ.get("FULL_DATASET_PATH", "dataset/realtime-benchmark.json")
    if not os.path.exists(full_dataset_path):
        full_dataset_path = None
    
    # 收集所有需要处理的任务
    tasks = []
    total_questions = 0
    for level in selected_levels:
        dataset_path = config["datasets"][level]
        items = load_dataset_items(dataset_path)
        num_q = len(items)
        
        if num_q == 0:
            continue
        
        total_questions += num_q
        safe_print(f"[{model_name}][{level}] {num_q}题")
        
        for q_idx in range(num_q):
            current_question_item = items[q_idx] if q_idx < len(items) else None
            tasks.append({
                "level": level,
                "dataset_path": dataset_path,
                "q_idx": q_idx,
                "num_q": num_q,
                "question_item": current_question_item,
            })
    
    # 使用线程池并发处理题目
    if not tasks:
        return {
            "model": model_name,
            "success_count": 0,
            "fail_count": 0,
            "total_count": 0,
            "duration": time.time() - model_start_time,
            "results": [],
        }
    
    # 确认模型可用（开始前检查一次）
    if not wait_until_model_available(config, model_name, check_interval=15.0, max_retries=3):
        safe_print(f"[{model_name}] ⚠️ 模型不可用，跳过")
        return {
            "model": model_name,
            "success_count": 0,
            "fail_count": len(tasks),
            "total_count": len(tasks),
            "duration": time.time() - model_start_time,
            "results": [],
        }
    
    safe_print(f"[{model_name}] 开始处理 {total_questions} 题（并发数: {question_concurrency}）...")
    
    with ThreadPoolExecutor(max_workers=question_concurrency) as executor:
        future_to_task = {
            executor.submit(
                process_single_question,
                config,
                model_name,
                task["level"],
                task["dataset_path"],
                task["q_idx"],
                task["num_q"],
                task["question_item"],
                full_dataset_path,
                model_id_for_output,
                rollout_count,
            ): task
            for task in tasks
        }
        
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                task_info = future_to_task[future]
                safe_print(f"[{model_name}][{task_info['level']}] Q{task_info['q_idx'] + 1}: ❌ 异常: {e}")
                all_results.append({
                    "success": False,
                    "model": model_name,
                    "level": task_info["level"],
                    "question_index": task_info["q_idx"],
                    "duration": 0.0,
                    "error": str(e),
                })
    
    model_duration = time.time() - model_start_time
    success_cnt = sum(1 for r in all_results if r.get("success") and not r.get("skipped"))
    fail_cnt = sum(1 for r in all_results if not r.get("success") and not r.get("skipped"))
    skipped_cnt = sum(1 for r in all_results if r.get("skipped"))
    
    safe_print(f"[{model_name}] ✅ 完成: 成功={success_cnt}, 失败={fail_cnt}, 跳过={skipped_cnt}, 耗时={model_duration:.2f}s")
    
    return {
        "model": model_name,
        "success_count": success_cnt,
        "fail_count": fail_cnt,
        "skipped_count": skipped_cnt,
        "total_count": len(all_results),
        "duration": model_duration,
        "results": all_results,
    }


def wait_until_model_available(
    config: Dict[str, Any],
    model_name: str,
    check_interval: float = 15.0,
    max_retries: int = 3,
) -> bool:
    """
    等待单个模型可用（最多重试 max_retries 次）。
    """
    models_conf = config["models"]
    model_conf = models_conf[model_name]
    
    for attempt in range(max_retries):
        probe = probe_model_via_provider(model_conf)
        if probe["ok"]:
            return True

        # Print detailed failure info for this model (helps debug transient issues).
        err = probe.get("error", "unknown")
        raw = probe.get("raw", {}) or {}
        api_url = model_conf.get("api_url", "")
        provider = (model_conf.get("llm_provider", "openai") or "openai").lower()
        model_id = model_conf.get("model_name", "")
        extra = ""
        try:
            if raw:
                bits = []
                if "choices_count" in raw:
                    bits.append(f"choices_count={raw.get('choices_count')}")
                if "model" in raw:
                    bits.append(f"resp_model={raw.get('model')}")
                if "repr" in raw and raw.get("repr"):
                    bits.append(f"repr={str(raw.get('repr'))[:200]}")
                if bits:
                    extra = " | " + ", ".join(bits)
        except Exception:
            pass
        safe_print(
            f"⚠️  模型探测失败 [{model_name}] attempt {attempt + 1}/{max_retries}\n"
            f"    provider: {provider}\n"
            f"    model_name: {model_id}\n"
            f"    api_url: {api_url}\n"
            f"    error: {err}{extra}"
        )
        
        if attempt < max_retries - 1:
            time.sleep(check_interval)
    
    return False


def run_single_model_for_question(
    config: Dict[str, Any],
    level: str,
    level_dataset_path: str,
    question_index: int,
    total_questions: int,
    model_name: str,
) -> Dict[str, Any]:
    """
    针对「某个 level 的某一道题」运行单个模型一次 run_multi_react.py。
    通过 total_splits / worker_split 保证只处理当前这 1 道题。
    """
    models_conf = config["models"]
    model_conf = models_conf[model_name]

    model_id_for_output = model_conf["model_name"]

    output_base = config["output_base"]
    python_script = config["python_script"]

    # 构造日志目录：{output_base}/{model_name}_sglang/log
    model_dir_name = f"{model_id_for_output}_sglang"
    log_dir = os.path.join(output_base, model_dir_name, "log")
    os.makedirs(log_dir, exist_ok=True)

    # 环境变量（与之前的主脚本保持一致）
    env = os.environ.copy()
    env["SERPER_KEY_ID"] = config["shared_keys"]["SERPER_KEY_ID"]
    env["JINA_API_KEYS"] = config["shared_keys"]["JINA_API_KEYS"]
    # IDP 配置（可选）
    if "IDP_KEY_ID" in config["shared_keys"] and config["shared_keys"]["IDP_KEY_ID"]:
        env["IDP_KEY_ID"] = config["shared_keys"]["IDP_KEY_ID"]
    if "IDP_KEY_SECRET" in config["shared_keys"] and config["shared_keys"]["IDP_KEY_SECRET"]:
        env["IDP_KEY_SECRET"] = config["shared_keys"]["IDP_KEY_SECRET"]
    # USE_IDP 配置（可选，默认为 True）
    if "USE_IDP" in config["shared_keys"]:
        env["USE_IDP"] = str(config["shared_keys"]["USE_IDP"]).lower()
    env["API_URL"] = model_conf["api_url"]
    env["API_KEY"] = model_conf["api_key"]
    env["MODEL_NAME"] = model_id_for_output
    env["LLM_PROVIDER"] = model_conf.get("llm_provider", "openai").lower()
    env["USE_FUNCTION_CALLING"] = str(model_conf.get("use_function_calling", True)).lower()
    env["DATASET"] = level_dataset_path
    env["OUTPUT_PATH"] = output_base
    env["MAX_WORKERS"] = str(model_conf.get("max_workers", 20))
    env["TEMPERATURE"] = str(model_conf.get("temperature", 0.6))
    env["PRESENCE_PENALTY"] = str(model_conf.get("presence_penalty", 1.1))
    env["ROLLOUT_COUNT"] = str(model_conf.get("rollout_count", 1))
    env["WORLD_SIZE"] = "1"
    env["RANK"] = "0"

    # 构造命令
    cmd = [
        sys.executable,
        "-u",
        python_script,
        "--dataset",
        level_dataset_path,
        "--output",
        output_base,
        "--max_workers",
        str(model_conf.get("max_workers", 20)),
        "--model",
        model_id_for_output,
        "--temperature",
        str(model_conf.get("temperature", 0.6)),
        "--presence_penalty",
        str(model_conf.get("presence_penalty", 1.1)),
        "--total_splits",
        str(total_questions),
        "--worker_split",
        str(question_index + 1),
        "--roll_out_count",
        str(model_conf.get("rollout_count", 1)),
        "--api_url",
        model_conf["api_url"],
        "--api_key",
        model_conf["api_key"],
        "--log_dir",
        log_dir,
    ]

    # 输出已由 process_single_question 处理，这里不输出
    question_id_display = question_index + 1

    start_ts = time.time()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=os.path.dirname(python_script),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        duration = time.time() - start_ts

        if proc.returncode == 0:
            return {
                "success": True,
                "model": model_name,
                "level": level,
                "question_index": question_index,
                "duration": duration,
            }
        else:
            output_tail = (proc.stdout or "")[-1800:]
            return {
                "success": False,
                "model": model_name,
                "level": level,
                "question_index": question_index,
                "duration": duration,
                "error": output_tail,  
            }
    except Exception as e:
        duration = time.time() - start_ts
        return {
            "success": False,
            "model": model_name,
            "level": level,
            "question_index": question_index,
            "duration": duration,
            "error": str(e),  
        }


def check_answer_already_exists(
    config: Dict[str, Any],
    model_name: str,
    question_item: Dict[str, Any],
    level: str,
) -> bool:
    """
    检查答案文件中是否已存在该题目的答案（断点续跑功能）。
    如果答案已存在，则不需要重新获取。
    根据level（L1/L2/L3）检查对应级别的答案是否存在。
    答案文件按级别分别存储：L1-{model_name}_answers.json, L2-{model_name}_answers.json, L3-{model_name}_answers.json
    """
    if not question_item:
        return False
    
    # 获取输出目录（当天目录）
    run_ts = os.environ.get("RUN_TS") or os.environ.get("RUN_TIMESTAMP")
    if not run_ts:
        run_ts = time.strftime("%Y-%m-%d")
    
    # 可以通过环境变量配置结果根目录，例如: export RESULTS_BASE="/path/to/results"
    results_base = os.environ.get("RESULTS_BASE", "results")
    output_dir = os.path.join(results_base, run_ts)
    # 按级别分别存储答案文件
    model_answers_file = os.path.join(output_dir, f"{level}-{model_name}_answers.json")
    
    if not os.path.exists(model_answers_file):
        return False
    
    # 读取答案文件检查是否已有该题目
    try:
        with open(model_answers_file, "r", encoding="utf-8") as f:
            answers = json.load(f)
        
        if not isinstance(answers, list):
            return False
        
        target_key_id = question_item.get("key_id")
        # 根据level获取对应的ID
        if level == "L1":
            target_id = question_item.get("L1_id")
            answer_field = "L1_标准答案"
        elif level == "L2":
            target_id = question_item.get("L2_id")
            answer_field = "L2_标准答案"
        elif level == "L3":
            target_id = question_item.get("L3_id")
            answer_field = "L3_标准答案"
        else:
            target_id = question_item.get("L1_id")
            answer_field = "L1_标准答案"
        
        for answer_item in answers:
            answer_key_id = answer_item.get("key_id")
            # 根据level检查对应的ID
            if level == "L1":
                answer_id = answer_item.get("L1_id")
            elif level == "L2":
                answer_id = answer_item.get("L2_id")
            elif level == "L3":
                answer_id = answer_item.get("L3_id")
            else:
                answer_id = answer_item.get("L1_id")
            
            # 检查是否匹配（key_id或对应级别的ID）
            if (target_key_id and answer_key_id == target_key_id) or \
               (target_id and answer_id == target_id):
                # 检查对应级别的答案是否有效（不为空且不是"null"）
                level_answer = answer_item.get(answer_field) or answer_item.get("answer") or ""
                if level_answer and level_answer != "null" and level_answer.strip():
                    return True
        
        return False
    except Exception:
        return False


def run_get_standard_answers_for_model(
    config: Dict[str, Any],
    model_name: str,
    question_item: Dict[str, Any] = None,
    full_dataset_path: str = None,
    level: str = "L1",
    async_mode: bool = True,  # 默认异步执行
) -> None:
    """
    为单个模型获取标准答案。
    答案文件按级别分别存储：{output_base}/{date}/L1-{model_name}_answers.json, L2-{model_name}_answers.json, L3-{model_name}_answers.json
    如果答案已存在，则跳过（断点续跑功能）。
    根据level（L1/L2/L3）获取对应级别的答案。
    默认异步执行，不阻塞模型继续处理下一个问题。
    """
    if not os.path.exists(GET_STANDARD_ANSWERS_SH):
        return
    
    # 检查答案是否已存在（断点续跑功能）
    if question_item:
        # 获取模型ID用于检查答案文件
        models_conf = config.get("models", {})
        if model_name in models_conf:
            model_conf = models_conf[model_name]
            model_id_for_output = model_conf["model_name"]
        else:
            model_id_for_output = model_name
        
        # 检查答案是否已存在（根据level检查对应级别的答案）
        if check_answer_already_exists(config, model_id_for_output, question_item, level):
            return
    
    # 获取输出目录（当天目录）
    run_ts = os.environ.get("RUN_TS") or os.environ.get("RUN_TIMESTAMP")
    if not run_ts:
        run_ts = time.strftime("%Y-%m-%d")
    
    # 可以通过环境变量配置结果根目录，例如: export RESULTS_BASE="/path/to/results"
    results_base = os.environ.get("RESULTS_BASE", "results")
    output_dir = os.path.join(results_base, run_ts)
    os.makedirs(output_dir, exist_ok=True)
    
    models_conf = config.get("models", {})
    if model_name in models_conf:
        model_conf = models_conf[model_name]
        model_id_for_output = model_conf["model_name"]
    else:
        model_id_for_output = model_name
    
    # 为每个模型和每个级别创建独立的答案文件路径
    model_answers_file = os.path.join(output_dir, f"{level}-{model_id_for_output}_answers.json")
    
    env = os.environ.copy()
    env["MODEL_NAME"] = model_id_for_output
    env["MODEL_ANSWERS_FILE"] = model_answers_file
    env["QUESTION_LEVEL"] = level  # 传递级别信息（L1/L2/L3）
    
    if question_item and full_dataset_path:
        try:
            import tempfile
            # 从完整数据集中查找对应的题目（包含完整字段）
            converted_question_item = None
            if os.path.exists(full_dataset_path):
                try:
                    with open(full_dataset_path, "r", encoding="utf-8") as f:
                        full_data = json.load(f)
                    
                    # 根据 key_id 或 L1_id 查找对应的题目
                    target_key_id = question_item.get("key_id")
                    target_l1_id = question_item.get("L1_id")
                    
                    for item in full_data:
                        item_key_id = item.get("key_id")
                        item_l1_id = item.get("L1_id")
                        if (target_key_id and item_key_id == target_key_id) or \
                           (target_l1_id and item_l1_id == target_l1_id):
                            converted_question_item = item
                            break
                except Exception as e:
                    safe_print(f"    ⚠️  从完整数据集读取失败: {e}")
            
            # 如果没有找到完整数据，则转换字段名
            if not converted_question_item:
                converted_question_item = question_item.copy()
                # 将 question 字段转换为 L1_问题
                if "question" in converted_question_item and "L1_问题" not in converted_question_item:
                    converted_question_item["L1_问题"] = converted_question_item.pop("question")
                # 确保必要的字段存在
                if "L1_id" not in converted_question_item:
                    converted_question_item["L1_id"] = question_item.get("L1_id", "")
                if "key_id" not in converted_question_item:
                    converted_question_item["key_id"] = question_item.get("key_id", "")
            
            temp_fd, temp_input_file = tempfile.mkstemp(suffix=".json", prefix="question_", text=True)
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                json.dump([converted_question_item], f, ensure_ascii=False, indent=2)
            env["SINGLE_QUESTION_JSON"] = temp_input_file
        except Exception as e:
            safe_print(f"    ⚠️  创建临时题目文件失败: {e}")
    
    try:
        if async_mode:
            proc = subprocess.Popen(["bash", GET_STANDARD_ANSWERS_SH], env=env)
            # 异步执行，不输出详细信息
        else:
            proc = subprocess.run(
                ["bash", GET_STANDARD_ANSWERS_SH],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if proc.returncode != 0:
                safe_print(f"[{model_id_for_output}][{level}] 答案获取失败 (exit code: {proc.returncode})")
    except Exception as e:
        safe_print(f"[{model_id_for_output}][{level}] 答案获取异常: {e}")


def run_get_standard_answers(async_mode: bool = False, question_item: Dict[str, Any] = None, full_dataset_path: str = None) -> None:
    """
    调用原有的 get_standard_answers.sh。
    - async_mode=False: 同步调用，等待脚本执行结束（用于最终兜底一次）。
    - async_mode=True: 异步启动子进程，不等待完成（用于每题后的标准答案更新，不阻塞下一题）。
    - question_item: 当前题目的数据项（可选，如果提供则只处理该题目）。
    - full_dataset_path: 完整数据集的路径（用于从完整数据集中查找对应题目）。
    注意：该脚本本身会生成带时间戳的结果和 standard_answers-*.json。
    """
    if not os.path.exists(GET_STANDARD_ANSWERS_SH):
        safe_print(
            f"⚠️  警告: 未找到 get_standard_answers.sh ({GET_STANDARD_ANSWERS_SH})，跳过标准答案获取步骤。"
        )
        return
    
    # 如果提供了单个题目，创建临时 JSON 文件只包含该题目
    temp_input_file = None
    env = os.environ.copy()
    
    if question_item and full_dataset_path:
        try:
            # 读取完整数据集，找到对应的题目
            import json
            with open(full_dataset_path, "r", encoding="utf-8") as f:
                full_data = json.load(f)
            
            # 根据 key_id 或 L1_id 找到对应的题目
            target_key_id = question_item.get("key_id")
            target_l1_id = question_item.get("L1_id")
            
            matching_item = None
            for item in full_data:
                if target_key_id and item.get("key_id") == target_key_id:
                    matching_item = item
                    break
                elif target_l1_id and item.get("L1_id") == target_l1_id:
                    matching_item = item
                    break
            
            if matching_item:
                # 创建临时 JSON 文件
                import tempfile
                temp_fd, temp_input_file = tempfile.mkstemp(suffix=".json", prefix="question_", text=True)
                with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                    json.dump([matching_item], f, ensure_ascii=False, indent=2)
                
                # 通过环境变量传递临时文件路径
                env["SINGLE_QUESTION_JSON"] = temp_input_file
                safe_print(f"  -> 已创建临时题目文件: {temp_input_file}")
            else:
                safe_print(f"  ⚠️  警告: 未在完整数据集中找到题目 (key_id={target_key_id}, L1_id={target_l1_id})，将处理整个数据集")
        except Exception as e:
            safe_print(f"  ⚠️  警告: 创建临时题目文件失败: {e}，将处理整个数据集")
    
    # 设置清理函数（异步模式下由脚本自己清理，同步模式下立即清理）
    def cleanup_temp_file():
        if temp_input_file and os.path.exists(temp_input_file):
            try:
                os.unlink(temp_input_file)
            except Exception:
                pass
    
    try:
        if async_mode:
            safe_print("  -> 异步启动 get_standard_answers.sh 获取 / 更新标准答案（不阻塞下一题） ...")
            try:
                # 异步模式：不捕获输出，直接继承当前终端 stdout/stderr
                # 注意：异步模式下，临时文件会在脚本执行完成后由脚本清理
                proc = subprocess.Popen(["bash", GET_STANDARD_ANSWERS_SH], env=env)
                safe_print(f"     ✅ get_standard_answers.sh 已在后台启动 (PID: {proc.pid})")
                # 异步模式下不立即清理，让脚本执行完成后再清理
            except Exception as e:
                safe_print(f"     ❌ 异步启动 get_standard_answers.sh 失败: {e}")
                cleanup_temp_file()
            return

        safe_print("  -> 同步调用 get_standard_answers.sh 获取 / 更新标准答案 ...")
        try:
            proc = subprocess.run(
                ["bash", GET_STANDARD_ANSWERS_SH],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if proc.returncode == 0:
                safe_print("     ✅ get_standard_answers.sh 执行成功。")
            else:
                safe_print(
                    "     ❌ get_standard_answers.sh 执行失败，退出码 "
                    f"{proc.returncode}，输出如下：\n{proc.stdout}"
                )
        except Exception as e:
            safe_print(f"     ❌ 调用 get_standard_answers.sh 过程中出现异常: {e}")
        finally:
            cleanup_temp_file()
    except Exception as e:
        safe_print(f"     ❌ 调用 get_standard_answers.sh 过程中出现异常: {e}")
        cleanup_temp_file()


# ========================== 主流程 ========================== #


def main() -> None:
    safe_print("=" * 60)
    safe_print("按模型独立执行 + 并发处理")
    safe_print("=" * 60)

    config = load_config(CONFIG_FILE)
    validate_config(config)

    # 允许通过环境变量为本次运行指定一个日期目录
    run_ts = os.environ.get("RUN_TS") or os.environ.get("RUN_TIMESTAMP")
    if not run_ts:
        run_ts = time.strftime("%Y-%m-%d")
        os.environ["RUN_TS"] = run_ts
    
    base_output = config["output_base"].rstrip("/ ")
    new_output = os.path.join(base_output, run_ts)
    config["output_base"] = new_output
    safe_print(f"输出目录: {new_output}")
    
    os.makedirs(new_output, exist_ok=True)

    selected_models = get_selected_models(config)
    selected_levels = get_selected_levels(config)

    if not selected_models:
        safe_print("❌ 错误: 没有可运行的模型（没有启用的模型或 selected_models 为空且 enabled 全为 false）")
        sys.exit(1)
    if not selected_levels:
        safe_print("❌ 错误: 没有可运行的问题级别（L1/L2/L3 数据集路径不存在或未选择）")
        sys.exit(1)

    safe_print(f"模型: {', '.join(selected_models)}")
    safe_print(f"级别: {', '.join(selected_levels)}")
    safe_print("")

    total_start = time.time()
    
    # 检查所有模型是否可用（初始检查）
    safe_print("🔍 检查模型可用性...")
    wait_until_all_models_available(config, selected_models, check_interval=15.0)
    safe_print("✅ 所有模型可用，开始执行")
    safe_print("")

    all_model_results: List[Dict[str, Any]] = []
    
    with ThreadPoolExecutor(max_workers=len(selected_models)) as executor:
        future_to_model = {
            executor.submit(
                run_single_model_independently,
                config,
                selected_levels,
                model_name,
            ): model_name
            for model_name in selected_models
        }
        
        for future in as_completed(future_to_model):
            model_result = future.result()
            all_model_results.append(model_result)

    total_duration = time.time() - total_start

    # 汇总所有模型的结果
    total_success = sum(r["success_count"] for r in all_model_results)
    total_fail = sum(r["fail_count"] for r in all_model_results)
    total_questions = sum(r["total_count"] for r in all_model_results)

    safe_print("")
    safe_print("=" * 60)
    safe_print("执行结果汇总")
    safe_print("=" * 60)
    safe_print(f"总模型数: {len(selected_models)}")
    safe_print(f"总题目数: {total_questions}")
    safe_print(f"✅ 成功: {total_success}")
    safe_print(f"❌ 失败: {total_fail}")
    safe_print(f"⏱️  总耗时: {total_duration:.2f}s")
    safe_print("")
    safe_print("各模型详情:")
    for model_result in all_model_results:
        model_name = model_result["model"]
        success_cnt = model_result["success_count"]
        fail_cnt = model_result["fail_count"]
        skipped_cnt = model_result.get("skipped_count", 0)
        duration = model_result["duration"]
        safe_print(
            f"  {model_name}: ✅{success_cnt} ❌{fail_cnt} ⏭️{skipped_cnt} ⏱️{duration:.1f}s"
        )

    if total_fail > 0:
        safe_print("\n⚠️  部分执行失败，请查看日志排查。")
        sys.exit(1)

    safe_print("\n🎉 所有任务完成。")


if __name__ == "__main__":
    main()


