#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import json
import time
import tempfile
import os
import sys
import re
import traceback
from datetime import datetime, timezone, timedelta

# ========== 配置 ==========
# 可以通过环境变量配置路径，例如: export INPUT_JSON="/path/to/input.json"
# 如果未设置环境变量，脚本会从命令行参数或默认路径读取
# 优先从环境变量获取，如果没有则使用默认值
# 注意：get_standard_answers.sh 会通过 sed 直接修改这些变量的赋值
INPUT_JSON = os.environ.get("INPUT_JSON", "")
OUTPUT_JSON = os.environ.get("OUTPUT_JSON", "")

# 如果 INPUT_JSON 为空，尝试从默认路径读取
if not INPUT_JSON:
    default_input = os.path.join(os.path.dirname(__file__), "..", "dataset", "realtime-benchmark.json")
    if os.path.exists(default_input):
        INPUT_JSON = default_input
    else:
        print("❌ 错误: INPUT_JSON 未设置且默认文件不存在", file=sys.stderr)
        sys.exit(1)

# 并发上限（可按需调整）
MAX_CONCURRENT = 1

# 单个脚本最大允许运行秒数（超时后会kill）
SCRIPT_TIMEOUT = 360  # 秒，按需调整

# 重试次数配置
MAX_RETRY_COUNT = 5  # 最多重试5次

# 如果工作流字段太长，保存到 JSON 时截断显示多少字符（原始工作流仍保存在 tmp 文件中）
WORKFLOW_PREVIEW_CHARS = 200000000
# =========================

# 追加到临时脚本末尾的 wrapper：它会尝试从 globals() 里取 "result"，否则尝试调用 main()（如果 safe）
# 并以明确的标记把 final result JSON 打到 stdout，方便主程序解析
APPEND_WRAPPER = r'''
# ---- auto-appended wrapper for capturing final result ----
import json, sys, asyncio
def __print_final(obj):
    try:
        print("__FINAL_RESULT_START__")
        # 使用 ensure_ascii=False 以保留中文
        print(json.dumps(obj, ensure_ascii=False, default=str))
        print("__FINAL_RESULT_END__")
    except Exception as _e:
        try:
            print("__FINAL_RESULT_START__")
            print(json.dumps({"_capture_error": str(_e)}, ensure_ascii=False))
            print("__FINAL_RESULT_END__")
        except:
            pass

try:
    # 优先使用已经存在的变量 'result'
    if "result" in globals():
        __print_final(globals().get("result"))
    else:
        # 如果定义了 main 并且没有 result，尝试调用 main（若是协程则 asyncio.run）
        m = globals().get("main", None)
        if m:
            try:
                if asyncio.iscoroutinefunction(m):
                    r = asyncio.run(m())
                    __print_final(r)
                else:
                    r = m()
                    __print_final(r)
            except Exception as e:
                __print_final({"_run_error": str(e)})
        else:
            # 没有 result，也没有 main，可以输出空
            __print_final(None)
except Exception as e:
    __print_final({"_wrapper_error": str(e)})
# ---- end wrapper ----
'''

# helper: 确保输出目录存在
if OUTPUT_JSON:
    output_dir = os.path.dirname(OUTPUT_JSON)
    if output_dir:  # 只有当目录路径不为空时才创建
        os.makedirs(output_dir, exist_ok=True)
    else:
        # 如果 OUTPUT_JSON 只有文件名（没有目录），使用当前目录
        pass  # 不需要创建目录
else:
    # OUTPUT_JSON 为空，后续代码会处理或报错
    pass


def format_timestamp(timestamp):
    """将时间戳格式化为北京时间字符串"""
    # 定义北京时区 (UTC+8)
    beijing_tz = timezone(timedelta(hours=8))
    # 转换为北京时间
    beijing_time = datetime.fromtimestamp(timestamp, tz=beijing_tz)
    return beijing_time.strftime("%Y年%m月%d日 %H时%M分%S秒")


def format_standard_answer_for_json(standard_answer):
    """将标准答案格式化为适合JSON的字符串"""
    if standard_answer == "":
        return ""
    
    try:
        # 如果是字符串，直接返回
        if isinstance(standard_answer, str):
            return standard_answer
        
        # 如果是其他类型，转为 JSON 字符串
        return json.dumps(standard_answer, ensure_ascii=False, default=str)
    except Exception:
        # 如果转换失败，返回字符串表示
        return str(standard_answer)


def save_results_to_files(original_data, results_by_order):
    """保存结果到 JSON 文件，保持原始结构
    
    如果输出文件已存在且只处理单个题目，则读取现有文件并合并结果。
    如果输出文件不存在或处理多个题目，则创建新的输出文件。
    """
    # 检查是否只处理单个题目（通过 original_data 的长度判断）
    is_single_question = len(original_data) == 1
    
    # 读取现有输出文件（如果存在且只处理单个题目）
    existing_data = []
    if is_single_question and os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = []
        except Exception as e:
            print(f"⚠️ 读取现有输出文件失败，将创建新文件：{e}")
            existing_data = []
    
    # 如果是单题目模式且有现有数据，更新现有数据中对应的题目
    if is_single_question and existing_data:
        original_item = original_data[0]
        original_key_id = original_item.get("key_id")
        original_l1_id = original_item.get("L1_id")
        
        # 在现有数据中查找对应的题目
        found_idx = None
        for idx, item in enumerate(existing_data):
            item_key_id = item.get("key_id")
            item_l1_id = item.get("L1_id")
            if (original_key_id and item_key_id == original_key_id) or \
               (original_l1_id and item_l1_id == original_l1_id):
                found_idx = idx
                break
        
        # 如果有执行结果，更新对应题目
        if 0 in results_by_order:
            result = results_by_order[0]
            execution_time = result.get("执行时间", "")
            standard_answer = format_standard_answer_for_json(result.get("标准答案", ""))
            
            if found_idx is not None:
                # 更新现有数据中的对应题目
                existing_data[found_idx]["L1_执行时间"] = execution_time
                existing_data[found_idx]["L1_标准答案"] = standard_answer
                
                if existing_data[found_idx].get("L2_id") and existing_data[found_idx].get("L2_问题"):
                    existing_data[found_idx]["L2_执行时间"] = execution_time
                    existing_data[found_idx]["L2_标准答案"] = standard_answer
                
                if existing_data[found_idx].get("L3_id") and existing_data[found_idx].get("L3_问题"):
                    existing_data[found_idx]["L3_执行时间"] = execution_time
                    existing_data[found_idx]["L3_标准答案"] = standard_answer
            else:
                # 如果没找到，创建新题目并添加到现有数据中
                new_item = original_item.copy()
                new_item["L1_执行时间"] = execution_time
                new_item["L1_标准答案"] = standard_answer
                
                if new_item.get("L2_id") and new_item.get("L2_问题"):
                    new_item["L2_执行时间"] = execution_time
                    new_item["L2_标准答案"] = standard_answer
                
                if new_item.get("L3_id") and new_item.get("L3_问题"):
                    new_item["L3_执行时间"] = execution_time
                    new_item["L3_标准答案"] = standard_answer
                
                existing_data.append(new_item)
        
        updated_data = existing_data
    else:
        # 多题目模式或没有现有数据，使用原始逻辑
        updated_data = []
        
        for i, original_item in enumerate(original_data):
            # 复制原始数据
            updated_item = original_item.copy()
            
            # 如果有对应的执行结果，更新相关字段
            if i in results_by_order:
                result = results_by_order[i]
                execution_time = result.get("执行时间", "")
                standard_answer = format_standard_answer_for_json(result.get("标准答案", ""))
                
                # 总是更新L1字段
                updated_item["L1_执行时间"] = execution_time
                updated_item["L1_标准答案"] = standard_answer
                
                # 检查L2是否需要更新
                if updated_item.get("L2_id") and updated_item.get("L2_问题"):
                    updated_item["L2_执行时间"] = execution_time
                    updated_item["L2_标准答案"] = standard_answer
                
                # 检查L3是否需要更新  
                if updated_item.get("L3_id") and updated_item.get("L3_问题"):
                    updated_item["L3_执行时间"] = execution_time
                    updated_item["L3_标准答案"] = standard_answer
            
            updated_data.append(updated_item)
    
    # 保存 JSON 文件
    try:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 写入 {OUTPUT_JSON} 失败：{e}")


async def execute_script_once(script_path):
    """
    执行单次脚本，返回执行结果的字典
    """
    start_timestamp = time.time()
    start_time_str = format_timestamp(start_timestamp)

    # 为Playwright设置独立环境
    env = os.environ.copy()
    # 设置独立的用户数据目录
    env['PLAYWRIGHT_BROWSERS_PATH'] = f"/tmp/playwright_{int(time.time())}_{os.getpid()}"
    
    # 禁用共享内存（避免多进程冲突）
    env['PLAYWRIGHT_CHROMIUM_USE_HEADLESS'] = '1'
    # 启动子进程

    proc = await asyncio.create_subprocess_exec(
        sys.executable, script_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=SCRIPT_TIMEOUT)
        timeout_happened = False
    except asyncio.TimeoutError:
        # 超时 -> kill
        try:
            proc.kill()
        except Exception:
            pass
        await proc.wait()
        # 尝试读一下残留输出（非阻塞）
        try:
            stdout, stderr = await proc.communicate()
        except Exception:
            stdout, stderr = b"", b""
        timeout_happened = True

    end_timestamp = time.time()
    duration = end_timestamp - start_timestamp

    stdout_text = stdout.decode(errors="replace") if stdout else ""
    stderr_text = stderr.decode(errors="replace") if stderr else ""

    # 解析 FINAL_RESULT 标记
    final_match = re.search(r"__FINAL_RESULT_START__\s*(.*?)\s*__FINAL_RESULT_END__", stdout_text, flags=re.S)
    standard_answer = ""
    parsed_standard = None
    if final_match:
        candidate = final_match.group(1).strip()
        # 试着用 json.loads 解析
        try:
            parsed_standard = json.loads(candidate)
            standard_answer = parsed_standard
        except Exception:
            # 无法解析成 JSON，就把原文放到标准答案（作为字符串）
            standard_answer = candidate

    # 判断是否出错（非 0 退出码 或 超时）
    proc_return_code = proc.returncode if proc else None
    is_error = False
    error_reason = ""
    if timeout_happened:
        is_error = True
        error_reason = f"timeout after {SCRIPT_TIMEOUT}s"
    elif proc_return_code is not None and proc_return_code != 0:
        is_error = True
        # prefer stderr
        error_reason = stderr_text.strip() or f"exit code {proc_return_code}"

    # 构造代码输出字段：把 stdout + stderr 合并，保留中间打印
    combined_output = ""
    if stdout_text:
        combined_output += "[STDOUT]\n" + stdout_text
    if stderr_text:
        if combined_output:
            combined_output += "\n"
        combined_output += "[STDERR]\n" + stderr_text

    return {
        "start_timestamp": start_timestamp,
        "start_time_str": start_time_str,
        "duration": duration,
        "combined_output": combined_output,
        "standard_answer": standard_answer,
        "parsed_standard": parsed_standard,
        "final_match": final_match,
        "is_error": is_error,
        "error_reason": error_reason,
        "timeout_happened": timeout_happened,
        "proc_return_code": proc_return_code
    }


async def run_single(row, order, sem, lock, original_data, results_by_order, failed_ids):
    """
    执行单个任务（把 row['工作流'] 写成临时文件并执行），并把结果写回 results_by_order（key 为 order）
    支持最多重试 MAX_RETRY_COUNT 次
    """
    async with sem:
        script_code = row.get("工作流", "") or ""
        await asyncio.sleep(10)  # 等待10秒避免冲突
        # 写临时文件（原始代码 + wrapper）
        script_path = None
        result_entry = None
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as tmp_file:
                tmp_file.write(script_code)
                tmp_file.write("\n\n")
                tmp_file.write(APPEND_WRAPPER)
                script_path = tmp_file.name

            # 重试逻辑：最多尝试 MAX_RETRY_COUNT 次
            last_result = None
            retry_count = 0
            total_duration = 0
            retry_details = []
            success_execution_time = None  # 记录成功执行的时间
            
            for attempt in range(MAX_RETRY_COUNT):
                try:
                    exec_result = await execute_script_once(script_path)
                    last_result = exec_result
                    total_duration += exec_result["duration"]
                    
                    retry_info = f"attempt {attempt + 1} at {exec_result['start_time_str']}: {exec_result['duration']:.2f}s"
                    if exec_result["is_error"]:
                        print(
                            f"❌ 工作流执行失败 | id={row.get('id', '')} | "
                            f"attempt={attempt + 1}/{MAX_RETRY_COUNT} | "
                            f"原因={exec_result.get('error_reason', '')}"
                        )
                        retry_info += f" (ERROR: {exec_result['error_reason']})"
                        retry_details.append(retry_info)
                        retry_count += 1
                        # 如果还有重试机会，继续下一次尝试
                        if attempt < MAX_RETRY_COUNT - 1:
                            continue
                    else:
                        print(
                            f"✅ 工作流执行返回 | id={row.get('id', '')} | "
                            f"attempt={attempt + 1}/{MAX_RETRY_COUNT} | "
                            f"返回值={format_standard_answer_for_json(exec_result.get('standard_answer', ''))}"
                        )
                        # 成功了，记录这次尝试并跳出循环
                        success_execution_time = exec_result["start_time_str"]  # 记录成功那次的时间
                        retry_info += " (SUCCESS)"
                        retry_details.append(retry_info)
                        break
                        
                except Exception as e:
                    # 单次执行过程中的异常
                    current_time = format_timestamp(time.time())
                    print(
                        f"❌ 工作流执行异常 | id={row.get('id', '')} | "
                        f"attempt={attempt + 1}/{MAX_RETRY_COUNT} | 异常={str(e)}"
                    )
                    retry_info = f"attempt {attempt + 1} at {current_time}: exception - {str(e)}"
                    retry_details.append(retry_info)
                    retry_count += 1
                    if attempt < MAX_RETRY_COUNT - 1:
                        continue
                    else:
                        # 最后一次尝试也失败了
                        success_execution_time = current_time  # 失败时也记录最后一次的时间
                        last_result = {
                            "start_timestamp": time.time(),
                            "start_time_str": current_time,
                            "duration": 0,
                            "combined_output": "",
                            "standard_answer": "",
                            "parsed_standard": None,
                            "final_match": None,
                            "is_error": True,
                            "error_reason": f"All attempts failed, last error: {str(e)}",
                            "timeout_happened": False,
                            "proc_return_code": None
                        }

            # 如果没有任何执行记录，设置默认时间
            if success_execution_time is None:
                success_execution_time = format_timestamp(time.time())

            # 构造最终结果
            retry_summary = f"tried {len(retry_details)} times, total duration: {total_duration:.2f}s"
            if retry_count > 0:
                retry_summary += f", failed {retry_count} times"
            
            # 在代码输出中添加重试信息
            combined_output_with_retry = f"[RETRY_INFO]\n{retry_summary}\n"
            for detail in retry_details:
                combined_output_with_retry += f"  {detail}\n"
            combined_output_with_retry += "\n" + (last_result["combined_output"] if last_result else "")

            result_entry = {
                "id": row.get("id", ""),
                "类型": row.get("类型", ""),
                "问题": row.get("问题", ""),
                "工作流": (row.get("工作流", "")[:WORKFLOW_PREVIEW_CHARS] + "...") if len(row.get("工作流", "")) > WORKFLOW_PREVIEW_CHARS else row.get("工作流", ""),
                "执行时间": success_execution_time,  # 第一次执行的时间点
                "代码输出": combined_output_with_retry,
                "标准答案": last_result["standard_answer"] if last_result and (last_result["parsed_standard"] is not None or last_result["final_match"]) else "",
                "error": bool(last_result["is_error"]) if last_result else True,
                "error_reason": last_result["error_reason"] if last_result else "No execution result",
                "retry_count": retry_count,
                "total_attempts": len(retry_details),
                "total_duration": f"{total_duration:.2f}s"  # 新增：总执行持续时间
            }

        except Exception as e:
            # 如果在写文件或启动进程时抛异常，也记录下来，不停止调度器
            tb = traceback.format_exc()
            current_time = format_timestamp(time.time())
            result_entry = {
                "id": row.get("id", ""),
                "类型": row.get("类型", ""),
                "问题": row.get("问题", ""),
                "工作流": (row.get("工作流", "")[:WORKFLOW_PREVIEW_CHARS] + "...") if len(row.get("工作流", "")) > WORKFLOW_PREVIEW_CHARS else row.get("工作流", ""),
                "执行时间": current_time,
                "代码输出": "",
                "标准答案": "",
                "error": True,
                "error_reason": f"scheduler exception: {str(e)}\n{tb}",
                "retry_count": 0,
                "total_attempts": 0,
                "total_duration": "0s"
            }
        finally:
            # 清理临时脚本
            try:
                if script_path and os.path.exists(script_path):
                    os.remove(script_path)
            except Exception:
                pass

        # 把结果写回共享结构，并实时写入 JSON（按原顺序排序）
        async with lock:
            results_by_order[order] = result_entry
            # 更新失败 id 列表
            if result_entry.get("error"):
                failed_ids.append(result_entry.get("id"))

            # 保存到文件（JSON）
            save_results_to_files(original_data, results_by_order)

        # 实时控制台反馈：id 与 标准答案（或空）以及重试信息
        try:
            sa_display = result_entry.get("标准答案")
            # 将标准答案格式化为字符串显示（长度限制）
            sa_str = json.dumps(sa_display, ensure_ascii=False) if sa_display != "" else ""
        except Exception:
            sa_str = str(result_entry.get("标准答案", ""))
        
        retry_info = ""
        if result_entry.get("retry_count", 0) > 0:
            retry_info = f" | 重试了{result_entry.get('retry_count')}次"
        
        status = "❌" if result_entry.get("error") else "✅"
        
        # 显示哪些级别会被更新
        original_item = original_data[order]
        levels_to_update = ["L1"]
        if original_item.get("L2_id") and original_item.get("L2_问题"):
            levels_to_update.append("L2")
        if original_item.get("L3_id") and original_item.get("L3_问题"):
            levels_to_update.append("L3")
        levels_str = "+".join(levels_to_update)
        
        print(f"{status} 完成任务 id={result_entry.get('id')} [{levels_str}] | 执行时间={result_entry.get('执行时间')} | 标准答案={sa_str} | 总耗时 {result_entry.get('total_duration')}{retry_info}")


async def main():
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    lock = asyncio.Lock()

    # 读取 JSON 文件并转换字段名
    original_data = []
    rows = []
    try:
        with open(INPUT_JSON, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            
        # 如果是列表格式
        if isinstance(json_data, list):
            original_data = json_data  # 保存原始数据结构
            
            for idx, item in enumerate(json_data):
                # 字段映射：JSON字段 -> 内部使用的字段名
                converted_row = {
                    "id": item.get("L1_id", ""),
                    "类型": item.get("类型", ""),
                    "问题": item.get("L1_问题", ""),
                    "工作流": item.get("工作流", "")
                }
                rows.append((idx, converted_row))
        else:
            print(f"❌ JSON 文件格式错误：期望是数组格式")
            return
            
    except Exception as e:
        print(f"❌ 无法读取 JSON 文件: {e}")
        return

    results_by_order = {}  # order_index -> result_entry
    failed_ids = []

    # 创建并发任务
    tasks = []
    for order, row in rows:
        tasks.append(run_single(row, order, sem, lock, original_data, results_by_order, failed_ids))

    # 并发执行所有任务（不会因为单个任务失败而抛出异常）
    await asyncio.gather(*tasks)

    # 最终输出失败的 id 列表
    if failed_ids:
        # 去重并按出现顺序输出
        seen = set()
        deduped = [x for x in failed_ids if not (x in seen or seen.add(x))]
        print("\n❗ 以下任务执行出错（JSON 中的 L1_id）：")
        print(", ".join(map(str, deduped)))
    else:
        print("\n🎉 所有任务均成功（没有检测到 error 标记）。")

    print(f"\n最终结果文件保存在：{OUTPUT_JSON}")


if __name__ == "__main__":
    asyncio.run(main())