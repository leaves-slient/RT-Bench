import json
import os
import re
from pydantic import BaseModel
from typing import Literal
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from collections import Counter
from datetime import datetime
import glob
from pathlib import Path

def natural_sort_key(text):
    """自然排序的键函数，支持字母和数字混合排序"""
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    
    # 将字符串分割成数字和非数字部分
    return [convert(c) for c in re.split('([0-9]+)', text)]

# ========== 基础配置 ==========
# 这些配置现在通过命令行参数传入
RESULTS_DIR = None  # 待评估结果目录
OUTPUT_DIR = None   # 评估结果输出目录

# 模型与密钥设置
MODEL_NAME = "gpt-5"
# API Token should be set via environment variable LLM_API_TOKEN
# Example: export LLM_API_TOKEN="your-token-here"
if "LLM_API_TOKEN" not in os.environ:
    raise ValueError("Please set LLM_API_TOKEN environment variable")

client = OpenAI(
    base_url="https://api.openai.com/v1",
    api_key=os.environ["LLM_API_TOKEN"],
    timeout=600.0,  # 设置超时时间，与 react_agent.py 保持一致
    max_retries=3  # 设置最大重试次数
)

# ========== Pydantic模型定义 ==========
class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]

# ========== Prompt模板 ==========
JUDGE_PROMPT = """请根据下方给出的明确且无歧义的 [correct_answer]，判断以下 [response] 对 [question] 的回答是否正确。

[question]：{question}

[response]：{response}

你的判断必须符合以下格式与标准：

extracted_final_answer：从 [response] 中提取出的最终且确切的答案。如果在 response 中找不到明确的最终答案，请写 "None"。

[correct_answer]：{correct_answer}

reasoning：说明为什么 extracted_final_answer 与 [correct_answer] 相比是正确或错误的，仅关注两者之间是否存在实质性的差异。不要讨论题目背景，不要尝试重新计算或解题，也不要为与 [correct_answer] 不同的答案进行辩护，只需判断答案是否一致。

correct：如果 extracted_final_answer 与给出的 [correct_answer] 完全一致，或在数值题中处于允许的微小误差范围（四舍五入）内，则填 "yes"；否则（存在任何不一致、歧义、非等价或答案错误）填 "no"。"""

# ========== 核心函数部分 ==========

def call_model(question: str, answer: str, standard: str) -> dict:
    """调用 OpenAI 模型判断回答是否正确"""
    prompt = JUDGE_PROMPT.format(question=question, response=answer, correct_answer=standard)
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=4096
        )

        message = completion.choices[0].message
        reply = (message.content or "").strip()  # 安全访问，避免 None
        print(f"模型原始回复: {reply}")
        
        # 解析回复
        result = parse_model_reply(reply)
        print(f"解析结果: {result}")
        
        # 添加原始输出到结果中
        result["raw_output"] = reply
        
        return result
    
    except (APIError, APIConnectionError, APITimeoutError) as e:
        print(f"❌ API/网络错误: {e}")
        return {
            "extracted_final_answer": "None",
            "reasoning": f"API调用失败: {str(e)}",
            "correct": "no",
            "raw_output": ""
        }
    except Exception as e:
        print(f"❌ 调用模型时出错: {e}")
        return {
            "extracted_final_answer": "None",
            "reasoning": f"模型调用失败: {str(e)}",
            "correct": "no",
            "raw_output": ""
        }

def parse_model_reply(reply: str) -> dict:
    """解析模型回复，提取结构化判断结果"""
    try:
        # 首先尝试解析JSON格式回复
        json_match = re.search(r'\{.*\}', reply, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed_json = json.loads(json_str)
            
            # 验证必需字段并规范化
            result = {
                "extracted_final_answer": str(parsed_json.get("extracted_final_answer", "None")),
                "reasoning": str(parsed_json.get("reasoning", "未提供理由")),
                "correct": "yes" if str(parsed_json.get("correct", "no")).lower() in ["yes", "true", "1"] else "no"
            }
            
            return result
    
    except json.JSONDecodeError:
        print(f"⚠️ JSON解析失败，尝试字段解析")
    
    # 解析字段列表格式（如你的模型输出格式）
    try:
        result = {
            "extracted_final_answer": "None",
            "reasoning": "解析失败",
            "correct": "no"
        }
        
        # 解析extracted_final_answer
        answer_patterns = [
            r'extracted_final_answer[：:]\s*([^\n\r]+)',
            r'extracted_final_answer\s*[：:]\s*([^\n\r]+)',
            r'提取的最终答案[：:]\s*([^\n\r]+)'
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, reply, re.IGNORECASE | re.MULTILINE)
            if match:
                result["extracted_final_answer"] = match.group(1).strip()
                break
        
        # 解析reasoning
        reasoning_patterns = [
            r'reasoning[：:]\s*(.*?)(?=\n\s*\w+[：:]|$)',
            r'理由[：:]\s*(.*?)(?=\n\s*\w+[：:]|$)',
            r'原因[：:]\s*(.*?)(?=\n\s*\w+[：:]|$)'
        ]
        for pattern in reasoning_patterns:
            match = re.search(pattern, reply, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                reasoning_text = match.group(1).strip()
                # 清理reasoning文本，移除多余的标记
                reasoning_text = re.sub(r'\[correct_answer\][：:][^\n]*\n?', '', reasoning_text)
                result["reasoning"] = reasoning_text
                break
        
        # 解析correct
        correct_patterns = [
            r'correct[：:]\s*(yes|no)',
            r'正确[：:]\s*(yes|no|是|否)',
            r'判断[：:]\s*(yes|no|正确|错误)'
        ]
        for pattern in correct_patterns:
            match = re.search(pattern, reply, re.IGNORECASE)
            if match:
                correct_value = match.group(1).lower()
                if correct_value in ['yes', '是', '正确']:
                    result["correct"] = "yes"
                else:
                    result["correct"] = "no"
                break
        
        print(f"🔍 字段解析结果: {result}")
        return result
        
    except Exception as e:
        print(f"⚠️ 字段解析也失败: {e}")
        # 作为最后的备用方案，尝试简单的文本匹配
        try:
            backup_result = {
                "extracted_final_answer": "None",
                "reasoning": reply[:500] + "..." if len(reply) > 500 else reply,
                "correct": "no"
            }
            
            # 简单匹配数字作为答案
            number_match = re.search(r'extracted_final_answer[：:]\s*(\d+)', reply)
            if number_match:
                backup_result["extracted_final_answer"] = number_match.group(1)
            
            # 简单匹配correct
            if re.search(r'correct[：:]\s*yes', reply, re.IGNORECASE):
                backup_result["correct"] = "yes"
            elif re.search(r'correct[：:]\s*no', reply, re.IGNORECASE):
                backup_result["correct"] = "no"
            
            return backup_result
            
        except:
            return {
                "extracted_final_answer": "None",
                "reasoning": f"完全解析失败: {reply[:200]}...",
                "correct": "no"
            }

def extract_time_from_messages(messages):
    """从messages中提取Current time"""
    if not messages:
        return None
    
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            # 查找 "Current time: 2026-01-11 11:27:30" 格式
            match = re.search(r'Current time:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', content)
            if match:
                return match.group(1)
    return None

def normalize_text(text):
    """规范化文本，用于比较"""
    if not text:
        return ""
    return text.strip()

def parse_time(time_str):
    """解析时间字符串为datetime对象，支持多种格式"""
    if not time_str:
        return None
    
    # 移除首尾空格
    time_str = time_str.strip()
    
    # 尝试多种时间格式
    formats = [
        "%Y-%m-%d %H:%M:%S",  # 2026-01-12 13:02:46
        "%Y年%m月%d日 %H时%M分%S秒",  # 2026年01月12日 13时02分46秒
        "%Y-%m-%d %H:%M",  # 2026-01-12 13:02
        "%Y/%m/%d %H:%M:%S",  # 2026/01/12 13:02:46
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt)
        except:
            continue
    
    return None

def find_nearest_answer(question_time_str, level, results_dir):
    """在所有模型的答案文件中找到距离问题回答时间最近的答案
    优先找后面的答案，如果没有后面的，就找前面的
    """
    question_time = parse_time(question_time_str)
    if not question_time:
        return None, None
    
    # 查找该级别所有模型的答案文件
    answer_files = glob.glob(os.path.join(results_dir, f"{level}-*_answers.json"))
    
    best_answer_after = None
    best_time_after = None
    min_time_diff_after = None
    
    best_answer_before = None
    best_time_before = None
    min_time_diff_before = None
    
    for answer_file in answer_files:
        try:
            with open(answer_file, "r", encoding="utf-8") as f:
                answers = json.load(f)
            
            question_key = f"{level}_问题"
            answer_key = "answer"
            exec_time_key = f"{level}_执行时间"
            
            for item in answers:
                exec_time_str = item.get(exec_time_key, "")
                if not exec_time_str:
                    continue
                
                exec_time = parse_time(exec_time_str)
                if not exec_time:
                    continue
                
                # 计算时间差
                time_diff = (exec_time - question_time).total_seconds()
                
                # 优先找后面的答案（time_diff > 0）
                if time_diff > 0:
                    if min_time_diff_after is None or time_diff < min_time_diff_after:
                        min_time_diff_after = time_diff
                        best_answer_after = item.get(answer_key, "")
                        best_time_after = exec_time_str
                # 如果没有后面的，找前面的（time_diff < 0）
                elif time_diff < 0:
                    if min_time_diff_before is None or time_diff > min_time_diff_before:
                        min_time_diff_before = time_diff
                        best_answer_before = item.get(answer_key, "")
                        best_time_before = exec_time_str
        except Exception as e:
            print(f"⚠️ 读取答案文件失败 {answer_file}: {e}")
            continue
    
    # 优先返回后面的答案，如果没有，返回前面的
    if best_answer_after:
        return best_answer_after, best_time_after
    elif best_answer_before:
        return best_answer_before, best_time_before
    else:
        return None, None

def get_standard_answer(question, level, standard_file):
    """从标准答案文件中获取答案和序号"""
    try:
        with open(standard_file, "r", encoding="utf-8") as f:
            standards = json.load(f)
        
        question_key = f"{level}_问题"
        answer_key = f"{level}_标准答案"
        exec_time_key = f"{level}_执行时间"
        id_key = f"{level}_id"
        
        for item in standards:
            std_question = item.get(question_key, "").strip()
            if std_question == question.strip():
                return item.get(answer_key, ""), item.get(exec_time_key, ""), item.get(id_key, "")
        
        return None, None, None
    except Exception as e:
        print(f"⚠️ 读取标准答案文件失败 {standard_file}: {e}")
        return None, None, None

def find_matching_result_by_question(question, results_data, level):
    """在results数据中通过问题文本找到匹配的条目"""
    question_key = f"{level}_问题"
    standard_answer_key = f"{level}_标准答案"
    exec_time_key = f"{level}_执行时间"
    id_key = f"{level}_id"
    
    question_normalized = normalize_text(question)
    
    for item in results_data:
        result_question = item.get(question_key, "")
        if normalize_text(result_question) == question_normalized:
            standard_answer = item.get(standard_answer_key, "")
            exec_time = item.get(exec_time_key, "")
            question_id = item.get(id_key, "")
            # 只返回非空的标准答案
            if standard_answer and normalize_text(standard_answer):
                return {
                    "standard_answer": standard_answer,
                    "exec_time": exec_time,
                    "question_id": question_id,
                    "level": level
                }
    return None

def find_best_match_by_time(question, question_time, candidates):
    """从候选结果中，按照时间找到最接近question_time的结果
    优先找question_time之后最近的时间，没有就找之前最近的时间
    """
    if not candidates:
        return None
    
    question_time_obj = parse_time(question_time)
    if not question_time_obj:
        # 如果无法解析question_time，返回第一个候选
        return candidates[0] if candidates else None
    
    # 解析所有候选的时间
    candidates_with_time = []
    for candidate in candidates:
        exec_time_obj = parse_time(candidate.get("exec_time", ""))
        if exec_time_obj:
            time_diff = (exec_time_obj - question_time_obj).total_seconds()
            candidates_with_time.append((time_diff, candidate))
    
    if not candidates_with_time:
        # 如果都无法解析时间，返回第一个
        return candidates[0] if candidates else None
    
    # 优先找后面的（time_diff > 0），如果没有就找前面的
    after_candidates = [(diff, cand) for diff, cand in candidates_with_time if diff > 0]
    before_candidates = [(diff, cand) for diff, cand in candidates_with_time if diff <= 0]
    
    if after_candidates:
        # 找后面最近的时间（时间差最小）
        after_candidates.sort(key=lambda x: x[0])
        return after_candidates[0][1]
    elif before_candidates:
        # 找前面最近的时间（时间差的绝对值最小，但diff是负数，所以找最大的diff）
        before_candidates.sort(key=lambda x: -x[0])
        return before_candidates[0][1]
    else:
        return None

def load_results_file(file_path):
    """加载results文件（JSON数组）"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 读取文件失败 {file_path}: {e}")
        return None

def find_standard_answer_from_other_models(question, question_time, level, results_dir, exclude_model_name=None):
    """从其他模型的标准答案文件中查找标准答案
    按照优先级查找：
    1. 同日期、同级别、其他模型（按时间排序，优先question_time之后的）
    2. 同日期、所有级别、其他模型（按时间排序，优先question_time之后的）
    
    返回: (standard_answer, exec_time, question_id, source_info)
    """
    # 策略1：同日期、同级别、其他模型
    level_results_files = glob.glob(os.path.join(results_dir, f"{level}-*_answers_standard.json"))
    candidates = []
    
    for file_path in level_results_files:
        # 排除当前模型
        if exclude_model_name and exclude_model_name in os.path.basename(file_path):
            continue
        
        results_data = load_results_file(file_path)
        if results_data:
            match = find_matching_result_by_question(question, results_data, level)
            if match:
                candidates.append(match)
    
    if candidates:
        best_match = find_best_match_by_time(question, question_time, candidates)
        if best_match:
            return (
                best_match["standard_answer"],
                best_match.get("exec_time", ""),
                best_match.get("question_id", ""),
                "同级别其他模型"
            )
    
    # 策略2：同日期、所有级别、其他模型
    all_levels = ["L1", "L2", "L3"]
    candidates = []
    
    for search_level in all_levels:
        level_results_files = glob.glob(os.path.join(results_dir, f"{search_level}-*_answers_standard.json"))
        for file_path in level_results_files:
            # 排除当前模型的所有级别文件
            if exclude_model_name and exclude_model_name in os.path.basename(file_path):
                continue
            
            results_data = load_results_file(file_path)
            if results_data:
                match = find_matching_result_by_question(question, results_data, search_level)
                if match:
                    candidates.append(match)
    
    if candidates:
        best_match = find_best_match_by_time(question, question_time, candidates)
        if best_match:
            source_level = best_match.get("level", "?")
            return (
                best_match["standard_answer"],
                best_match.get("exec_time", ""),
                best_match.get("question_id", ""),
                f"所有级别其他模型(来源级别:{source_level})"
            )
    
    return None, None, None, None

def save_result_to_jsonl(result, file_path):
    """追加一条结果到jsonl文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

def load_processed_questions(jsonl_file):
    """加载已处理的问题集合（用于断点续跑）"""
    processed = set()
    if os.path.exists(jsonl_file):
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            # 使用问题+文件名的组合作为唯一标识
                            question = item.get("question", "")
                            source_file = item.get("source_file", "")
                            if question and source_file:
                                processed.add((question, source_file))
                        except:
                            continue
        except Exception as e:
            print(f"⚠️ 加载已处理问题失败: {e}")
    return processed

def evaluate_all_models(results_dir, output_dir):
    """评估所有模型的结果"""
    # 创建输出目录，使用输入文件夹的名称作为输出文件夹名
    folder_name = os.path.basename(os.path.normpath(results_dir))
    output_base = os.path.join(output_dir, folder_name)
    os.makedirs(output_base, exist_ok=True)
    
    # 获取所有模型目录
    model_dirs = [d for d in os.listdir(results_dir) 
                  if os.path.isdir(os.path.join(results_dir, d)) and d.endswith("_sglang")]
    
    # 获取所有标准答案文件
    standard_files = {}
    for level in ["L1", "L2", "L3"]:
        standard_files[level] = glob.glob(os.path.join(results_dir, f"{level}-*_answers_standard.json"))
    
    # 遍历每个模型
    for model_dir in model_dirs:
        model_name = model_dir.replace("_sglang", "")
        print(f"\n{'='*60}")
        print(f"开始评估模型: {model_name}")
        print(f"{'='*60}")
        
        # 为每个级别创建输出文件
        level_output_files = {}
        processed_questions = {}
        
        for level in ["L1", "L2", "L3"]:
            level_lower = level.lower()
            level_folder = f"{level}_questions" if level == "L1" else f"{level_lower}_questions"
            
            output_file = os.path.join(output_base, model_name, f"{level}.jsonl")
            level_output_files[level] = output_file
            processed_questions[level] = load_processed_questions(output_file)
        
        # 遍历每个级别
        for level in ["L1", "L2", "L3"]:
            level_lower = level.lower()
            level_folder = f"{level}_questions" if level == "L1" else f"{level_lower}_questions"
            
            questions_dir = os.path.join(results_dir, model_dir, level_folder)
            if not os.path.exists(questions_dir):
                print(f"⚠️ 跳过：{questions_dir} 不存在")
                continue
            
            # 获取该模型的标准答案文件
            standard_file = None
            for sf in standard_files[level]:
                if model_name in sf:
                    standard_file = sf
                    break
            
            if not standard_file:
                print(f"⚠️ 跳过：未找到 {level} 级别的标准答案文件")
                continue
            
            print(f"\n处理 {level} 级别...")
            print(f"标准答案文件: {standard_file}")
            
            # 遍历该级别下的所有jsonl文件，使用自然排序
            jsonl_files = glob.glob(os.path.join(questions_dir, "*.jsonl"))
            print(f"找到 {len(jsonl_files)} 个jsonl文件")
            
            # 使用自然排序：按文件名排序（字母和数字混合排序）
            jsonl_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
            
            for jsonl_file in jsonl_files:
                print(f"\n处理文件: {os.path.basename(jsonl_file)}")
                
                try:
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            if not line.strip():
                                continue
                            
                            try:
                                item = json.loads(line)
                                question = item.get("question", "").strip()
                                prediction = item.get("prediction", "")
                                messages = item.get("messages", [])
                                
                                if not question:
                                    continue
                                
                                # 检查是否已处理
                                file_key = os.path.basename(jsonl_file)
                                if (question, file_key) in processed_questions[level]:
                                    print(f"⏭️ 跳过已处理: {question[:50]}...")
                                    continue
                                
                                # 获取标准答案和序号
                                std_answer, std_exec_time, question_id = get_standard_answer(question, level, standard_file)
                                
                                # 提取问题时间（用于时间匹配）
                                question_time = extract_time_from_messages(messages) or ""
                                
                                # 如果标准答案为空，尝试从其他模型的标准答案文件中找
                                answer_time = std_exec_time
                                if not std_answer or std_answer.strip() == "":
                                    # 从标准答案文件名中提取模型名称（用于排除当前模型）
                                    exclude_model_name = None
                                    if standard_file:
                                        # 从文件名中提取模型名称，例如：L2-aws-claude-sonnet-4-5_answers_standard.json -> aws-claude-sonnet-4-5
                                        match = re.search(rf'{level}-(.+?)_answers_standard\.json', os.path.basename(standard_file))
                                        if match:
                                            exclude_model_name = match.group(1)
                                    
                                    # 从其他模型的标准答案文件中查找标准答案（使用新的搜索策略）
                                    other_std_answer, other_exec_time, other_question_id, source_info = find_standard_answer_from_other_models(
                                        question, question_time, level, results_dir, exclude_model_name
                                    )
                                    if other_std_answer and other_std_answer.strip():
                                        std_answer = other_std_answer
                                        answer_time = other_exec_time or std_exec_time
                                        # 如果当前模型的标准答案文件中没有question_id，使用其他模型的
                                        if not question_id and other_question_id:
                                            question_id = other_question_id
                                        print(f"✅ 从其他模型的标准答案文件中找到标准答案 (来源: {source_info})")
                                
                                # 如果标准答案仍然为空，跳过（不纳入统计）
                                if not std_answer or std_answer.strip() == "":
                                    print(f"⚠️ 标准答案为空，跳过: {question[:50]}...")
                                    result = {
                                        "question_id": question_id or "",
                                        "question": question,
                                        "messages": messages,
                                        "prediction": prediction,
                                        "standard_answer": "",
                                        "answer_time": answer_time or "",
                                        "question_time": extract_time_from_messages(messages) or "",
                                        "evaluation_output": "",
                                        "evaluation_result": "skipped_no_standard_answer"
                                    }
                                    save_result_to_jsonl(result, level_output_files[level])
                                    processed_questions[level].add((question, file_key))
                                    continue
                                
                                # 如果prediction为空，跳过（不纳入统计）
                                if not prediction or (isinstance(prediction, str) and prediction.strip() == ""):
                                    print(f"⚠️ prediction为空，跳过: {question[:50]}...")
                                    result = {
                                        "question_id": question_id or "",
                                        "question": question,
                                        "messages": messages,
                                        "prediction": "",
                                        "standard_answer": std_answer,
                                        "answer_time": answer_time or "",
                                        "question_time": extract_time_from_messages(messages) or "",
                                        "evaluation_output": "",
                                        "evaluation_result": "skipped_no_prediction"
                                    }
                                    save_result_to_jsonl(result, level_output_files[level])
                                    processed_questions[level].add((question, file_key))
                                    continue
                                
                                # 进行评估
                                print(f"🔄 评估: {question[:50]}...")
                                eval_result = call_model(question, prediction, std_answer)
                                
                                # 保存结果
                                result = {
                                    "question_id": question_id or "",
                                    "question": question,
                                    "messages": messages,
                                    "prediction": prediction,
                                    "standard_answer": std_answer,
                                    "answer_time": answer_time or "",
                                    "question_time": extract_time_from_messages(messages) or "",
                                    "evaluation_output": eval_result.get("raw_output", ""),
                                    "evaluation_result": eval_result.get("correct", "no"),
                                    "extracted_final_answer": eval_result.get("extracted_final_answer", "None"),
                                    "reasoning": eval_result.get("reasoning", "")
                                }
                                
                                save_result_to_jsonl(result, level_output_files[level])
                                processed_questions[level].add((question, file_key))
                                
                                print(f"✅ 完成，判断: {eval_result.get('correct', 'no')}")
                                
                            except json.JSONDecodeError as e:
                                print(f"❌ JSON解析失败 (行 {line_num}): {e}")
                                continue
                            except Exception as e:
                                print(f"❌ 处理失败 (行 {line_num}): {e}")
                                import traceback
                                traceback.print_exc()
                                continue
                
                except Exception as e:
                    print(f"❌ 读取文件失败 {jsonl_file}: {e}")
                    continue
    
    # 统计所有结果
    print(f"\n{'='*60}")
    print("开始统计结果...")
    print(f"{'='*60}")
    
    stats_file = os.path.join(output_base, "统计结果.txt")
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("评估结果统计\n")
        f.write("="*60 + "\n\n")
        
        # 遍历所有模型
        for model_dir in model_dirs:
            model_name = model_dir.replace("_sglang", "")
            f.write(f"\n模型: {model_name}\n")
            f.write("-"*60 + "\n")
            
            total_correct = 0
            total_incorrect = 0
            total_skipped = 0
            total_empty_standard = 0
            total_empty_prediction = 0
            
            level_stats = {}
            
            for level in ["L1", "L2", "L3"]:
                output_file = os.path.join(output_base, model_name, f"{level}.jsonl")
                if not os.path.exists(output_file):
                    continue
                
                correct = 0
                incorrect = 0
                skipped = 0
                empty_standard = 0
                empty_prediction = 0
                
                try:
                    with open(output_file, "r", encoding="utf-8") as f_in:
                        for line in f_in:
                            if not line.strip():
                                continue
                            try:
                                item = json.loads(line)
                                result = item.get("evaluation_result", "")
                                
                                if result == "yes":
                                    correct += 1
                                elif result == "no":
                                    incorrect += 1
                                elif result == "skipped_no_standard_answer":
                                    empty_standard += 1
                                elif result == "skipped_no_prediction":
                                    empty_prediction += 1
                                else:
                                    skipped += 1
                            except:
                                continue
                except Exception as e:
                    print(f"⚠️ 统计 {output_file} 失败: {e}")
                    continue
                
                total = correct + incorrect
                acc = (correct / total * 100) if total > 0 else 0
                
                level_stats[level] = {
                    "total": total,
                    "correct": correct,
                    "incorrect": incorrect,
                    "acc": acc,
                    "empty_standard": empty_standard,
                    "empty_prediction": empty_prediction
                }
                
                total_correct += correct
                total_incorrect += incorrect
                total_skipped += skipped
                total_empty_standard += empty_standard
                total_empty_prediction += empty_prediction
                
                f.write(f"\n{level} 级别:\n")
                f.write(f"  总题目数: {total}\n")
                f.write(f"  正确: {correct}\n")
                f.write(f"  错误: {incorrect}\n")
                f.write(f"  准确率: {acc:.2f}%\n")
                if empty_standard > 0:
                    f.write(f"  标准答案为空: {empty_standard} 题\n")
                if empty_prediction > 0:
                    f.write(f"  prediction为空: {empty_prediction} 题\n")
            
            # 总体统计
            total_all = total_correct + total_incorrect
            acc_all = (total_correct / total_all * 100) if total_all > 0 else 0
            
            f.write(f"\n总体统计:\n")
            f.write(f"  总题目数: {total_all}\n")
            f.write(f"  正确: {total_correct}\n")
            f.write(f"  错误: {total_incorrect}\n")
            f.write(f"  准确率: {acc_all:.2f}%\n")
            if total_empty_standard > 0:
                f.write(f"  标准答案为空: {total_empty_standard} 题\n")
            if total_empty_prediction > 0:
                f.write(f"  prediction为空: {total_empty_prediction} 题\n")
    
    print(f"\n✅ 所有评估完成！统计结果已写入：{stats_file}")

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python evaluate.py <结果目录> [输出目录]")
        print("示例: python evaluate.py /path/to/results/2026-01-11 /path/to/eval_results")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    # 可以通过环境变量配置默认输出目录，例如: export EVAL_OUTPUT_DIR="/path/to/eval_results"
    default_output_dir = os.environ.get("EVAL_OUTPUT_DIR", "eval_results")
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_output_dir
    
    if not os.path.exists(results_dir):
        print(f"❌ 结果目录不存在: {results_dir}")
        sys.exit(1)
    
    print(f"结果目录: {results_dir}")
    print(f"输出目录: {output_dir}")
    
    evaluate_all_models(results_dir, output_dir)

if __name__ == "__main__":
    main()
