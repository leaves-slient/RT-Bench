#!/bin/bash

# 获取标准答案的bash脚本
# 功能：执行Python脚本获取标准答案，并保存到带时间戳的目录中

# 注意：不在这里使用 set -e，因为我们需要检查Python脚本的执行结果

# ========== 配置 ==========
# 标准答案使用的数据集（一般是整合好的包含L1/L2/L3及工作流的JSON）
# 如果环境变量 SINGLE_QUESTION_JSON 设置了，则使用该临时文件（只包含单个题目）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_INPUT_JSON="${SCRIPT_DIR}/../dataset/realtime-benchmark.json"
INPUT_JSON="${SINGLE_QUESTION_JSON:-$DEFAULT_INPUT_JSON}"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_realtime_flowl.py"

# 结果根目录，可以通过环境变量 RESULTS_BASE 覆盖
DEFAULT_RESULTS_DIR="${SCRIPT_DIR}/../results"
RESULTS_DIR="${RESULTS_BASE:-$DEFAULT_RESULTS_DIR}"
# =========================

# 时间戳：
# - 默认使用当前日期（格式：YYYY-MM-DD）
# - 如果外部通过 RUN_TS/RUN_TIMESTAMP 传入，则优先使用，以便和推理阶段的目录对齐
if [ -n "${RUN_TS:-}" ]; then
    TIMESTAMP="${RUN_TS}"
elif [ -n "${RUN_TIMESTAMP:-}" ]; then
    TIMESTAMP="${RUN_TIMESTAMP}"
else
    TIMESTAMP=$(date +"%Y-%m-%d")
fi

TIMESTAMP_DIR="${RESULTS_DIR}/${TIMESTAMP}"

# 创建时间戳目录
mkdir -p "${TIMESTAMP_DIR}"

# 获取模型名称（如果设置了，则为该模型生成独立的答案文件）
MODEL_NAME="${MODEL_NAME:-}"
MODEL_ANSWERS_FILE="${MODEL_ANSWERS_FILE:-}"

if [ -n "${MODEL_NAME}" ] && [ -n "${MODEL_ANSWERS_FILE}" ]; then
    # 按模型生成独立的答案文件
    OUTPUT_JSON="${MODEL_ANSWERS_FILE}"
else
    # 默认输出文件路径
    OUTPUT_JSON="${TIMESTAMP_DIR}/results-${TIMESTAMP}.json"
fi

# 确保 OUTPUT_JSON 不为空（防止后续错误）
if [ -z "${OUTPUT_JSON}" ]; then
    echo "⚠️  警告: OUTPUT_JSON 为空，使用默认路径" >&2
    OUTPUT_JSON="${TIMESTAMP_DIR}/results-${TIMESTAMP}.json"
fi

# echo "=========================================="
# echo "开始获取标准答案"
# echo "=========================================="
# if [ -n "${SINGLE_QUESTION_JSON:-}" ]; then
#     echo "模式: 单题目处理"
#     echo "临时题目文件: ${INPUT_JSON}"
# else
#     echo "模式: 全数据集处理"
#     echo "输入文件: ${INPUT_JSON}"
# fi
# echo "输出目录: ${TIMESTAMP_DIR}"
# echo "输出文件: ${OUTPUT_JSON}"
# echo "时间戳: ${TIMESTAMP}"
# echo "=========================================="

# 固定使用当前环境中的 python，避免 Git Bash 下 python3 指向 WindowsApps 的问题
PYTHON_BIN="python"

# Cross-platform directory lock (works on Linux and Windows Git Bash).
acquire_lock() {
    local lock_base="$1"
    local timeout_seconds="${2:-300}"
    local lock_dir="${lock_base}.d"
    local waited=0
    while [ "${waited}" -lt "${timeout_seconds}" ]; do
        if mkdir "${lock_dir}" 2>/dev/null; then
            echo "${lock_dir}"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    return 1
}

release_lock() {
    local lock_dir="$1"
    [ -n "${lock_dir}" ] && rmdir "${lock_dir}" 2>/dev/null || true
}

# 验证 OUTPUT_JSON 不为空
if [ -z "${OUTPUT_JSON}" ]; then
    echo "❌ 错误: OUTPUT_JSON 为空，无法继续执行" >&2
    exit 1
fi

# 检查conda环境是否激活（如果未激活，可以在这里激活）
# 如果需要激活特定环境，取消下面的注释并修改环境名称
# eval "$(conda shell.bash hook)"
# conda activate your_env_name

# 执行Python脚本
echo ""
# echo "正在执行Python脚本..."
# 直接通过环境变量把路径传给 Python，避免 sed/grep 在 Windows 路径（含反斜杠）下出错
export INPUT_JSON
export OUTPUT_JSON
if "${PYTHON_BIN}" "${PYTHON_SCRIPT}"; then
    PYTHON_SUCCESS=true
else
    PYTHON_SUCCESS=false
fi

# 检查执行是否成功
if [ "$PYTHON_SUCCESS" = true ]; then
    # echo ""
    # echo "=========================================="
    # echo "✅ Python脚本执行成功！"
    # echo "=========================================="
    
    # 如果是临时文件模式，清理临时文件
    if [ -n "${SINGLE_QUESTION_JSON:-}" ] && [ -f "${SINGLE_QUESTION_JSON}" ]; then
        echo "清理临时题目文件: ${SINGLE_QUESTION_JSON}"
        rm -f "${SINGLE_QUESTION_JSON}"
    fi
    
    # 检查输出文件是否存在
    if [ -f "${OUTPUT_JSON}" ]; then
        echo "输出文件已保存: ${OUTPUT_JSON}"
        
        # 从输出JSON中提取标准答案，保存到一个简化的文件中
        # 答案文件按级别分别存储：L1-{model_name}_answers.json, L2-{model_name}_answers.json, L3-{model_name}_answers.json
        if [ -n "${MODEL_NAME}" ] && [ -n "${MODEL_ANSWERS_FILE}" ]; then
            # 按模型和级别生成独立的答案文件
            STANDARD_ANSWERS_FILE="${MODEL_ANSWERS_FILE%.json}_standard.json"
            MODEL_MAIN_ANSWERS_FILE="${MODEL_ANSWERS_FILE}"
        else
            # 默认标准答案文件
            STANDARD_ANSWERS_FILE="${TIMESTAMP_DIR}/standard_answers-${TIMESTAMP}.json"
            MODEL_MAIN_ANSWERS_FILE=""
        fi
        
        # 如果设置了模型答案文件，同时更新主答案文件（包含执行时间等完整信息）
        if [ -n "${MODEL_MAIN_ANSWERS_FILE}" ]; then
            echo "正在更新主答案文件..."
            QUESTION_LEVEL="${QUESTION_LEVEL:-L1}"
            # 使用跨平台目录锁防止并发写入
            LOCK_FILE="${MODEL_MAIN_ANSWERS_FILE}.lock"
            LOCK_DIR="$(acquire_lock "${LOCK_FILE}" 300)"
            if [ -z "${LOCK_DIR}" ]; then
                echo "❌ 错误: 获取主答案文件锁超时: ${LOCK_FILE}" >&2
                exit 1
            fi

            # 通过环境变量传参，避免 Windows 路径反斜杠在 Python 字符串里被转义
            export OUTPUT_JSON_PATH_FOR_PY="${OUTPUT_JSON}"
            export MODEL_MAIN_ANSWERS_FILE_FOR_PY="${MODEL_MAIN_ANSWERS_FILE}"
            export QUESTION_LEVEL_FOR_PY="${QUESTION_LEVEL}"

            "${PYTHON_BIN}" << MAIN_EOF
import json
import os

output_json_path = os.environ["OUTPUT_JSON_PATH_FOR_PY"]
model_main_answers_file = os.environ["MODEL_MAIN_ANSWERS_FILE_FOR_PY"]
question_level = os.environ.get("QUESTION_LEVEL_FOR_PY", "L1")

with open(output_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 读取现有的主答案文件（如果存在）
existing_main_answers = []
if os.path.exists(model_main_answers_file):
    try:
        with open(model_main_answers_file, "r", encoding="utf-8") as f:
            existing_main_answers = json.load(f)
        if not isinstance(existing_main_answers, list):
            existing_main_answers = []
    except Exception as e:
        print(f"⚠️ 读取现有主答案文件失败: {e}，将创建新文件")
        existing_main_answers = []

# 创建现有答案的索引（根据 key_id 或对应级别的ID）
# 由于每个级别有独立的答案文件，只需要检查当前级别的ID即可
existing_main_index = {}
for idx, item in enumerate(existing_main_answers):
    key_id = item.get("key_id")
    # 根据level获取对应的ID（当前级别的文件只包含当前级别的答案）
    if question_level == "L1":
        level_id = item.get("L1_id")
    elif question_level == "L2":
        level_id = item.get("L2_id")
    elif question_level == "L3":
        level_id = item.get("L3_id")
    else:
        level_id = item.get("L1_id")
    
    if key_id:
        existing_main_index[f"key_{key_id}"] = idx
    if level_id:
        existing_main_index[f"{question_level.lower()}_{level_id}"] = idx

# 合并新答案到主答案文件
for item in data:
    # 根据level获取对应级别的信息
    if question_level == "L1":
        level_id = item.get("L1_id", "")
        level_question = item.get("L1_问题", "") or item.get("question", "")
        level_answer = item.get("L1_标准答案", "") or item.get("answer", "")
        level_exec_time = item.get("L1_执行时间", "")
        level_answer_field = "L1_标准答案"
        level_time_field = "L1_执行时间"
    elif question_level == "L2":
        level_id = item.get("L2_id", "")
        level_question = item.get("L2_问题", "")
        level_answer = item.get("L2_标准答案", "")
        level_exec_time = item.get("L2_执行时间", "")
        level_answer_field = "L2_标准答案"
        level_time_field = "L2_执行时间"
    elif question_level == "L3":
        level_id = item.get("L3_id", "")
        level_question = item.get("L3_问题", "")
        level_answer = item.get("L3_标准答案", "")
        level_exec_time = item.get("L3_执行时间", "")
        level_answer_field = "L3_标准答案"
        level_time_field = "L3_执行时间"
    else:
        level_id = item.get("L1_id", "")
        level_question = item.get("L1_问题", "") or item.get("question", "")
        level_answer = item.get("L1_标准答案", "") or item.get("answer", "")
        level_exec_time = item.get("L1_执行时间", "")
        level_answer_field = "L1_标准答案"
        level_time_field = "L1_执行时间"
    
    # 构造主答案项（由于每个级别有独立的答案文件，只保存当前级别的信息）
    main_answer_item = {
        "key_id": item.get("key_id", ""),
    }
    
    # 根据level添加对应级别的信息（只添加当前级别的信息，不包含其他级别）
    if question_level == "L1":
        main_answer_item["L1_id"] = level_id
        main_answer_item["question"] = level_question
        main_answer_item["answer"] = level_answer
        main_answer_item["L1_执行时间"] = level_exec_time
        main_answer_item["L1_标准答案"] = level_answer
        # L1文件只包含L1的信息，不包含L2、L3
    elif question_level == "L2":
        main_answer_item["L2_id"] = level_id
        main_answer_item["L2_问题"] = level_question
        main_answer_item["L2_执行时间"] = level_exec_time
        main_answer_item["L2_标准答案"] = level_answer
        # L2文件只包含L2的信息，不包含L1、L3
    elif question_level == "L3":
        main_answer_item["L3_id"] = level_id
        main_answer_item["L3_问题"] = level_question
        main_answer_item["L3_执行时间"] = level_exec_time
        main_answer_item["L3_标准答案"] = level_answer
        # L3文件只包含L3的信息，不包含L1、L2
    
    # 检查是否已存在该题目的答案
    key_id = main_answer_item.get("key_id")
    existing_idx = None
    
    if key_id and f"key_{key_id}" in existing_main_index:
        existing_idx = existing_main_index[f"key_{key_id}"]
    elif level_id and f"{question_level.lower()}_{level_id}" in existing_main_index:
        existing_idx = existing_main_index[f"{question_level.lower()}_{level_id}"]
    
    if existing_idx is not None:
        # 更新现有答案（根据level更新对应级别的答案）
        existing_item = existing_main_answers[existing_idx]
        # 只更新当前级别的字段，不添加其他级别的字段
        if question_level == "L1":
            if main_answer_item.get("L1_id"):
                existing_item["L1_id"] = main_answer_item["L1_id"]
            if main_answer_item.get("question"):
                existing_item["question"] = main_answer_item["question"]
            if main_answer_item.get("answer"):
                existing_item["answer"] = main_answer_item["answer"]
            if main_answer_item.get("L1_执行时间"):
                existing_item["L1_执行时间"] = main_answer_item["L1_执行时间"]
            if main_answer_item.get("L1_标准答案"):
                existing_item["L1_标准答案"] = main_answer_item["L1_标准答案"]
            # 确保不包含L2、L3字段（如果存在则删除）
            if "L2_id" in existing_item:
                del existing_item["L2_id"]
            if "L2_问题" in existing_item:
                del existing_item["L2_问题"]
            if "L2_执行时间" in existing_item:
                del existing_item["L2_执行时间"]
            if "L2_标准答案" in existing_item:
                del existing_item["L2_标准答案"]
            if "L3_id" in existing_item:
                del existing_item["L3_id"]
            if "L3_问题" in existing_item:
                del existing_item["L3_问题"]
            if "L3_执行时间" in existing_item:
                del existing_item["L3_执行时间"]
            if "L3_标准答案" in existing_item:
                del existing_item["L3_标准答案"]
        elif question_level == "L2":
            for key in ["L2_id", "L2_问题", "L2_执行时间", "L2_标准答案"]:
                if main_answer_item.get(key):
                    existing_item[key] = main_answer_item[key]
            # 确保不包含L1、L3字段（如果存在则删除）
            for key in ["L1_id", "question", "answer", "L1_执行时间", "L1_标准答案"]:
                if key in existing_item:
                    del existing_item[key]
            for key in ["L3_id", "L3_问题", "L3_执行时间", "L3_标准答案"]:
                if key in existing_item:
                    del existing_item[key]
        elif question_level == "L3":
            for key in ["L3_id", "L3_问题", "L3_执行时间", "L3_标准答案"]:
                if main_answer_item.get(key):
                    existing_item[key] = main_answer_item[key]
            # 确保不包含L1、L2字段（如果存在则删除）
            for key in ["L1_id", "question", "answer", "L1_执行时间", "L1_标准答案"]:
                if key in existing_item:
                    del existing_item[key]
            for key in ["L2_id", "L2_问题", "L2_执行时间", "L2_标准答案"]:
                if key in existing_item:
                    del existing_item[key]
    else:
        # 添加新答案
        existing_main_answers.append(main_answer_item)

# 保存合并后的主答案文件
with open(model_main_answers_file, "w", encoding="utf-8") as f:
    json.dump(existing_main_answers, f, ensure_ascii=False, indent=2)

print(f"主答案文件已保存到: {model_main_answers_file}")
print(f"  总题目数: {len(existing_main_answers)}")
MAIN_EOF
            MAIN_EXIT_CODE=$?
            release_lock "${LOCK_DIR}"
            if [ ${MAIN_EXIT_CODE} -ne 0 ]; then
                echo "❌ 错误: 更新主答案文件失败，退出码: ${MAIN_EXIT_CODE}" >&2
                exit ${MAIN_EXIT_CODE}
            fi
        fi
        
        echo "正在提取标准答案..."
        QUESTION_LEVEL="${QUESTION_LEVEL:-L1}"
        # 使用跨平台目录锁防止并发写入
        STANDARD_LOCK_FILE="${STANDARD_ANSWERS_FILE}.lock"
        STANDARD_LOCK_DIR="$(acquire_lock "${STANDARD_LOCK_FILE}" 300)"
        if [ -z "${STANDARD_LOCK_DIR}" ]; then
            echo "❌ 错误: 获取标准答案文件锁超时: ${STANDARD_LOCK_FILE}" >&2
            exit 1
        fi

        # 通过环境变量传参，避免 Windows 路径反斜杠在 Python 字符串里被转义
        export OUTPUT_JSON_PATH_FOR_PY="${OUTPUT_JSON}"
        export STANDARD_ANSWERS_FILE_FOR_PY="${STANDARD_ANSWERS_FILE}"
        export QUESTION_LEVEL_FOR_PY="${QUESTION_LEVEL}"

        "${PYTHON_BIN}" << EOF
import json
import os

# 读取输出JSON文件
output_json_path = os.environ["OUTPUT_JSON_PATH_FOR_PY"]
standard_answers_file = os.environ["STANDARD_ANSWERS_FILE_FOR_PY"]
question_level = os.environ.get("QUESTION_LEVEL_FOR_PY", "L1")

with open(output_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 读取现有的标准答案文件（如果存在）
existing_answers = []
if os.path.exists(standard_answers_file):
    try:
        with open(standard_answers_file, "r", encoding="utf-8") as f:
            existing_answers = json.load(f)
        if not isinstance(existing_answers, list):
            existing_answers = []
    except Exception as e:
        print(f"⚠️ 读取现有答案文件失败: {e}，将创建新文件")
        existing_answers = []

# 创建现有答案的索引（根据 key_id 或对应级别的ID）
# 由于每个级别有独立的答案文件，只需要检查当前级别的ID即可
existing_index = {}
for idx, item in enumerate(existing_answers):
    key_id = item.get("key_id")
    # 根据level获取对应的ID（当前级别的文件只包含当前级别的答案）
    if question_level == "L1":
        level_id = item.get("L1_id")
    elif question_level == "L2":
        level_id = item.get("L2_id")
    elif question_level == "L3":
        level_id = item.get("L3_id")
    else:
        level_id = item.get("L1_id")
    
    if key_id:
        existing_index[f"key_{key_id}"] = idx
    if level_id:
        existing_index[f"{question_level.lower()}_{level_id}"] = idx

# 提取标准答案（只保留ID和标准答案字段）
# 由于每个级别有独立的答案文件，只需要保存当前级别的信息
# 同时添加执行时间字段
for item in data:
    # 根据level获取对应级别的信息（包括执行时间）
    if question_level == "L1":
        level_id = item.get("L1_id", "")
        level_question = item.get("L1_问题", "")
        level_answer = item.get("L1_标准答案", "")
        level_exec_time = item.get("L1_执行时间", "")
        answer_item = {
            "key_id": item.get("key_id", ""),
            "L1_id": level_id,
            "L1_问题": level_question,
            "L1_标准答案": level_answer,
            "L1_执行时间": level_exec_time,  # 添加执行时间
        }
    elif question_level == "L2":
        level_id = item.get("L2_id", "")
        level_question = item.get("L2_问题", "")
        level_answer = item.get("L2_标准答案", "")
        level_exec_time = item.get("L2_执行时间", "")
        answer_item = {
            "key_id": item.get("key_id", ""),
            "L2_id": level_id,
            "L2_问题": level_question,
            "L2_标准答案": level_answer,
            "L2_执行时间": level_exec_time,  # 添加执行时间
        }
    elif question_level == "L3":
        level_id = item.get("L3_id", "")
        level_question = item.get("L3_问题", "")
        level_answer = item.get("L3_标准答案", "")
        level_exec_time = item.get("L3_执行时间", "")
        answer_item = {
            "key_id": item.get("key_id", ""),
            "L3_id": level_id,
            "L3_问题": level_question,
            "L3_标准答案": level_answer,
            "L3_执行时间": level_exec_time,  # 添加执行时间
        }
    else:
        level_id = item.get("L1_id", "")
        level_question = item.get("L1_问题", "")
        level_answer = item.get("L1_标准答案", "")
        level_exec_time = item.get("L1_执行时间", "")
        answer_item = {
            "key_id": item.get("key_id", ""),
            "L1_id": level_id,
            "L1_问题": level_question,
            "L1_标准答案": level_answer,
            "L1_执行时间": level_exec_time,  # 添加执行时间
        }
    
    # 检查是否已存在该题目的答案
    key_id = answer_item.get("key_id")
    existing_idx = None
    
    if key_id and f"key_{key_id}" in existing_index:
        existing_idx = existing_index[f"key_{key_id}"]
    elif level_id and f"{question_level.lower()}_{level_id}" in existing_index:
        existing_idx = existing_index[f"{question_level.lower()}_{level_id}"]
    
    if existing_idx is not None:
        # 更新现有答案（根据level更新对应级别的答案）
        existing_item = existing_answers[existing_idx]
        # 只更新当前级别的字段
        for key, value in answer_item.items():
            if value:  # 只更新非空值
                existing_item[key] = value
        # 确保不包含其他级别的字段（如果存在则删除）
        if question_level == "L1":
            for key in ["L2_id", "L2_问题", "L2_标准答案", "L2_执行时间", "L3_id", "L3_问题", "L3_标准答案", "L3_执行时间"]:
                if key in existing_item:
                    del existing_item[key]
        elif question_level == "L2":
            for key in ["L1_id", "L1_问题", "L1_标准答案", "L1_执行时间", "L3_id", "L3_问题", "L3_标准答案", "L3_执行时间"]:
                if key in existing_item:
                    del existing_item[key]
        elif question_level == "L3":
            for key in ["L1_id", "L1_问题", "L1_标准答案", "L1_执行时间", "L2_id", "L2_问题", "L2_标准答案", "L2_执行时间"]:
                if key in existing_item:
                    del existing_item[key]
    else:
        # 添加新答案
        existing_answers.append(answer_item)

# 保存合并后的标准答案到文件
with open(standard_answers_file, "w", encoding="utf-8") as f:
    json.dump(existing_answers, f, ensure_ascii=False, indent=2)

print(f"标准答案已保存到: {standard_answers_file}")
print(f"  总题目数: {len(existing_answers)}")
EOF
        STANDARD_EXIT_CODE=$?
        release_lock "${STANDARD_LOCK_DIR}"
        if [ ${STANDARD_EXIT_CODE} -ne 0 ]; then
            echo "❌ 错误: 提取标准答案失败，退出码: ${STANDARD_EXIT_CODE}" >&2
            exit ${STANDARD_EXIT_CODE}
        fi
        
        echo "✅ 标准答案提取完成: ${STANDARD_ANSWERS_FILE}"
    else
        echo "⚠️  警告：输出文件不存在: ${OUTPUT_JSON}"
    fi
else
    echo ""
    echo "=========================================="
    echo "❌ Python脚本执行失败！"
    echo "=========================================="
    exit 1
fi

# echo ""
# echo "=========================================="
# echo "完成！"
# echo "结果保存在: ${TIMESTAMP_DIR}"
# echo "=========================================="

