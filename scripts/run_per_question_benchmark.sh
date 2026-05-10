#!/bin/bash

# =============================================================================
# 一键按题多模型运行 + 标准答案生成 启动脚本
# -----------------------------------------------------------------------------
# 使用方式：
#   1. 编辑 scripts/models_config.yaml，配置好模型 API、启用状态、selected_models、selected_levels
#   2. 可选：修改本脚本中的基础路径（一般无需改动）
#   3. 运行：
#        bash scripts/run_per_question_benchmark.sh
#
# 本脚本会：
#   - 生成一个本次运行的时间戳 RUN_TS
#   - 将 RUN_TS 通过环境变量传递给 Python 控制脚本和标准答案脚本
#   - 使模型输出目录和标准答案目录都落在 {results_root}/{RUN_TS} 下，便于区分不同批次
# =============================================================================

set -euo pipefail

# 当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 项目根目录（根据你当前结构，这里是 RealtimeBench 的根）
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 结果根目录（与 models_config.yaml 中的 output_base 保持一致即可）
# 可以通过环境变量配置，例如: export RESULTS_ROOT="/path/to/results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/../results}"

# 生成本次运行的日期目录（以天为单位，格式：YYYY-MM-DD）
RUN_TS="$(date +"%Y-%m-%d")"
# RUN_TS="2026-01-10"                     # 指定日期


echo "=========================================="
echo "按题多模型基准测试一键启动脚本"
echo "=========================================="
echo "项目根目录: ${PROJECT_ROOT}"
echo "结果根目录: ${RESULTS_ROOT}"
echo "本次运行日期目录 (RUN_TS): ${RUN_TS}"
echo "本次运行结果目录: ${RESULTS_ROOT}/${RUN_TS}"
echo "使用的配置文件: ${SCRIPT_DIR}/models_config.yaml"
echo "=========================================="
echo ""

# 将时间戳和结果根目录传递给后续脚本
export RUN_TS
export RUN_TIMESTAMP="${RUN_TS}"
export RESULTS_BASE="${RESULTS_ROOT}"

# 进入项目根目录后再运行 Python 控制脚本，避免相对路径混乱
cd "${PROJECT_ROOT}"

echo ">>> 开始按题多模型推理流程（Python 控制脚本）..."
python "${SCRIPT_DIR}/run_multi_models_per_question.py"

echo ""
echo ">>> 整个按题多模型流程已结束。"
echo ">>> 所有模型输出和标准答案请在目录中查看: ${RESULTS_ROOT}/${RUN_TS}"
echo "=========================================="


