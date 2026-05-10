#!/bin/bash

# 评估脚本启动器
# 用法: ./run_evaluation.sh <结果目录> [输出目录]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate.py"
# 可以通过环境变量配置默认输出目录，例如: export EVAL_OUTPUT_DIR="/path/to/eval_results"
DEFAULT_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-eval_results}"

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <结果目录> [输出目录]"
    echo "示例: $0 results/2026-01-11"
    echo "      $0 results/2026-01-11 /path/to/output"
    exit 1
fi

RESULTS_DIR="$1"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"

# 检查结果目录是否存在
if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ 错误: 结果目录不存在: $RESULTS_DIR"
    exit 1
fi

# 检查评估脚本是否存在
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "❌ 错误: 评估脚本不存在: $EVAL_SCRIPT"
    exit 1
fi

echo "=========================================="
echo "开始评估"
echo "=========================================="
echo "结果目录: $RESULTS_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "评估脚本: $EVAL_SCRIPT"
echo "=========================================="
echo ""

# 运行评估脚本
python "$EVAL_SCRIPT" "$RESULTS_DIR" "$OUTPUT_DIR"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 评估完成！"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ 评估失败，退出码: $EXIT_CODE"
    echo "=========================================="
    exit $EXIT_CODE
fi

