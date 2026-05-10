# RealtimeBench

实时性benchmark，实现实时问答能力评估。

## 项目简介

RealtimeBench 是一个用于评估多个大语言模型在实时问答任务上表现的基准测试框架。项目支持：

- **多模型并发测试**：同时测试多个 LLM 模型（如 GLM4.7、Qwen3、Kimi、DeepSeek、GPT5.2 等）
- **多难度级别**：支持 L1、L2、L3 三个难度级别的问题集
- **自动评估**：自动生成标准答案并评估模型回答的正确性

## 环境要求

- Python 3.10.0（推荐使用 conda 或 virtualenv 创建独立环境）
- 依赖安装基于 DeepResearch 项目

## 安装步骤

### 1. 创建虚拟环境

```bash
# 使用 conda
conda create -n realtimebench_env python=3.10.0
conda activate realtimebench_env

# 或使用 virtualenv
python3.10 -m venv realtimebench_env
source realtimebench_env/bin/activate
```

### 2. 安装依赖

```bash
# 安装 DeepResearch 项目的依赖
cd DeepResearch
pip install -r requirements.txt

# 安装 playwright（用于网页访问）
pip install playwright
playwright install
```

### 3. 配置环境

复制并编辑配置文件：

```bash
cp scripts/models_config.yaml.example scripts/models_config.yaml
```

编辑 `scripts/models_config.yaml`，配置：
- API 密钥（SERPER_KEY_ID、JINA_API_KEYS 等）
- 模型 API 地址和密钥
- 数据集路径
- 输出路径

## 使用方法

### 1. 运行基准测试

```bash
bash scripts/run_per_question_benchmark.sh
```

该脚本会：
- 读取 `scripts/models_config.yaml` 中的配置
- 对选定的模型和难度级别进行测试
- 将结果保存到 `results/{日期}/` 目录下

### 2. 评估结果

```bash
bash scripts/run_evaluation.sh results/2026-01-11 [输出目录]
```

评估脚本会：
- 读取指定结果目录中的模型输出
- 使用 LLM 判断答案正确性
- 生成评估报告

**注意**：评估需要设置环境变量 `LLM_API_TOKEN`：

```bash
export LLM_API_TOKEN="your-api-token"
```

## 项目结构

```
RealtimeBench-code/
├── DeepResearch/          # DeepResearch 项目代码
├── dataset/              # 测试数据集（L1/L2/L3）
├── scripts/              # 脚本目录
│   ├── run_per_question_benchmark.sh  # 启动脚本
│   ├── run_evaluation.sh              # 评估脚本
│   ├── models_config.yaml.example     # 配置示例
│   ├── run_multi_models_per_question.py  # 核心控制脚本
│   ├── evaluate.py                    # 评估脚本
│   ├── real-time-generalization.py    # 问题和工作流生成工具
│   └── workflow-fix.py                # 工作流修复工具
└── results/              # 结果输出目录
```

### 工具脚本说明

#### real-time-generalization.py

用于自动生成新的实时性问题和工作流代码的工具。主要功能：

- **问题生成**：基于示例问题和示例工作流，使用 LLM 生成新的实时性问题
- **工作流生成**：为每个新问题生成对应的可执行工作流代码（使用 playwright 等工具）
- **自动验证**：通过代码执行、网页截图验证等方式确保生成的工作流代码可执行
- **批量处理**：支持从 CSV 文件读取示例，批量生成多个新问题和工作流

使用场景：扩展测试数据集，生成更多实时性问题用于基准测试。

#### workflow-fix.py

用于修复已有但出现问题的的工作流代码的工具。主要功能：

- **问题诊断**：自动诊断工作流代码无法正常工作的原因（如网站结构变化、元素定位失效等）
- **代码修复**：使用 LLM 修复代码，确保能够重新获取实时数据
- **功能保持**：修复后保持原有功能和输出格式不变
- **批量修复**：支持批量处理多个错误的工作流，并生成修复日志

使用场景：当网站结构变化或工作流代码失效时，自动修复并更新工作流代码。

## 配置说明

在 `scripts/models_config.yaml` 中，你可以：

- **选择要运行的模型**：通过 `selected_models` 配置
- **选择难度级别**：通过 `selected_levels` 配置（L1/L2/L3）
- **调整并发数**：通过 `max_parallel_jobs` 和 `question_concurrency` 配置
- **启用/禁用模型**：通过 `enabled` 字段控制

## 数据集格式

数据集文件为 JSON 格式，包含 `question` 和 `answer` 字段：

```json
[
  {
    "key_id": "realtime0001",
    "L1_id": "0001",
    "question": "问题内容",
    ......
    "工作流": "Python代码"
  }
]
```

## 相关项目

本项目基于 [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) 开发。


