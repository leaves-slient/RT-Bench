# RealtimeBench

A real-time benchmarking project for evaluating the real-time question-answering capabilities of models.

## Overview

RealtimeBench is a benchmarking framework designed to evaluate the performance of multiple large language models in real-time question-answering tasks. The project supports:

- **Multi-model concurrent testing**: Simultaneously test multiple LLM models (such as GLM4.7, Qwen3, Kimi, DeepSeek, GPT5.2, etc.)
- **Multiple difficulty levels**: Supports question sets at three difficulty levels: L1, L2, and L3
- **Automated evaluation**: Automatically generates standard answers and evaluates the correctness of model responses

## Requirements

- Python 3.10.0 (recommended to use conda or virtualenv to create an isolated environment)
- Dependencies are based on the DeepResearch project

## Installation

### 1. Create Virtual Environment

```bash
# Using conda
conda create -n realtimebench_env python=3.10.0
conda activate realtimebench_env

# Or using virtualenv
python3.10 -m venv realtimebench_env
source realtimebench_env/bin/activate
```

### 2. Install Dependencies

```bash
# Install DeepResearch project dependencies
cd DeepResearch
pip install -r requirements.txt

# Install playwright (for web access)
pip install playwright
playwright install
```

### 3. Configure Environment

Copy and edit the configuration file:

```bash
cp scripts/models_config.yaml.example scripts/models_config.yaml
```

Edit `scripts/models_config.yaml` to configure:
- API keys (SERPER_KEY_ID, JINA_API_KEYS, etc.)
- Model API addresses and keys
- Dataset paths
- Output paths

## Usage

### 1. Run Benchmark

```bash
bash scripts/run_per_question_benchmark.sh
```

This script will:
- Read configuration from `scripts/models_config.yaml`
- Test selected models and difficulty levels
- Save results to the `results/{date}/` directory

### 2. Evaluate Results

```bash
bash scripts/run_evaluation.sh results/2026-01-11 [output_directory]
```

The evaluation script will:
- Read model outputs from the specified results directory
- Use LLM to judge answer correctness
- Generate evaluation reports

**Note**: Evaluation requires setting the `LLM_API_TOKEN` environment variable:

```bash
export LLM_API_TOKEN="your-api-token"
```

## Project Structure

```
RealtimeBench-code/
├── DeepResearch/          # DeepResearch project code
├── dataset/              # Test datasets (L1/L2/L3)
├── scripts/              # Scripts directory
│   ├── run_per_question_benchmark.sh  # Startup script
│   ├── run_evaluation.sh              # Evaluation script
│   ├── models_config.yaml.example     # Configuration example
│   ├── run_multi_models_per_question.py  # Core control script
│   ├── evaluate.py                    # Evaluation script
│   ├── real-time-generalization.py    # Question and workflow generation tool
│   └── workflow-fix.py                # Workflow repair tool
└── results/              # Results output directory
```

### Tool Scripts Description

#### real-time-generalization.py

A tool for automatically generating new real-time questions and workflow code. Main features:

- **Question generation**: Uses LLM to generate new real-time questions based on example questions and example workflows
- **Workflow generation**: Generates executable workflow code (using playwright and other tools) for each new question
- **Automatic verification**: Ensures generated workflow code is executable through code execution, webpage screenshot verification, etc.
- **Batch processing**: Supports reading examples from CSV files and batch generating multiple new questions and workflows

Use case: Expand the test dataset by generating more real-time questions for benchmarking.

#### workflow-fix.py

A tool for repairing existing but problematic workflow code. Main features:

- **Problem diagnosis**: Automatically diagnoses why workflow code cannot work properly (e.g., website structure changes, element positioning failures, etc.)
- **Code repair**: Uses LLM to fix code, ensuring real-time data can be retrieved again
- **Function preservation**: Maintains original functionality and output format after repair
- **Batch repair**: Supports batch processing of multiple error workflows and generating repair logs

Use case: Automatically repair and update workflow code when website structures change or workflow code becomes invalid.

## Configuration

In `scripts/models_config.yaml`, you can:

- **Select models to run**: Configure through `selected_models`
- **Select difficulty levels**: Configure through `selected_levels` (L1/L2/L3)
- **Adjust concurrency**: Configure through `max_parallel_jobs` and `question_concurrency`
- **Enable/disable models**: Control through the `enabled` field

## Dataset Format

Dataset files are in JSON format, containing `question` and `answer` fields:

```json
[
  {
    "key_id": "realtime0001",
    "L1_id": "0001",
    "question": "Question content",
    ......
    "工作流": "Python code"
  }
]
```

## Related Projects

This project is developed based on [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch).
