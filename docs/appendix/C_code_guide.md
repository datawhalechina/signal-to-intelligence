# 代码运行指南

**版本：** v1.3
**最后更新：** 2026-06-08

本指南是各章代码实验的依赖参考手册。环境配置和安装步骤见 [附录B](B_environment_setup.md)。

---

## 目录

1. [快速命令](#快速命令)
2. [依赖总览](#依赖总览)
3. [代码实验清单](#代码实验清单)
4. [常见问题](#常见问题)

---

## 快速命令

```bash
# 运行任意实验
python code/ch{N}_{topic}/{script}.py

# 运行所有测试
pytest tests/ -v

# 生成覆盖率报告
pytest tests/ --cov=code --cov-report=html

# 按章节运行（使用 Makefile）
make run-exp-ch01   # 离线，<5s
make run-exp-ch05   # 离线（8个脚本）
make run-exp-ch05-api  # 需要 llama.cpp / Ollama 或 API Key
make run-exp-ch08   # 离线，仅标准库
make test
```

---

## 🎯 快速判断：我能运行什么？

根据你的资源和需求，查看能运行哪些脚本：

### 场景 1️⃣：只有 CPU、完全离线

✅ **能运行：** 第 1-6 章默认实验、第 6 章扩展实验、第 7 章默认离线实验、第 8 章实验
❌ **不能运行：** 第 7 章重度脚本（需要下载模型权重）

```bash
pip install -e .
make run-exp-ch01 run-exp-ch02 run-exp-ch03 run-exp-ch04
make run-exp-ch05 run-exp-ch06 run-exp-ch06-extra run-exp-ch07 run-exp-ch08
```

### 场景 2️⃣：有国内 API（DeepSeek / 阿里云） 或 本地 llama.cpp / Ollama

✅ **额外能运行：** 第 5 章的 `llm_api_demo.py`  
**依赖：** `pip install -e ".[llm]"` + 设置 API Key 或启动 llama.cpp / Ollama 本地服务

```bash
export DEEPSEEK_API_KEY="your-key"
python code/ch05_llm_basics/llm_api_demo.py
```

### 场景 3️⃣：有 GPU + 能下载模型权重

✅ **额外能运行：** 第 7 章重度脚本（CLIP、Qwen2.5-VL）  
**依赖：** `pip install -e ".[hf]"` + 按脚本下载 CLIP 或 Qwen2.5-VL 权重

```bash
pip install -e ".[hf]"
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./models
make run-exp-ch07-heavy
```

### 场景 4️⃣：用云端环境（Google Colab / CNB）

✅ **都能运行**（Colab 有免费 T4 GPU）  
**推荐：** CNB（预配置所有依赖）

---

## 依赖总览

各章代码实验所需依赖一览。核心依赖（numpy / matplotlib / scipy / pytest）通过 `pip install -e .` 安装；可选依赖按章节按需安装。

| 章节 | 核心依赖 | 可选依赖 | API / 外部资源 |
|------|---------|---------|--------------|
| 第1章 DSP | numpy, matplotlib, scipy | — | — |
| 第2章 优化 | numpy, matplotlib | — | — |
| 第3章 深度学习 | numpy, matplotlib | — | — |
| 第4章 Transformer | numpy, matplotlib, networkx¹ | — | — |
| 第5章 LLM基础 | numpy, matplotlib, scipy | openai, anthropic² | llama.cpp / Ollama（本地）或任一 API Key |
| 第6章 LLM应用 | numpy, matplotlib | langchain, faiss-cpu³ | — |
| 第7章 多模态 | numpy, matplotlib | torch, Pillow, seaborn, scikit-learn⁴ | 下载 Qwen2.5-VL-7B 或 CLIP ViT-B/32⁵ |
| 第8章 工程实践 | — （仅标准库） | — | — |

**注释：**
1. `networkx` 仅 `ch04_transformer/graph_theory_demo.py` 使用，不在 `make run-exp-ch04` 中：`pip install networkx`
2. 可选；`llm_api_demo.py` 优先检测本地 llama.cpp / Ollama 兼容服务，无需 API Key 也可运行
3. `make run-exp-ch06` 只运行 `rag_demo.py`（纯 numpy 模拟），无需安装 langchain/faiss
4. 仅 `ch07_multimodal_llm/` 中的重度脚本（`make run-exp-ch07-heavy`），先执行 `pip install -e ".[hf]"`
5. `clip_alignment_demo.py` 还需 `pip install openai-clip`，首次运行自动下载 CLIP 权重（~350MB）；`case_studies.py` / `multimodal_applications.py` 需下载 Qwen2.5-VL-7B（~14GB）

模型/API 和数据集的统一登记见 `configs/models.yaml` 与 `configs/datasets.yaml`。章节文档、代码注释和 Notebook 涉及外部模型或数据集时，应以这两个文件为事实来源，避免在多个文档中维护互相冲突的版本、依赖和下载说明。

---

## 代码实验清单

### 第1章：DSP基础

> ⭐ **完全离线** — 仅需 numpy, matplotlib, scipy，无需任何 API 或模型

#### 主力脚本（`make run-exp-ch01`）

| 脚本 | 说明 | 输出图表 | 运行时间 |
|------|------|---------|---------|
| `fft_spectrum.py` | FFT 频谱分析，正弦信号分解 | `assets/ch01_fft_spectrum.png` | <1s |
| `positional_encoding.py` | Transformer 位置编码可视化 | `assets/ch01_positional_encoding.png` | <1s |

#### 扩展脚本（按需运行）

| 脚本 | 说明 | 额外依赖 | 输出图表 |
|------|------|---------|---------|
| `signal_three_views.py` | 信号的时域/频域/相位三视角 | scipy | `ch01_three_views.png` |
| `signal_dimensions.py` | 信号维度与表示 | — | `ch01_signal_dimensions.png` |
| `feature_extraction_comparison.py` | 特征提取方法对比 | — | `ch01_feature_extraction.png` |
| `fourier_2d.py` | 二维傅里叶变换 | — | `ch01_fourier_2d.png` |
| `time_freq_analysis.py` | 时频分析（STFT/小波） | scipy | `ch01_time_freq_*.png` |
| `random_signals.py` | 随机信号与功率谱 | scipy | `ch01_random_signals.png` |
| `signal_detection.py` | 信号检测与假设检验 | scipy | `ch01_signal_detection.png` |
| `parameter_estimation.py` | 参数估计（MLE/贝叶斯） | scipy | `ch01_parameter_estimation.png` |
| `music_algorithm.py` | MUSIC 超分辨率算法 | scipy | `ch01_matrix_decomposition*.png` |
| `stochastic_processes_demo.py` | 随机过程演示 | scipy | `ch01_stochastic_processes.png` |

#### 测试文件：`tests/test_ch01_dsp.py`（8 个测试）

| 测试类 | 覆盖内容 |
|-------|---------|
| `TestFFTSpectrum` | FFT 输出形状、对称性、频率检测精度、噪声鲁棒性 |
| `TestPositionalEncoding` | PE 形状、数值稳定性、周期性、无 NaN |

```bash
pytest tests/test_ch01_dsp.py -v
```

---

### 第2章：优化与机器学习

> ⭐ **完全离线** — 仅需 numpy, matplotlib，无需任何 API 或模型

#### 主力脚本（`make run-exp-ch02`）

| 脚本 | 说明 | 输出图表 | 运行时间 |
|------|------|---------|---------|
| `lms_vs_adam.py` | LMS 与 Adam 收敛速度对比 | `assets/ch02_lms_vs_adam.png` | <1s |
| `mmse_vs_nn.py` | MMSE 估计与神经网络对比 | `assets/ch02_mmse_vs_nn.png` | <1s |

#### 扩展脚本（按需运行）

| 脚本 | 说明 | 额外依赖 | 输出图表 |
|------|------|---------|---------|
| `linear_logistic_regression.py` | 线性/逻辑回归可视化 | — | `ch02_linear_logistic_regression.png` |
| `convex_analysis_demo.py` | 凸函数与梯度下降演示 | mpl_toolkits (已内置) | `ch02_convex_analysis.png` |
| `decision_tree_random_forest.py` | 决策树与随机森林 | — | `ch02_decision_tree_random_forest.png` |
| `svm_kernel.py` | SVM 核函数可视化 | — | `ch02_svm_kernel.png` |

#### 测试文件（2 个文件，12 个测试）

| 文件 | 测试类 | 覆盖内容 |
|------|-------|---------|
| `test_ch02_optimization.py` | `TestLMSOptimizer` | LMS 收敛、权重估计 |
| `test_ch02_optimization.py` | `TestAdamOptimizer` | Adam 收敛、Adam vs LMS 性能 |
| `test_ch02_ml_extended.py` | `TestLinearRegression` | 线性回归收敛、参数估计 |
| `test_ch02_ml_extended.py` | `TestLogisticRegression` | 逻辑回归收敛、分类准确率 |
| `test_ch02_ml_extended.py` | `TestSVM` | SVM 收敛性 |
| `test_ch02_ml_extended.py` | `TestDecisionTree` | 基尼系数、纯节点判定 |
| `test_ch02_ml_extended.py` | `TestRandomForest` | Bootstrap 采样 |

```bash
pytest tests/test_ch02_optimization.py tests/test_ch02_ml_extended.py -v
```

---

### 第3章：深度学习快速通道

> ⭐ **完全离线** — 仅需 numpy, matplotlib，无需任何 API 或模型（所有脚本纯 numpy 手写实现）

#### 主力脚本（`make run-exp-ch03`）

| 脚本 | 说明 | 输出图表 | 运行时间 |
|------|------|---------|---------|
| `polynomial_vs_mlp.py` | 多项式拟合 vs MLP 对比 | `assets/ch03_polynomial_vs_mlp.png` | <1s |
| `mnist_cnn.py` | CNN 特征提取结构演示（纯 numpy） | `assets/ch03_mnist_cnn.png` | <1s |
| `rnn_structure.py` | RNN 隐状态演化可视化 | `assets/ch03_rnn_structure.png` | <1s |

#### 扩展脚本（按需运行）

| 脚本 | 说明 | 额外依赖 | 输出图表 |
|------|------|---------|---------|
| `detection_segmentation_demo.py` | 目标检测与分割对比 | — | `ch03_yolo_vs_segmentation.png` 等 |
| `sequence_models_1d_signal.py` | 序列模型处理一维信号 | — | `ch03_sequence_models.png` |

#### 测试文件：`tests/test_ch03_deep_learning.py`（5 个测试）

| 测试类 | 覆盖内容 |
|-------|---------|
| `TestPolynomialVsMLP` | MLP 输出形状、非线性能力 |
| `TestRNNStructure` | 隐状态形状、隐状态演化、激活值有界性 |

```bash
pytest tests/test_ch03_deep_learning.py -v
```

---

### 第4章：Transformer详解

> ⭐ **完全离线** — 仅需 numpy, matplotlib，无需任何 API 或模型

#### 主力脚本（`make run-exp-ch04`）

| 脚本 | 说明 | 输出图表 | 运行时间 |
|------|------|---------|---------|
| `self_attention.py` | 自注意力机制计算与可视化 | `assets/ch04_self_attention.png` | <1s |

#### 扩展脚本（按需运行）

| 脚本 | 说明 | 额外依赖 | 输出图表 |
|------|------|---------|---------|
| `causal_mask_demo.py` | 因果掩码可视化 | — | `ch04_causal_mask.png` |
| `scaled_attention_demo.py` | 缩放点积注意力演示 | — | `ch04_scaled_attention.png` |
| `positional_encoding.py` | 位置编码热力图 | — | `ch04_positional_encoding.png` |
| `graph_theory_demo.py` | 图论基础与注意力图 | **networkx** | `ch04_graph_theory.png`, `ch04_attention_graph.png` |

#### 测试文件：`tests/test_ch04_transformer.py`（5 个测试）

| 测试类 | 覆盖内容 |
|-------|---------|
| `TestSelfAttention` | 注意力输出形状、权重归一化、多头注意力 |
| `TestTransformerBlock` | LayerNorm 稳定性、前馈网络 |

```bash
pytest tests/test_ch04_transformer.py -v
```

---

### 第5章：LLM基础

**依赖：** numpy, matplotlib, scipy（离线脚本）；openai 或 anthropic（API 脚本）
**安装（离线）：** `pip install -e .`
**安装（API）：** `pip install -e ".[llm]"`

#### 主力脚本——离线（`make run-exp-ch05`，8 个）

> ⭐ **这 8 个脚本完全离线，使用纯模拟数据和轻量计算，无需 PyTorch / 模型下载**

| 脚本 | 说明 | 输出图表 | 运行时间 |
|------|------|---------|---------|
| `bpe_tokenization.py` | BPE 分词算法演示 | `assets/ch05_bpe_tokenization.png` | <1s |
| `scaling_laws.py` | 缩放律拟合与可视化 | `assets/ch05_scaling_laws.png` | <1s |
| `autoregressive_generation.py` | 自回归生成过程演示 | `assets/ch05_autoregressive_generation.png` | <1s |
| `training_data_composition.py` | 训练数据构成分析 | `assets/ch05_training_data_composition.png` | <1s |
| `model_families_evolution.py` | 模型家族演化时间线 | `assets/ch05_model_families_evolution.png` | <1s |
| `lora_visualization.py` | LoRA 参数量可视化 | `assets/ch05_lora_visualization.png` | <1s |
| `rlhf_pipeline.py` | RLHF 流程图 | `assets/ch05_rlhf_pipeline.png` | <1s |
| `benchmark_comparison.py` | 基准测试对比图 | `assets/ch05_benchmark_comparison.png` | <1s |

#### 主力脚本——API（`make run-exp-ch05-api`）

| 脚本 | 说明 | 运行条件 |
|------|------|---------|
| `llm_api_demo.py` | 基本调用、Prompt 工程、ICL、多轮对话 | 本地 llama.cpp / Ollama **或** 任一 API Key |

```bash
# 离线（本地 llama.cpp / Ollama）
llama.cpp 或 Ollama 启动本地 OpenAI 兼容服务，例如 `http://localhost:8000/v1` 或 `http://localhost:11434/v1`
python code/ch05_llm_basics/llm_api_demo.py

# 国内 API
export DEEPSEEK_API_KEY="your-key"
python code/ch05_llm_basics/llm_api_demo.py
```

#### 测试文件：`tests/test_ch05_llm_basics.py`（5 个测试）

| 测试类 | 覆盖内容 |
|-------|---------|
| `TestScalingLaws` | 损失随模型/数据规模下降规律 |
| `TestInContextLearning` | Few-shot 学习、Prompt 工程效果 |
| `TestTokenization` | Token 计数 |

```bash
pytest tests/test_ch05_llm_basics.py -v
```

---

### 第6章：LLM应用

> ⭐ **完全离线** — 所有脚本使用模拟数据和轻量计算，无需 LangChain/FAISS/API/模型

#### 主力脚本（`make run-exp-ch06`）

| 脚本 | 说明 | 输出图表 | 运行时间 |
|------|------|---------|---------|
| `rag_demo.py` | RAG 检索增强生成（模拟 embedding） | `assets/ch06_rag_vector_search.png` | <1s |

#### 扩展脚本（`make run-exp-ch06-extra`）

| 脚本 | 说明 | 输出图表 | 运行时间 |
|------|------|---------|---------|
| `prompt_demo.py` | Zero-shot / Few-shot / CoT 效果对比 | `assets/ch06_prompt_techniques.png` | <1s |
| `agent_demo.py` | ReAct Agent 框架演示（规则替代 LLM） | `assets/ch06_agent_error_accumulation.png` | <1s |
| `finetuning_demo.py` | LoRA 不同 rank 下参数量对比 | `assets/ch06_lora_parameters.png` | <1s |
| `system_design_demo.py` | Prompt/RAG/微调/Agent 方案选型雷达图 | `assets/ch06_system_design.png` | <1s |

#### 测试文件：`tests/test_ch06_llm_applications.py`（4 个测试）

| 测试类 | 覆盖内容 |
|-------|---------|
| `TestRAGSystem` | 文档检索、上下文增强 |
| `TestFineTuning` | 微调收敛性 |
| `TestAgentFramework` | Agent 决策逻辑 |

```bash
pytest tests/test_ch06_llm_applications.py -v
```

---

### 第7章：多模态LLM

**依赖（主力脚本）：** numpy, matplotlib
**依赖（重度脚本）：** torch, Pillow, scikit-learn（multimodal_applications）；torch, Pillow, seaborn, openai-clip（clip_alignment_demo）；torch, Pillow, transformers（case_studies）
**安装（主力）：** `pip install -e .`
**安装（重度）：** `pip install -e ".[hf]"` 并下载模型

#### 主力脚本（`make run-exp-ch07`，完全离线，6 个）

> ⭐ **这些脚本完全离线，仅用 numpy/matplotlib，无需 PyTorch / 模型**

| 脚本 | 说明 | 输出图表 | 运行时间 |
|------|------|---------|---------|
| `vit_patches.py` | ViT 图像分块可视化 | `assets/ch07_vit_patches.png` | <1s |
| `clip_similarity.py` | CLIP 余弦相似度演示（模拟向量） | `assets/ch07_clip_similarity.png` | <1s |
| `high_resolution_processing.py` | 高分辨率动态分块演示 | `assets/ch07_high_resolution_processing.png` | <1s |
| `qwen_vl_analysis.py` | Qwen2.5-VL 架构性能分析（静态数据） | `assets/ch07_qwen_vl_analysis.png` | <1s |
| `architecture_diagrams.py` | ViT vs CNN 结构对比、温度参数效果 | `assets/ch07_vit_cnn_comparison.png`, `assets/ch07_temperature_effect.png` | <1s |
| `explainer_diagrams.py` | LLaVA vs Qwen2.5-VL 架构、融合策略、动态分辨率 | `assets/ch07_architecture_comparison.png` 等 4 张 | <1s |

#### 重度脚本（`make run-exp-ch07-heavy`，需额外依赖 + 下载模型）

> ⚠️ **这些脚本不属于默认实验。CLIP 脚本会下载约 350MB 权重；Qwen2.5-VL 相关脚本需要下载约 14GB 权重，建议使用 GPU。**

| 脚本 | 说明 | 额外依赖 | 外部资源 |
|------|------|---------|---------|
| `clip_alignment_demo.py` | CLIP 图文对齐热力图 | torch, Pillow, seaborn, openai-clip | 自动下载 CLIP ViT-B/32（~350MB） |
| `multimodal_applications.py` | 图像描述 / VQA / 图像检索 | torch, Pillow, scikit-learn, transformers | 下载 Qwen2.5-VL-7B（~14GB） |
| `case_studies.py` | 文档理解 / 图表分析 / 多语言应用 | torch, Pillow, transformers | 下载 Qwen2.5-VL-7B（~14GB） |

```bash
# 重度脚本安装
pip install -e ".[hf]"
pip install openai-clip  # clip_alignment_demo 额外需要

# 下载模型（二选一）
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./models/Qwen2.5-VL-7B-Instruct
# 或 设置 HF_ENDPOINT=https://hf-mirror.com 后再下载
```

#### 测试文件：`tests/test_ch07_multimodal.py`（6 个测试）

| 测试类 | 覆盖内容 |
|-------|---------|
| `TestVisionTransformer` | 图像分块逻辑、patch embedding |
| `TestCLIPModel` | 图文对齐、对比学习损失 |
| `TestHighResolutionProcessing` | 动态分块方法、分辨率自适应 |

```bash
pytest tests/test_ch07_multimodal.py -v
```

---

### 第8章：LLM工程实践

> ⭐ **完全离线** — 仅需标准库（time, json, dataclasses, enum），无需任何依赖安装

#### 主力脚本（`make run-exp-ch08`）

| 脚本 | 说明 | 输出 | 运行时间 |
|------|------|------|---------|
| `llm_engineering_demo.py` | 模型选型、成本优化、错误处理、监控演示 | 终端输出（无图表） | <1s |

#### 测试文件：`tests/test_ch08_engineering.py`（8 个测试）

| 测试类 | 覆盖内容 |
|-------|---------|
| `TestModelDeployment` | 模型量化逻辑、缓存机制 |
| `TestCostOptimization` | 批处理优化、模型选择策略 |
| `TestSafetyAlignment` | 内容过滤、输出验证 |
| `TestMonitoring` | 性能指标采集、错误追踪 |

```bash
pytest tests/test_ch08_engineering.py -v
```

---

## 常见问题

> 环境安装、虚拟环境、PyTorch/CUDA 问题见 [附录B 常见问题](B_environment_setup.md#常见问题)。

### Q1: 如何配置 LLM API 密钥？

**方法1：** 设置环境变量

```bash
# Linux/macOS
export DEEPSEEK_API_KEY="your-key"
export DASHSCOPE_API_KEY="your-key"
export ZHIPUAI_API_KEY="your-key"
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# 本地 llama.cpp / Ollama（无需 API Key）
export LOCAL_LLM_BASE_URL="http://localhost:11434/v1"
export LOCAL_LLM_MODEL="qwen2.5:7b"
export OLLAMA_BASE_URL="http://localhost:11434/v1"
export OLLAMA_MODEL="qwen2.5:7b"

# Windows (PowerShell)
$env:DEEPSEEK_API_KEY="your-key"
$env:DASHSCOPE_API_KEY="your-key"
$env:ZHIPUAI_API_KEY="your-key"
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

**方法2：** 在代码中设置

```python
import os
os.environ["DEEPSEEK_API_KEY"] = "your-key"
os.environ["DASHSCOPE_API_KEY"] = "your-key"
os.environ["ZHIPUAI_API_KEY"] = "your-key"
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
```

**方法3：** 使用 .env 文件

创建 `.env` 文件：
```
DEEPSEEK_API_KEY=your-key
DASHSCOPE_API_KEY=your-key
ZHIPUAI_API_KEY=your-key
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
LOCAL_LLM_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5:7b
```

然后在代码中加载：
```python
from dotenv import load_dotenv
load_dotenv()
```

### Q2: LLM API 调用超时或失败？

**原因：** 网络连接问题、API 配额限制或密钥无效

**解决方案：**
```bash
# 检查网络连接
ping api.openai.com

# 检查 API 密钥是否正确
python -c "import os; print(os.environ.get('OPENAI_API_KEY'))"

# 查看 API 使用情况和配额
# 访问 https://platform.openai.com/account/usage/overview
```

### Q3: 如何降低 LLM API 的成本？

- 使用更便宜的模型（如 GPT-3.5 而不是 GPT-4）
- 减少 token 数量（缩短 prompt 和 response）
- 使用缓存和批处理
- 参考 [第6章 6.5 LLM 系统设计](../06_llm_applications/05_system_design.md) 的成本估算

### Q4: 如何在离线环境中运行代码？

某些实验需要网络连接（LLM API 调用），但可以：
- 运行第1-4章的所有实验（不需要网络）
- 运行第5章、第6章、第8章的默认离线实验
- 运行第7章默认离线实验（不下载模型）
- 使用本地模型替代 API（如 Ollama、LLaMA）
- 预先缓存 API 响应

---

## 获取帮助

- 环境配置问题：[附录B](B_environment_setup.md)
- 提交 Bug：[GitHub Issue](https://github.com/datawhalechina/signal-to-intelligence/issues)
