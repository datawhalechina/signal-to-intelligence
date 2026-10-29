# 本地推理指南

本指南说明如何在本地运行 GPU 加速和大模型推理，支持第 5-7 章的重度脚本。

> **前置条件：** 已完成 [快速开始](B_environment_setup.md)  
> **需要的命令：** `pip install -e ".[ml]"` 和/或 `pip install -e ".[hf]"`

---

## GPU 加速（可选）

如果你有 NVIDIA GPU，安装 PyTorch GPU 版本可加速深度学习实验。

### 检查 GPU

```bash
nvidia-smi  # 查看 GPU 型号和 CUDA 版本
```

### 安装 GPU 版本 PyTorch

**推荐：自动检测**
```bash
pip install light-the-torch
ltt install torch torchvision
```

**手动指定（如果自动检测失败）**
```bash
# RTX 30/40 系（CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# RTX 50 系 Blackwell（CUDA 12.8，需 PyTorch 2.7+）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 验证 GPU 安装

```bash
python3 -c "import torch; print(torch.cuda.is_available())"  # 应输出 True
```

---

## 显存需求速查

### 推理显存需求

| 模型规模 | FP16 | INT8 | INT4 (GPTQ) |
|---------|------|------|------------|
| 0.5B | ~1GB | ~0.5GB | ~0.3GB |
| 1.5B | ~3GB | ~1.5GB | ~1GB |
| 7B | ~14GB | ~7GB | ~4GB |
| 13B | ~26GB | ~13GB | ~7GB |
| 70B | ~140GB | ~70GB | ~35GB |

### 消费级 GPU 对应方案

| GPU | 显存 | 推荐方案 |
|-----|------|--------|
| RTX 3060 / 3070 | 8~12GB | 7B INT4 推理 |
| RTX 3090 | 24GB | 7B FP16 推理 |
| RTX 4060 / 4070 | 8~12GB | 7B INT4 推理 |
| RTX 4080 / 4090 | 16~24GB | 7B FP16 推理 |
| RTX 5060 / 5070 | 8~12GB | 7B INT4 推理 |
| RTX 5080 / 5090 | 16~32GB | 7B FP16 推理 |
| A100 40G | 40GB | 13B FP16 推理 |

**没有 GPU？** 见下方 Ollama 部分（CPU 也能运行小模型）。

---

## Ollama（最简单的本地推理）

Ollama 是最简单的本地推理方式，无需配置，一键启动。

### 安装 Ollama

访问 [ollama.com](https://ollama.com) 下载安装。

### 运行模型

```bash
# 启动 Qwen2.5-7B（第 5 章推荐）
ollama run qwen2.5:7b

# 或启动 Qwen3-8B
ollama run qwen3:8b

# 或启动轻量级 0.5B 模型（CPU 可运行）
ollama run qwen2.5:0.5b
```

### 配置环境变量（可选）

```bash
# 切换默认模型
export OLLAMA_MODEL=qwen3:8b

# 自定义 Ollama 服务地址
export OLLAMA_BASE_URL=http://localhost:11434/v1
```

### 运行 ch05 实验

```bash
# 启动 Ollama（在另一个终端）
ollama run qwen2.5:7b

# 运行实验（自动检测本地 Ollama）
python code/ch05_llm_basics/llm_api_demo.py
```

**输出示例：**
```
使用本地 Ollama（qwen2.5:7b）
```

---

## Hugging Face 本地推理

用于更高级的控制和微调。

### 安装依赖

```bash
pip install -e ".[hf]"
```

### 下载模型（推荐 Qwen2.5-7B）

**方式 1：Hugging Face Hub**
```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir ./models/Qwen2.5-7B-Instruct
```

**方式 2：国内加速（设置 HF 镜像）**
```bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir ./models/Qwen2.5-7B-Instruct
```

**方式 3：ModelScope（国内备选）**
```bash
pip install modelscope

modelscope download --model Qwen/Qwen2.5-7B-Instruct \
  --local_dir ./models/Qwen2.5-7B-Instruct
```

### 推荐入门模型

| 模型 | 参数量 | 磁盘 | 特点 |
|------|--------|------|------|
| Qwen2.5-0.5B | 0.5B | ~1GB | CPU 可运行 |
| Qwen2.5-1.5B | 1.5B | ~3GB | 轻量级 |
| Qwen2.5-7B | 7B | ~14GB | 推荐，均衡 |
| Qwen2.5-7B-GPTQ-Int4 | 7B | ~4GB | 量化版 |

### 运行 ch07 重度实验

```bash
# 下载模型
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir ./models/Qwen2.5-VL-7B-Instruct

# 运行重度脚本
make run-exp-ch07-heavy
```

---

## 常见问题

### Ollama 启动失败？

确认：
- Ollama 已正确安装
- 端口 11434 未被占用
- 有足够的磁盘空间

### 模型下载很慢？

**国内用户：**
```bash
# 使用 HF 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或使用 ModelScope
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./models
```

### 显存不足？

使用量化版本（INT4）：
```bash
huggingface-cli download Qwen/Qwen2.5-7B-GPTQ-Int4 \
  --local-dir ./models/Qwen2.5-7B-GPTQ-Int4
```

### 没有 GPU 能运行吗？

可以，但速度会很慢：
- Ollama + qwen2.5:0.5b（CPU 可接受）
- llama.cpp + GGUF 格式（专为 CPU 优化）
- 或使用 [API](B_api_configuration.md)

---

## 获取帮助

- 📖 [快速开始](B_environment_setup.md)
- 📡 [API 配置指南](B_api_configuration.md)
- ☁️ [云端部署指南](B_cloud_deployment.md)
- 🧪 [代码运行指南](C_code_guide.md)
- 📚 [完整导航](INDEX.md)
