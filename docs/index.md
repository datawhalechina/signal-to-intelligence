# Signals to Intelligence

从数字信号处理到大语言模型的思想演化与代码实践

**版本：** v3.1 | **最后更新：** 2026-06-01

---

## 项目简介

面向工程师的**思想对照式**快速入门教程。帮你把已有的信号处理 / 统计知识迁移到现代 AI（深度学习、Transformer、LLM、多模态）的理解中。

**目标读者：** 具有理工科背景、学过 DSP / 控制 / 通信 / 机器学习基础，想快速理解 LLM 本质的工程师。

---

## 快速开始

```bash
git clone <repo-url> && cd signal-to-intelligence
python3 -m venv venv && source venv/bin/activate
pip install -e .
python code/ch01_dsp/fft_spectrum.py   # 验证安装
```

> 第 1-5 章代码实验全部使用模拟数据，**不需要 GPU，不需要下载模型**。
> 完整环境配置（Docker / 云端 / API Key）见 → [附录 B：环境配置](/appendix/B_environment_setup)

---

## 章节目录

| 章节 | 主题 | 核心内容 |
|------|------|---------|
| [第0章：导论](/00_introduction/) | 学习路径与全书概览 | 三条学习路径，章节速览 |
| [第1章：DSP基础](/01_dsp/) | 信号的三种视角 | 傅里叶变换、卷积、滤波器 |
| [第2章：优化与机器学习](/02_optimization/) | 从优化算法到传统ML | 梯度下降、Adam、SVM、决策树 |
| [第3章：深度学习快速通道](/03_deep_learning_fast/) | CNN、RNN、为什么Transformer更好 | 卷积网络、序列模型、对比分析 |
| [第4章：Transformer详解](/04_transformer/) | 自注意力机制 | QKV注意力、多头、位置编码 |
| [第5章：LLM原理](/05_llm_basics/) | 预训练、缩放律、对齐 | BPE、Scaling Laws、RLHF、DPO |
| [第6章：LLM应用](/06_llm_applications/) | Prompt → 微调 → RAG → Agent | 系统设计、工程实践 |
| [第7章：多模态LLM](/07_multimodal_llm/) | 视觉与语言的融合 | ViT、CLIP、Qwen2.5-VL |

---

## 附录

| 文件 | 内容 |
|------|------|
| [附录A：数学备忘](/appendix/A_math_reference) | 线性代数、概率、复数、信号处理、优化公式速查 |
| [附录B：环境配置](/appendix/B_environment_setup) | 本地 / Docker / 云端 / CNB，API Key 配置，验证脚本 |
| [附录C：代码运行指南](/appendix/C_code_guide) | 代码结构、运行方式、所有实验汇总表 |
| [附录D：数学基础速查表](/appendix/D_math_quick_reference) | 各章数学前置速查 + extensions 深度阅读导航 |

---

## 工程文件说明

| 文件 / 目录 | 作用 |
|------------|------|
| `requirements.txt` | 便利入口，等价于 `pip install -e .`，实际依赖由 `pyproject.toml` 管理 |
| `pyproject.toml` | 依赖单点维护：核心依赖 + 可选依赖组（`pip install -e ".[llm]"` 等） |
| `Makefile` | 常用命令快捷方式，`make help` 查看所有命令 |
| `.env.example` | API Key 模板，复制为 `.env` 后填入密钥 |
| `deploy/` | Docker Compose 配置（CPU 版 + GPU 版），见 [deploy/README.md](https://github.com/datawhalechina/signal-to-intelligence/blob/main/deploy/README.md) |
| `code/` | 所有可运行代码实验，按章节分目录 |
| `docs/` | 教程文档，按章节分目录，每章含 `extensions/` 扩展内容 |
| `assets/` | 代码实验生成的图表（由脚本自动生成，不手动编辑） |

---

## 学习路径

- **快速通道**（8-12h）：第4章 → 第5章 → 第6章 → 第7章
- **完整路径**（20-30h）：第1章 → 第2章 → ... → 第7章
- **深度探索**（40-50h）：所有章节 + 所有 `extensions/`

详见 [第0章：学习路径](/00_introduction/02_learning_paths)
