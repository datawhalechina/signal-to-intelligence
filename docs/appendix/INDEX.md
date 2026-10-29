# 附录导航指南

欢迎来到《Signals to Intelligence》附录。这是你快速找到所需资源的入口。

---

## 🎯 快速决策树（选择你的路径）

### 问题 1：你有本地 GPU 吗？

- **没有 GPU（或只有 CPU）** → 继续问题 2
- **有 NVIDIA GPU** → 见 [本地推理指南 — GPU 加速](B_local_inference.md#gpu-加速-可选)

### 问题 2：你想怎么运行？

- **想快速开始，运行离线实验** → [快速开始](B_environment_setup.md)
- **想用 API（国内/海外）** → [API 配置指南](B_api_configuration.md)
- **想在云端运行（Colab/CNB）** → [云端部署指南](B_cloud_deployment.md)
- **想本地运行模型（Ollama/HF）** → [本地推理指南](B_local_inference.md)

### 问题 3：你想了解什么？

- **哪些脚本能运行？需要什么？** → [代码运行指南](C_code_guide.md)
- **如何维护代码、测试、Notebook、图片、模型、数据集和文档站？** → [项目维护说明](E_project_maintenance.md)
- **如何下载模型和数据集？** → [模型获取指南](B_model_sources.md)
- **如何容器化和云部署？** → [Docker/CNB 部署指南](B_docker_cnb.md)
- **数学基础和快速参考** → [数学速查表](D_math_quick_reference.md)
- **数学深度内容** → [数学参考文档](A_math_reference.md)

---

## 📚 完整文档列表

### 核心配置（必读）

| 文档 | 用途 |
|------|------|
| **[B_environment_setup.md](B_environment_setup.md)** | 最简环境配置与首次运行 |
| **[C_code_guide.md](C_code_guide.md)** | 了解每章有哪些脚本，能做什么 |

### 场景化指南（按需选择）

| 文档 | 场景 |
|------|------|
| **[B_api_configuration.md](B_api_configuration.md)** | 想用 LLM API（DeepSeek/OpenAI） |
| **[B_local_inference.md](B_local_inference.md)** | 有 GPU，或想用 Ollama 本地推理 |
| **[B_model_sources.md](B_model_sources.md)** | 想下载模型或数据集 |
| **[B_docker_cnb.md](B_docker_cnb.md)** | 想容器化应用或云端部署 |
| **[B_cloud_deployment.md](B_cloud_deployment.md)** | 想用 Colab/CNB/Docker 云端运行 |

### 参考资料（深度学习）

| 文档 | 用途 |
|------|------|
| **[A_math_reference.md](A_math_reference.md)** | 数学基础深度讲解（复数、线性代数、信息论等） |
| **[D_math_quick_reference.md](D_math_quick_reference.md)** | 数学概念速查表 |
| **[E_project_maintenance.md](E_project_maintenance.md)** | 项目维护约定：单一源头、图片同步、测试、Notebook、模型/数据集 registry、文档站发布、Makefile |

---

## 🚀 常见场景快速导航

### 场景 1：我是新手，想快速开始

1. 阅读 [快速开始](B_environment_setup.md)
2. 查看 [代码运行指南](C_code_guide.md) 的"快速判断表"
3. 运行第一个实验

---

### 场景 2：我有 GPU，想本地训练

1. 阅读 [快速开始](B_environment_setup.md)
2. 按照 [本地推理指南 — GPU 加速](B_local_inference.md#gpu-加速-可选) 安装 PyTorch
3. 查看 [代码运行指南](C_code_guide.md) 了解各章 GPU 需求
4. 运行实验

---

### 场景 3：我想用 API（国内推荐 DeepSeek）

1. 阅读 [快速开始](B_environment_setup.md)
2. 按照 [API 配置指南 — DeepSeek](B_api_configuration.md#deepseek) 配置密钥
3. 查看 [代码运行指南](C_code_guide.md) 的快速判断表
4. 运行第 5 章 `llm_api_demo.py`

---

### 场景 4：我没有 GPU，想在云端运行

1. 选择云端方案：[Google Colab](B_cloud_deployment.md#google-colab-免费-推荐)
2. 按照指南复制代码到 Notebook
3. 运行实验

---

### 场景 5：我想用本地 Ollama（离线推理）

1. 阅读 [快速开始](B_environment_setup.md)
2. 按照 [本地推理指南 — Ollama](B_local_inference.md#ollama-最简单的本地推理) 安装
3. 运行第 5 章 `llm_api_demo.py`（自动检测本地 Ollama）

---

## 📖 按学习阶段推荐

### 第 0 阶段：准备环境

**必读**
- [B_environment_setup.md](B_environment_setup.md)
- [C_code_guide.md](C_code_guide.md) 的"快速判断表"

**可选**
- 根据你的资源选择对应配置指南

---

### 第 1 阶段：学习第 1-6 章（离线）

**必读**
- 无需额外配置，已完成环境配置即可运行

**可选**
- [本地推理指南](B_local_inference.md) — 如果要 GPU 加速

---

### 第 2 阶段：学习第 5 章（API 部分）

**需要配置**
- [API 配置指南](B_api_configuration.md) — 选择一个 API 配置密钥
- 或 [本地推理指南 — Ollama](B_local_inference.md#ollama-最简单的本地推理) — 离线方式

---

### 第 3 阶段：学习第 7 章（重度脚本）

**需要配置**
- [本地推理指南](B_local_inference.md) — GPU + 下载模型
- 或 [云端部署指南](B_cloud_deployment.md) — 使用云端 GPU

---

## ❓ 常见问题快速查询

| 问题 | 文档 |
|------|------|
| 如何快速开始？ | [快速开始](B_environment_setup.md) |
| 哪些脚本需要 API？ | [代码运行指南 — 快速判断表](C_code_guide.md#🎯-快速判断-我能运行什么) |
| 如何配置 DeepSeek API？ | [API 配置指南](B_api_configuration.md#获取-api-密钥) |
| 如何安装 GPU 版本 PyTorch？ | [本地推理指南 — GPU 加速](B_local_inference.md#gpu-加速-可选) |
| 如何使用 Ollama？ | [本地推理指南 — Ollama](B_local_inference.md#ollama-最简单的本地推理) |
| 如何下载模型？ | [模型获取指南 — 快速开始](B_model_sources.md#快速开始下载模型) |
| 如何使用 Hugging Face？ | [模型获取指南 — Hugging Face](B_model_sources.md#hugging-face) |
| 如何获取数据集？ | [模型获取指南 — 数据集获取](B_model_sources.md#数据集获取) |
| 如何容器化我的应用？ | [Docker/CNB 部署指南 — 快速开始](B_docker_cnb.md#快速开始docker-化-llm-应用) |
| Docker 还是 CNB？ | [Docker/CNB 部署指南 — 对比](B_docker_cnb.md#docker-vs-cnb-对比) |
| 如何部署到云平台？ | [Docker/CNB 部署指南 — 云平台部署](B_docker_cnb.md#云平台部署) |
| 如何同步图片并发布文档站？ | [项目维护说明 — 图片资产同步](E_project_maintenance.md#图片资产同步) |
| 测试和 Notebook 应该怎么复用代码？ | [项目维护说明 — 单一源头原则](E_project_maintenance.md#单一源头原则) |
| 模型和数据集应该如何登记？ | [项目维护说明 — 模型与数据集维护规则](E_project_maintenance.md#模型与数据集维护规则) |
| Makefile 里有哪些常用命令？ | [项目维护说明 — Makefile 使用](E_project_maintenance.md#makefile-使用) |
| 在 Colab 中运行？ | [云端部署指南 — Google Colab](B_cloud_deployment.md#google-colab-免费-推荐) |
| 显存不足怎么办？ | [本地推理指南 — 显存需求速查](B_local_inference.md#显存需求速查) |
| 模型下载或加载失败怎么办？ | [模型获取指南 — 常见问题](B_model_sources.md#常见问题) |
| Docker 镜像太大？ | [Docker/CNB 部署指南 — 常见问题](B_docker_cnb.md#常见问题与解决) |

---

## 🔗 其他资源

- **主项目 README** → [README.md](https://github.com/datawhalechina/signal-to-intelligence/blob/main/README.md)
- **完整教程** → [docs/](../) 目录
- **代码** → [code/](https://github.com/datawhalechina/signal-to-intelligence/tree/main/code) 目录
- **测试** → [tests/](https://github.com/datawhalechina/signal-to-intelligence/tree/main/tests) 目录

---

## 💡 建议阅读顺序

**第一次来访问的新手：**

1. 本文（INDEX.md）— 了解大局
2. [B_environment_setup.md](B_environment_setup.md) — 快速开始
3. [C_code_guide.md](C_code_guide.md) — 了解能做什么
4. 选择对应的场景指南
5. 开始学习第一章 ✨

---

## 📞 获取帮助

- 遇到问题？先查看对应文档的"常见问题"部分
- 提交 Bug → [GitHub Issue](https://github.com/datawhalechina/signal-to-intelligence/issues)
- 改进建议 → [GitHub Discussions](https://github.com/datawhalechina/signal-to-intelligence/discussions)

---

**开始吧！** [快速开始 →](B_environment_setup.md)
