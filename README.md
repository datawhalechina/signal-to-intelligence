# 从信号到智能

`signal-to-intelligence` | 从数字信号处理到大语言模型的思想演化与代码实践

**版本：** v3.2

> [!WARNING]
> 🧪 本教程已完成主体内容和主要工程重构，当前在持续优化细节、导航和扩展阅读，欢迎通过 Issue 反馈问题或建议。

## 项目概览

**定位：** 面向工程师的思想对照式教程，把 DSP、统计、优化、深度学习、Transformer、LLM 和多模态串成一条可迁移的学习路径。

> 定位说明：本教程不是严格数学教材，而是面向工程师的概念迁移与实践教程。正文优先建立直觉、工程边界和可运行实验；需要严格证明的内容放在扩展章节和参考资料中。

**特色：** 思想对照式主线；`docs/` 讲概念，`code/` 跑实验，`tests/` 验证行为，`notebooks/` 做交互补充；工程化维护减少“一处修改，多处维护”。

**受众：** 具有理工科背景，学过 DSP / 控制 / 通信 / 机器学习基础，想快速理解 LLM 本质的工程师。

[📖 **在线文档：** https://datawhalechina.github.io/signal-to-intelligence/](https://datawhalechina.github.io/signal-to-intelligence/)
[🔗 **GitHub 仓库：** https://github.com/datawhalechina/signal-to-intelligence](https://github.com/datawhalechina/signal-to-intelligence)

## Quick Start

以下流程使用 CPU 和默认离线依赖，可验证代码与测试环境，无需 API Key、GPU 或模型下载。

```bash
git clone https://github.com/datawhalechina/signal-to-intelligence.git
cd signal-to-intelligence

python3 -m venv .venv
source .venv/bin/activate          # Windows 请在 WSL2 或使用对应的激活命令
python -m pip install --upgrade pip
pip install -e .

# 验证安装：运行全部测试
make test

# 运行一个章节实验：自注意力与图表生成
make run-exp-ch04
```

接下来可按目标选择：

| 目标 | 入口 |
|------|------|
| 系统学习理论主线 | [在线文档](https://datawhalechina.github.io/signal-to-intelligence/) 或 [学习路径](docs/00_introduction/02_learning_paths.md) |
| 运行其他章节的轻量实验 | `make help` 或 [代码运行指南](docs/appendix/C_code_guide.md) |
| 交互式修改参数 | [Notebook 指南](notebooks/README.md) |
| 完成最低实践任务或后续章节项目 | [项目实践说明](projects/README.md) |
| 本地预览文档站 | `npm install && npm run docs:dev` |

## 最近更新

- 首页 README 已重排，使用方法按在线阅读、Notebook、本地代码、Docker 和 CNB 分流。
- 章节导航补充了每章主线节数统计，方便快速判断内容体量。
- 第 0 章、附录 B/C/E 和 Notebook 说明已进一步收口，入口和手册职责更清晰。

---

## 使用方法

**怎么选：**
1. **没有 NVIDIA GPU**，优先选 **A 在线阅读 VitePress** 或 **B Google Colab Notebook**；只运行轻量实验也可以选择 **C.1 本地运行代码**。
2. **有 NVIDIA GPU** 且想本地复现，再选 **C.1 本地运行代码** 或 **C.2 Docker 本地环境**。
3. **Linux / WSL2 / macOS** 可以先按同一套本地流程理解；如果是 **原生 Windows**，优先通过 **WSL2** 进入同一流程。

| 序号 | 方式 | 适合场景 | 主要入口 | 说明 |
|-------|----------|----------|------|------|
| A | 在线阅读 VitePress | 系统学习正文 | [在线网页](https://datawhalechina.github.io/signal-to-intelligence/) | 阅读指南见 [如何使用本教程](docs/00_introduction/05_how_to_use_this_tutorial.md) |
| B | Google Colab Notebook | 云端交互实验 | [Notebook 使用方式](docs/00_introduction/05_how_to_use_this_tutorial.md) | 交互逻辑统一复用 `code/` |
| C.1 | 本地运行代码 | 环境安装、跑实验、改脚本 | [源码仓库 code/](https://github.com/datawhalechina/signal-to-intelligence/tree/main/code) | 运行说明见 [附录 C：代码运行指南](docs/appendix/C_code_guide.md)，代码以 `code/` 为准 |
| C.2 | Docker 本地环境 | 本地算力 + 统一容器环境 | [附录 B：环境配置](docs/appendix/B_environment_setup.md) | 适合在本机复现一致运行环境 |
| D.1 | Docker 云端环境 | 云端算力 + 统一容器环境 | [附录 B：环境配置](docs/appendix/B_environment_setup.md) | 适合在云端机器上运行容器化实验 |

注：Notebook 当前提供 **Google Colab** 路径；Docker 按 **本地 / 云端** 运行位置区分。ModelScope 与 CNB 尚无可验证的一键入口，因此不列入当前可用路径。
如果不确定从哪里开始，先阅读 [第0章：学习路径](docs/00_introduction/02_learning_paths.md)。



---

## 章节导航

| 章节 | 主线节数 | 主线内容 | 扩展阅读 | 状态 |
|------|----------|----------|----------|------|
| [第0章：导论](docs/00_introduction/README.md) | 5 | 学习路径与全书概览 | 见章节页 | 已完成 |
| [第1章：DSP基础](docs/01_dsp/README.md) | 8 | 信号的三种视角 | [3 个扩展](docs/01_dsp/README.md#扩展内容) | 已完成 |
| [第2章：优化与机器学习](docs/02_optimization/README.md) | 7 | 从优化算法到传统ML | [3 个扩展](docs/02_optimization/README.md#扩展内容) | 已完成 |
| [第3章：深度学习快速通道](docs/03_deep_learning_fast/README.md) | 6 | CNN、RNN、Transformer规模化建模 | [1 个扩展](docs/03_deep_learning_fast/README.md#扩展内容) | 已完成 |
| [第4章：Transformer详解](docs/04_transformer/README.md) | 4 | 自注意力机制 | [3 个扩展](docs/04_transformer/README.md#扩展内容) | 已完成 |
| [第5章：LLM原理](docs/05_llm_basics/README.md) | 6 | 预训练、缩放律、对齐 | [6 个扩展](docs/05_llm_basics/README.md#扩展内容) | 已完成 |
| [第6章：LLM应用](docs/06_llm_applications/README.md) | 5 | Prompt → 微调 → RAG → Agent | [4 个扩展](docs/06_llm_applications/README.md#扩展内容) | 已完成 |
| [第7章：多模态LLM](docs/07_multimodal_llm/README.md) | 6 | 从视觉主干到信号—语言对齐 | [2 个扩展](docs/07_multimodal_llm/README.md#扩展内容) | 已完成 |
| [第8章：LLM工程实践与部署](docs/08_llm_engineering/README.md) | 6 | 云端生产闭环与端侧联合验收 | [2 个扩展](docs/08_llm_engineering/README.md#扩展阅读) | 已完成 |

代码细目和运行方式见 [附录 C：代码运行指南](docs/appendix/C_code_guide.md)，数学前置导航见 [附录 D：数学前置导航](docs/appendix/D_math_quick_reference.md)。章节项目和综合项目见 [`projects/`](projects/README.md)。

---

## 贡献者名单

| 姓名 | 职责 | 简介 |
| :---- | :---- | :---- |
| `lynn_jingjing` | 项目负责人 | 仓库维护者，负责教程主线、代码重构、文档整理与发布维护 |

项目维护约定、工程文件说明、图片同步、测试和 Notebook 复用规则见 [附录E：项目维护说明](docs/appendix/E_project_maintenance.md)。

---

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="CC BY-NC-SA 4.0" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />
本项目采用 **知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议（CC BY-NC-SA 4.0）**。

- 使用本项目内容时请保留署名并注明来源。
- 不得用于商业用途。
- 如对内容进行再创作或再发布，请保持相同许可。

具体条款可参考 [Creative Commons 官方说明](http://creativecommons.org/licenses/by-nc-sa/4.0/)。如需商用授权或其他许可安排，请联系仓库维护者。
