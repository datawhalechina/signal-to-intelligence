# 附录 E：项目维护说明

本文记录仓库维护约定，避免图片资产、文档站发布、测试、Notebook、模型、数据集和常用命令分散在多个地方。

---

## 教程设计准则

本教程的长期定位是“面向工程师的中高阶 LLM 系统理解教程”。它不应被维护成零基础科普、论文综述、数学证明教材或某个框架的 API 手册，而应帮助读者建立从信号、优化、深度学习、Transformer 到 LLM 训练、应用、多模态和工程部署的完整判断链路。

### 深度边界

- 数学深度：保留支撑理解的关键公式、前提条件和直觉，不追求完整定理证明。
- 工程深度：必须强于普通概念教程，讲清真实系统里的成本、延迟、显存、质量、可靠性和上线风险。
- 模型原理深度：解释机制、适用边界和失败模式，不追逐所有最新论文细节。
- 实践深度：代码和 Notebook 服务于理解与验证，不把教程写成工具调用清单。

### 内容取舍

每个核心章节都应优先回答四个问题：

1. 这个概念解决什么问题？
2. 它为什么有效？
3. 它在真实系统里有什么代价？
4. 它失败时通常是什么原因？

如果一节内容只能回答“是什么”，但不能回答“为什么”和“如何取舍”，就需要继续收紧或补充。正文应减少百科式罗列，多写机制解释、边界条件、工程权衡和失败模式。

### 章节角色

- 第 1 章提供信号、频域、检测、估计和矩阵分解等思想工具。
- 第 2 章解释学习问题为什么会变成优化问题。
- 第 3 章说明深度学习如何从特征工程走向端到端表示学习。
- 第 4 章解释 Transformer 为什么适合序列建模和大模型。
- 第 5 章解释 LLM 如何通过数据、规模、预训练、微调和对齐形成能力。
- 第 6 章解释 Prompt、微调、RAG 和 Agent 各自适合解决什么问题。
- 第 7 章解释多模态系统中的视觉编码、跨模态对齐、融合和任务约束。
- 第 8 章解释模型如何上线，并在质量、成本、延迟和可靠性之间取舍。

### 主线与扩展

- 主线章节讲核心机制、工程意义、适用边界和常见失败模式。
- 扩展章节承接更深的推导、论文路线、复杂实现和补充案例。
- 模型名、工具名和框架名优先作为例子，不作为章节结构中心。
- 章节间桥接要短而准确，避免用大段跨领域类比压过本章主线。

### 跨学科内容边界

控制论、运筹学、信息论、图论等跨学科内容只作为理解 LLM 系统的辅助视角，不作为本教程的独立主线。维护时应遵循：

- 第 1 章可以用状态空间、稳定性和反馈控制解释动态系统直觉，但不展开完整控制论课程。
- 第 2 章可以用线性规划、整数规划、动态规划、排队论和约束优化解释系统决策，但不讲求解算法细节。
- 第 6 章可以借用反馈闭环解释 Agent 的 ReAct 循环，但必须强调延迟、成本、错误累积和权限风险。
- 第 8 章可以借用运筹学视角解释批处理、路由、缓存、容量规划和 SLO 成本优化，但重点应落在 LLM Serving 的工程取舍。
- 这类内容应优先放在 extension 或短桥接段落中；除非重新定义教程定位，否则不要新增独立主章。
- 公式只保留最基础、最能支撑理解的形式；避免引入读者难以审核的复杂证明和专业求解流程。

---

## 单一源头原则

同一类信息只能有一个主要维护位置。其他位置可以引用、同步或展示，但不应手动维护第二份。

| 内容 | 唯一源头 | 展示/生成位置 |
|------|----------|---------------|
| 项目版本、包信息、Python 依赖 | `pyproject.toml` | README、测试、安装命令 |
| 文档站依赖和脚本命令 | `package.json` | README、附录 E |
| 图片资产 | `assets/` | `docs/public/assets/` |
| 教程正文 | `docs/` Markdown | VitePress 站点 |
| Python 实验实现 | `code/` | `tests/`、`notebooks/`、文档说明 |
| Notebook 交互演示 | `notebooks/` | 本地 Jupyter、云端 Notebook |
| 测试规则 | `tests/` | CI、本地验证 |
| 外部模型与 API 元数据 | `configs/models.yaml` | 附录 B/C、代码实验、Notebook |
| 外部数据集元数据 | `configs/datasets.yaml` | 附录 B/C、代码实验、Notebook |
| 文档站构建产物 | 不提交 | `docs/.vitepress/dist/` |
| 本地依赖目录 | 不提交 | `node_modules/`、虚拟环境 |

判断规则：

- 算法或数据生成逻辑变化，优先改 `code/`。
- 测试只验证 `code/` 的行为，不复制一份算法。
- Notebook 只做参数调整、可视化和演示编排，不复制一份核心实现。
- 文档正文只解释概念和用法，不承载可运行实现的第二份源头。
- 模型和数据集说明只在 registry 中维护元数据，正文和代码引用 registry 中的名称或约定。

---

## 图片资产同步

本项目约定：

- `assets/` 是图片资产的唯一源头。
- `docs/public/assets/` 是 VitePress 使用的静态资源镜像目录。
- 不手动编辑 `docs/public/assets/`。
- 文档站启动或构建前，由脚本自动把 `assets/` 同步到 `docs/public/assets/`。

同步脚本：

```bash
python scripts/sync_assets.py
```

等价行为：

```text
删除 docs/public/assets/
复制 assets/ -> docs/public/assets/
```

这样可以避免同一张图片在两个目录里手动维护。

---

## 文档站命令

文档站命令由 `package.json` 管理。

| 命令 | 作用 |
|------|------|
| `npm run docs:sync-indexes` | 从各章 `README.md` 生成对应 `index.md` |
| `npm run docs:sync-assets` | 同步 `assets/` 到 `docs/public/assets/` |
| `npm run docs:dev` | 同步章节首页和图片，并启动本地 VitePress 开发服务 |
| `npm run docs:build` | 同步章节首页和图片，并构建 VitePress 静态站点 |
| `npm run docs:preview` | 预览已经构建好的静态站点 |

本地写文档时，通常只需要运行：

```bash
npm run docs:dev
```

发布构建时运行：

```bash
npm run docs:build
```

---

## 发布前检查

在准备推送或合并前，建议按下面顺序检查：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -q
npm run docs:build
```

含义：

- 第一条检查 `code/` 和 `tests/` 的行为是否稳定。
- 第二条检查文档站是否能同步图片并正常构建。
- `npm run docs:build` 已经包含 `docs:sync-indexes` 和 `docs:sync-assets`，所以不需要先单独执行同步命令。

---

## GitHub Pages 发布

GitHub Pages 使用 `.github/workflows/deploy-docs.yml` 中的自定义 workflow 发布。

推送到 `main` 后，如果改动路径匹配 workflow 触发条件，GitHub Actions 会执行：

```bash
npm run docs:build
```

因为 `docs:build` 已经包含图片同步步骤，所以远程发布时不需要手动提交 `docs/public/assets/`。

需要触发文档站发布的常见路径包括：

- `assets/**`
- `docs/**`
- `scripts/**`
- `package.json`
- `package-lock.json`
- `.github/workflows/deploy-docs.yml`

---

## 测试维护规则

测试目录的职责是验证章节脚本是否可导入、可运行、输出形状和关键行为是否稳定。

约定：

- `tests/` 复用 `code/` 中的函数、类和实验入口。
- 不在测试中复制 LMS、Adam、Attention、MLP、RNN、CLIP 等核心算法。
- `tests/conftest.py` 中的 `load_code_module` 用于按文件路径加载 `code/` 下的脚本，避免和 Python 标准库 `code` 模块重名。
- 测试数据可以在测试内构造，但算法主体应来自 `code/`。

推荐验证命令：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -q
```

其中 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 用于避免本机全局 pytest 插件影响本项目测试。

---

## Notebook 维护规则

Notebook 保留，但定位是交互实验补充。

约定：

- Notebook 不作为教程正文的第二份源头。
- Notebook 不作为算法实现的第二份源头。
- 核心实验逻辑优先复用 `code/` 下的脚本。
- Notebook 通过 `notebooks/project.py` 中的 `load_code_module` 加载脚本。
- 如果 `code/` 中函数签名变化，需要同步更新对应 Notebook 的调用方式。
- 如果正文逻辑变化，Notebook 只同步核心实验和关键图示，不复制整篇解释。

示例：

```python
from pathlib import Path
import os
import sys
import subprocess

if not Path("notebooks/bootstrap.py").exists():
    root = Path("/content/signal-to-intelligence")
    if not root.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/datawhalechina/signal-to-intelligence.git", str(root)],
            check=True,
        )
    os.chdir(root)

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from notebooks.bootstrap import load_code_module

self_attention = load_code_module("code/ch04_transformer/self_attention.py")
```

---

## 模型与数据集维护规则

本项目默认保证正文阅读、文档构建、测试和轻量实验不依赖 API Key、GPU、大模型权重或大型外部数据集。大模型 API、本地推理和外部数据集都属于可选增强能力。

### API 与本地模型的分工

| 场景 | 默认策略 | 说明 |
|------|----------|------|
| 第 1-4 章轻量实验 | 本地生成数据或小型依赖 | 不需要 API、GPU 或模型下载 |
| 第 5-7 章 LLM 行为演示 | API 或本地模型可选（llama.cpp / Ollama 等） | 默认代码应能跳过或降级，不阻塞测试 |
| 第 7 章多模态重度实验 | 本地模型可选 | 需要显式安装依赖和下载模型 |
| 第 8 章工程部署 | API、本地、云端都可作为案例 | 用于解释部署、成本、量化和推理权衡 |
| 重型训练复现 | 不作为默认要求 | 用流程图、toy demo 或伪实验解释原理 |

### Registry

模型和数据集元数据统一登记在：

```text
configs/models.yaml
configs/datasets.yaml
```

登记内容包括：

- 资源名称和用途。
- 适用章节。
- API / 本地 / 生成数据 / 外部数据集等访问方式。
- 是否为默认必需。
- 环境变量、安装说明或附录链接。
- 许可证和硬件要求的提示。

这些 registry 只记录元数据，不保存 API Key、模型权重、数据文件或私有下载地址。

### 提交规则

- 大模型权重不提交，统一放在 `models/` 或提供商缓存目录。
- 大型数据集不提交，统一放在 `data/raw/`、`data/cache/`、`data/external/` 或提供商缓存目录。
- 小型项目自有 fixture 可以提交，但必须说明来源和许可。
- 外部模型必须登记到 `configs/models.yaml`。
- 外部数据集必须登记到 `configs/datasets.yaml`。
- API Key 只通过 `.env` 或环境变量读取，不写入文档、Notebook、代码或 registry。
- 测试不依赖真实 API、网络下载、大模型权重或大型数据集。
- Notebook 可以包含 API 或模型调用，但必须标注可选，并提供跳过逻辑或轻量替代路径。

---

## Makefile 使用

`Makefile` 是项目常用操作的快捷入口，不是必须使用的工具。所有可用命令可以通过：

```bash
make help
```

查看。

常用命令：

| 命令 | 作用 |
|------|------|
| `make install` | 安装核心 Python 依赖，等价于 `pip install -e .` |
| `make install-ml` | 安装深度学习依赖 |
| `make install-llm` | 安装 LLM API 依赖 |
| `make install-rag` | 安装 RAG 依赖 |
| `make install-hf` | 安装 Hugging Face 生态依赖 |
| `make test` | 运行全部测试 |
| `make test-cov` | 运行测试并生成覆盖率报告 |
| `make run-exp-ch01` | 运行第 1 章主要离线实验 |
| `make run-exp-ch02` | 运行第 2 章主要离线实验 |
| `make run-exp-ch03` | 运行第 3 章主要离线实验 |
| `make run-exp-ch04` | 运行第 4 章主要离线实验 |
| `make run-exp-ch05` | 运行第 5 章离线实验 |
| `make run-exp-ch06` | 运行第 6 章 RAG 演示 |
| `make run-exp-ch07` | 运行第 7 章轻量多模态实验 |
| `make run-exp-ch08` | 运行第 8 章工程实践演示 |
| `make run-all-exp` | 运行所有默认离线实验 |
| `make clean` | 删除 Python 构建产物和缓存 |
| `make clean-cache` | 删除测试、类型检查和覆盖率缓存 |

`Makefile` 中的章节实验命令也承担了“每章主实验清单”的作用。对不熟悉 Makefile 的读者，可以优先使用 README 和附录 C 中列出的原始 `python code/...` 命令。

---

## 工程文件说明

| 文件 / 目录 | 作用 |
|------------|------|
| `requirements.txt` | 便利入口，等价于 `pip install -e .`，实际依赖由 `pyproject.toml` 管理 |
| `pyproject.toml` | 依赖单点维护：核心依赖 + 可选依赖组（`pip install -e ".[llm]"` 等） |
| `Makefile` | 常用命令快捷方式，`make help` 查看所有命令 |
| `.env.example` | API Key 模板，复制为 `.env` 后填入密钥 |
| `configs/models.yaml` | 外部模型和 API 元数据 registry |
| `configs/datasets.yaml` | 外部数据集元数据 registry |
| `data/` | 小型 fixture 和本地数据说明；大型数据不提交 |
| `deploy/` | Docker Compose 配置（CPU 版 + GPU 版），见 [deploy/README.md](../../deploy/README.md) |
| `code/` | 所有可运行代码实验，按章节分目录 |
| `docs/` | 教程文档，按章节分目录，每章含 `extensions/` 扩展内容 |
| `notebooks/` | 交互实验补充，通过 `notebooks/project.py` 复用 `code/` |
| `assets/` | 代码实验生成的图表（由脚本自动生成，不手动编辑） |

工程文件说明与维护约定统一遵循“单一源头”原则：README 只保留入口级信息，详细职责边界见本附录前文。

---

## 维护原则

- 元信息以 `pyproject.toml` 为准。
- 图片源文件以 `assets/` 为准。
- 文档内容以 `docs/` 下的 Markdown 为准。
- Python 实验实现以 `code/` 为准。
- 测试以复用 `code/` 为准，不复制核心算法。
- Notebook 是交互补充，以调用 `code/` 为准。
- 模型和数据集元数据以 `configs/` 下的 registry 为准。
- 文档站发布产物 `docs/.vitepress/dist/` 不提交。
- 本地依赖目录 `node_modules/` 不提交。
- 过程文档和临时文件放入 `tmpdoc/`，不进入正式文档结构。

---

## 临时文件和归档

维护过程中可以使用 `tmpdoc/` 存放过程记录、备份和中间方案，但它不是正式文档入口。

约定：

- 有长期价值的内容整理到 `docs/`。
- 临时备份和过程草稿保留在 `tmpdoc/`。
- 不从 README 或 VitePress sidebar 链接到 `tmpdoc/`。
- 明显本地生成物不提交，例如缓存、构建产物、虚拟环境、`node_modules/`。
