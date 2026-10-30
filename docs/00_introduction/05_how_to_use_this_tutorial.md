# 如何使用本教程

教程内容分为正文、代码、Notebook 和项目。它们解决不同问题，不是四份重复教材。

| 入口 | 用途 | 内容源头 |
|------|------|----------|
| 在线文档 | 阅读概念、公式、边界和章节衔接 | `docs/` |
| Python 脚本与测试 | 复现实验、修改参数、验证行为 | `code/`、`tests/` |
| Notebook | 交互运行和课堂演示 | `notebooks/`，核心逻辑复用 `code/` |
| 项目 | 组织数据、对照、指标、失败案例和结论 | `projects/` |

## 正文和扩展的边界

正文面向工程理解：定义需要准确，类比需要说明边界，结论需要给出适用条件。扩展文章补充数学推导、专题背景和更重的实现，不是进入下一章的必要条件。

遇到跨章节术语时，先查[附录 G：术语表](../appendix/G_glossary.md)；需要核对符号和公式时查[附录 A：数学与符号参考](../appendix/A_math_reference.md)；需要判断某章的数学前置时查[附录 D：数学前置导航](../appendix/D_math_quick_reference.md)。

每章的建议顺序是：

```text
主图与核心问题 → 正文 → 代码实验 → 最低实践
                         ↓
              需要时再读扩展或做项目
```

最低实践通常只需修改已有脚本。需要多个实验、统一指标或报告时，再建立项目目录。

## 运行边界

| 章节 | 默认实验 | 可选依赖 |
|------|----------|----------|
| 第1—4章 | NumPy、SciPy、Matplotlib 等离线实验 | 深度学习框架用于部分模型示例 |
| 第5章 | Token、预训练和生成机制的离线实验 | API 或本地模型 |
| 第6章 | Prompt、微调、RAG、Agent 的轻量模拟 | 真实模型、向量数据库 |
| 第7章 | 视觉与非视觉信号表示、模态对齐的轻量实验 | 真实多模态权重与 GPU |
| 第8章 | 成本、推理和可靠性分析 | 真实部署环境 |

准确的脚本清单和依赖见[代码运行指南](../appendix/C_code_guide.md)。

## 本地运行

```bash
pip install -e .
make test
make run-exp-ch04
```

需要 API、GPU 或额外模型的实验不会放进默认离线验证。环境配置见[附录 B](../appendix/B_environment_setup.md)。

## Notebook

Notebook 用于逐步执行和调参，不复制完整正文。每章对应一个交互入口：

| 章节 | Notebook | Colab |
|------|----------|-------|
| 第1章 | `ch01_dsp_interactive.ipynb` | [打开](https://colab.research.google.com/github/datawhalechina/signal-to-intelligence/blob/main/notebooks/ch01_dsp_interactive.ipynb) |
| 第2章 | `ch02_optimization_interactive.ipynb` | [打开](https://colab.research.google.com/github/datawhalechina/signal-to-intelligence/blob/main/notebooks/ch02_optimization_interactive.ipynb) |
| 第3章 | `ch03_deep_learning_interactive.ipynb` | [打开](https://colab.research.google.com/github/datawhalechina/signal-to-intelligence/blob/main/notebooks/ch03_deep_learning_interactive.ipynb) |
| 第4章 | `ch04_transformer_interactive.ipynb` | [打开](https://colab.research.google.com/github/datawhalechina/signal-to-intelligence/blob/main/notebooks/ch04_transformer_interactive.ipynb) |
| 第5章 | `ch05_llm_interactive.ipynb` | [打开](https://colab.research.google.com/github/datawhalechina/signal-to-intelligence/blob/main/notebooks/ch05_llm_interactive.ipynb) |
| 第6章 | `ch06_llm_applications_interactive.ipynb` | [打开](https://colab.research.google.com/github/datawhalechina/signal-to-intelligence/blob/main/notebooks/ch06_llm_applications_interactive.ipynb) |
| 第7章 | `ch07_multimodal_interactive.ipynb` | [打开](https://colab.research.google.com/github/datawhalechina/signal-to-intelligence/blob/main/notebooks/ch07_multimodal_interactive.ipynb) |
| 第8章 | `ch08_engineering_interactive.ipynb` | [打开](https://colab.research.google.com/github/datawhalechina/signal-to-intelligence/blob/main/notebooks/ch08_engineering_interactive.ipynb) |

本地启动使用 `jupyter notebook`。Notebook 通过 `notebooks/project.py` 加载仓库代码。

## 项目

项目不以“代码能运行”为完成标准，而要同时给出问题、数据、基线、指标、失败案例和结论。当前目录约定为：

```text
projects/<project-name>/
├── README.md
├── main.py
├── experiment.py       # 按需
├── evaluate.py         # 按需
└── outputs/            # 本地生成，不提交
```

详细要求见[项目说明](../../projects/README.md)。维护文档、图片、代码和 Notebook 时，遵循[项目维护说明](../appendix/E_project_maintenance.md)。
