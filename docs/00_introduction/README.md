# 第0章：导论

本章说明教程为什么从信号处理讲到 LLM、如何选择学习入口，以及正文、代码和项目各自承担什么职责。

![从信号到智能的教程主线](/diagrams/tutorial-roadmap.svg)

*章节主图：信息先被表示和变换，再由模型学习、预测和生成；评估结果最终回到系统改进。*

## 本章内容

| 小节 | 回答的问题 |
|------|------------|
| [0.1 为什么从 LLM 重新理解信息处理](01_why_llm_era.md) | LLM 与信号处理、优化和深度学习有什么联系，边界在哪里？ |
| [0.2 学习路径](02_learning_paths.md) | 应从第 1 章还是第 4 章开始？ |
| [0.3 全书概览](03_overview.md) | 八章分别处理哪一类问题？ |
| [0.4 模型架构与工具链选择](04_tools_and_infrastructure.md) | API、本地模型和云端算力如何选择？ |
| [0.5 如何使用本教程](05_how_to_use_this_tutorial.md) | 正文、脚本、Notebook 和项目如何配合？ |

## 建议入口

- 有信号、通信或控制背景：从[第1章](../01_dsp/README.md)开始。
- 熟悉深度学习：读完本章后直接进入[第4章](../04_transformer/README.md)。
- 准备做 LLM 应用：先读[第5章](../05_llm_basics/README.md)和[第6章](../06_llm_applications/README.md)，缺少前置概念时回查第 4 章。
- 只想运行代码：从根目录 [Quick Start](../../README.md#quick-start) 和[代码运行指南](../appendix/C_code_guide.md)开始。

教程不要求固定进度。每章以完成最低实践并能解释结果为结束条件，详见[全书概览](03_overview.md#完成一章的判断标准)。
