# 从信号到智能

从数字信号处理到大语言模型的思想演化与代码实践。

本教程面向具有 DSP、控制、通信或机器学习基础的工程学习者。重点不是罗列模型，而是追踪一条可迁移的信息处理主线：信息如何被表示、变换、判断、生成，并在真实系统中持续改进。

![从信号处理到智能系统的教程主线](/diagrams/tutorial-roadmap.svg)

*章节主图：第 1 章从信号与观测出发，第 2—5 章逐步建立表示、学习、注意力和生成，第 6—7 章进入应用与多模态，第 8 章用评估和工程反馈形成闭环。*

## 从哪里开始

| 目标 | 建议入口 |
| --- | --- |
| 建立完整思想链路 | 从[第 1 章](/01_dsp/)开始，依次完成各章最低实践任务 |
| 已有机器学习基础 | 从[第 3 章](/03_deep_learning_fast/)开始，再回看第 1—2 章的桥接内容 |
| 聚焦 Transformer 与 LLM | 从[第 4 章](/04_transformer/)开始，继续阅读第 5—6 章 |
| 运行代码或检查环境 | 阅读[代码运行指南](/appendix/C_code_guide) |
| 选择 Prompt、微调、RAG 或 Agent | 阅读[第 6 章](/06_llm_applications/) |

路径只表示知识依赖，不规定学习时长。每章都提供最低实践任务和完成标准，可根据自己的背景决定是否阅读扩展内容。

## 章节目录

| 章节 | 回答的问题 | 核心内容 |
| --- | --- | --- |
| [第 0 章：导论](/00_introduction/) | 如何使用这套教程？ | 定位、学习路径、结构和工具 |
| [第 1 章：DSP 基础](/01_dsp/) | 如何从观测中提取信息？ | 时频表示、滤波、随机信号、检测与估计 |
| [第 2 章：优化与机器学习](/02_optimization/) | 如何把目标转化为可学习的参数？ | 损失、梯度、传统机器学习和数值方法 |
| [第 3 章：深度学习快速通道](/03_deep_learning_fast/) | 如何从数据中学习表示？ | CNN、序列建模、检测和 Transformer 桥接 |
| [第 4 章：Transformer](/04_transformer/) | 如何根据上下文动态聚合信息？ | QKV、多头注意力、位置和架构 |
| [第 5 章：LLM 基础](/05_llm_basics/) | 如何从下一个 token 预测得到生成模型？ | 预训练、数据、微调、对齐和评估 |
| [第 6 章：LLM 应用](/06_llm_applications/) | 不同应用方法分别解决什么问题？ | Prompt、微调、RAG、Agent 和系统设计 |
| [第 7 章：多模态 LLM](/07_multimodal_llm/) | 不同信号怎样与语言建立可验证的对应关系？ | 视觉主干、非视觉信号、表示对齐和语言接口 |
| [第 8 章：工程实践与部署](/08_llm_engineering/) | 如何在云端与端侧约束下保留任务信息？ | 压缩、推理、评估、生产闭环和端侧验收 |

## 快速验证代码环境

```bash
git clone https://github.com/datawhalechina/signal-to-intelligence.git
cd signal-to-intelligence

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
make test
```

核心测试和轻量实验使用 CPU 与本地模拟数据，不需要 API Key、GPU 或模型下载。需要模型、视觉或 RAG 依赖时，再按对应章节安装可选依赖。

## 理论、实验与项目的关系

| 目录 | 职责 |
| --- | --- |
| `docs/` | 解释概念、主线、边界和实验结论 |
| `code/` | 保存按章节组织的算法实验和可复用实现 |
| `projects/` | 组织基线、对照、评估、失败分析和报告 |
| `notebooks/` | 用于交互式讲解和参数探索，不复制核心算法 |
| `tests/` | 验证核心行为、项目结构和文档约束 |

先完成章节最低实践任务，再进入[章节项目说明](https://github.com/datawhalechina/signal-to-intelligence/blob/main/projects/README.md)。第一阶段项目覆盖信号检测和视觉检测比较，后续再扩展到音频事件检测、图文检索和综合多模态系统。

## 附录入口

- [附录 A：数学备忘](/appendix/A_math_reference)
- [附录 B：环境配置](/appendix/B_environment_setup)
- [附录 C：代码运行指南](/appendix/C_code_guide)
- [附录 D：数学基础速查](/appendix/D_math_quick_reference)
- [附录 E：项目维护说明](/appendix/E_project_maintenance)
- [附录 F：图像与可视化规范](/appendix/F_visual_guide)
