# 第6章：LLM应用

## 章节概览

本章介绍如何使用和定制 LLM，回答一个核心问题：**有了预训练好的 LLM，怎么用最低风险和成本解决实际问题？**

![LLM 应用方案选型图](/diagrams/ch06-solution-selection.svg)

*章节主图：Prompt、微调、RAG 和 Agent 的并列选型关系。它们分别主要改变输入、参数、外部知识和执行流程，应从任务瓶颈而非技术热度出发选择。*

本章主线是应用方案选型，而不是某一种框架或单一技术。Prompt、微调、RAG 和 Agent 分别改变输入、参数、外部知识和执行流程；它们可以组合，但不应被理解成必须依次升级的四个阶段。每条路线都要回答同一组问题：适合什么任务、解决什么瓶颈、引入什么成本、失败时如何排查。

## 四种方法分别解决什么问题？

先定位瓶颈，再选择方法：

| 方法 | 主要改变什么 | 主要解决的问题 | 不擅长解决的问题 |
|---|---|---|---|
| Prompt | 输入指令、示例和输出约束 | 快速验证任务、澄清目标、控制格式 | 注入稳定的新知识，修复模型根本能力不足 |
| 微调 | 模型参数或参数高效适配器 | 固定风格、格式、任务行为和领域模式 | 持续变化的事实知识、实时数据访问 |
| RAG | 外部知识的表示、检索和上下文 | 新鲜知识、可追溯资料、私有文档问答 | 检索不到的知识、复杂行动流程、模型推理能力不足 |
| Agent | 多步执行流程和工具调用 | 查询外部系统、执行操作、完成长流程任务 | 无边界地提高可靠性；步骤越多，错误和成本越容易累积 |

可以用下面的顺序进行低风险决策：

```text
先用 Prompt 明确任务和验收标准
  ↓ 如果问题是稳定行为/格式，再考虑微调
  ↓ 如果问题是外部或变化知识，再考虑 RAG
  ↓ 如果问题需要查询、计算或执行动作，再引入 Agent
```

这是一棵按问题定位的决策树，不是技术等级排序。RAG 是一个重要的外部知识增强案例，但它与微调、Agent 处于并列关系，不能代表整个 LLM 应用章节。

## 小节目录

### 基础使用（6.1）

**6.1 Prompt 工程** — [📖 阅读](01_prompt_engineering.md)
- Zero-shot / Few-shot / Chain-of-Thought
- 角色设定与结构化输出
- 低成本验证、失败模式和迭代优化

### 适配方法（6.2-6.3）

**6.2 微调：让 LLM 适配你的任务** — [📖 阅读](02_finetuning.md)
- 任务微调 vs 对齐 SFT 的区别
- LoRA：低秩分解降低微调成本
- 何时用微调，何时不要微调

**6.3 RAG（检索增强生成）** — [📖 阅读](03_rag.md)
- 为什么 LLM 需要外部知识
- 检索 + 生成的工作流程
- Embedding 与向量检索
- 知识新鲜度、可解释性和检索失败模式

### 扩展能力（6.4）

**6.4 Agent 框架** — [📖 阅读](04_agent.md)
- LLM 作为规划者
- 工具调用与 ReAct 模式
- 多步任务的执行流程
- 工具可靠性、错误累积和权限边界

### 系统设计（6.5）

**6.5 LLM 系统设计** — [📖 阅读](05_system_design.md)
- 技术路线选择决策树（Prompt vs 微调 vs RAG vs Agent）
- 三种架构模式（简单问答 / RAG 问答 / Agent 工作流）
- 实战案例：客服机器人、代码助手、数据分析
- 模型选择、部署选项、成本和风险控制

## 核心问题

完成本章后，你应该能回答：

1. 如何通过 Prompt 设计充分发挥 LLM 的能力？
2. 任务微调和对齐 SFT 有什么区别？
3. LoRA 减少了哪些训练参数，何时仍需比较全量微调？
4. RAG 如何让 LLM 使用外部知识？
5. Agent 框架的核心是什么？ReAct 模式如何工作？
6. 面对一个实际任务，如何选择 Prompt / 微调 / RAG / Agent？
7. 如何根据成本、延迟、知识新鲜度、可维护性和风险选择 LLM 应用架构？

## 本章学习闭环

1. 先根据章节主图定位任务瓶颈：输入表达、稳定行为、外部知识或多步行动；
2. 运行 `prompt_demo.py`、`finetuning_demo.py`、`rag_demo.py`、`agent_demo.py` 中至少两个对照脚本；
3. 完成最低实践任务，为同一个问题写出四种方案的对象、依赖和风险；
4. 用 `system_design_demo.py` 复查最终选择是否满足质量、成本、延迟和风险边界。

## 最低实践任务

为一个具体问题分别写出 Prompt、微调、RAG、Agent 四种方案的“解决对象、额外依赖、主要风险”，再选择其中一种做轻量实验。完成标准：选择理由必须对应任务瓶颈，而不是因为某个技术更流行。

## 代码实验

本章共有 **5 个代码脚本**，覆盖 Prompt 技术对比、微调参数分析、RAG 向量检索、Agent ReAct 循环和系统设计选型框架。

| 小节 | 脚本 | 生成图表 | 内容 |
|------|------|---------|------|
| 6.1 Prompt 工程 | [`prompt_demo.py`](../../code/ch06_llm_applications/prompt_demo.py) | `ch06_prompt_techniques.png` | 用示意分数解释策略差异与迭代流程，不是模型 Benchmark |
| 6.2 微调 | [`finetuning_demo.py`](../../code/ch06_llm_applications/finetuning_demo.py) | `ch06_lora_parameters.png` | 全量微调 vs LoRA 参数量对比 |
| 6.3 RAG | [`rag_demo.py`](../../code/ch06_llm_applications/rag_demo.py) | `ch06_rag_vector_search.png` | 可运行的字符 n-gram 词法向量基线 + 手工二维 embedding 几何示意 |
| 6.4 Agent | [`agent_demo.py`](../../code/ch06_llm_applications/agent_demo.py) | `ch06_agent_error_accumulation.png` | ReAct 循环 + 错误累积曲线 |
| 6.5 系统设计 | [`system_design_demo.py`](../../code/ch06_llm_applications/system_design_demo.py) | `ch06_system_design.png` | 基于显式场景假设的四方案示意，不构成通用排名 |

**运行方式：**
```bash
python code/ch06_llm_applications/prompt_demo.py
python code/ch06_llm_applications/finetuning_demo.py
python code/ch06_llm_applications/rag_demo.py
python code/ch06_llm_applications/agent_demo.py
python code/ch06_llm_applications/system_design_demo.py
```

## 扩展内容

### 微调工程实践 — [📖 阅读](extensions/finetuning_advanced.md)
- 数据格式（Alpaca / ShareGPT）与质量检查
- 工具选型（PEFT、LLaMA-Factory、Unsloth）
- 训练配置参考与监控指标
- LoRA 权重合并与部署

### RAG 系统深度优化 — [📖 阅读](extensions/rag_advanced.md)
- 混合检索（BM25 + 向量）与重排序
- 分块策略与父子分块
- 上下文管理与 Lost in the Middle 问题
- RAGAS 评估框架与工具选型

### Agent 框架深度 — [📖 阅读](extensions/agent_advanced.md)
- 工具定义规范与可靠性
- ReAct 模式详解与多步推理控制
- 记忆系统与主流框架选型（LangChain / AutoGen / CrewAI）

### 推理部署优化 — [📖 阅读](extensions/inference_deployment.md)
- 量化方案选型（INT8/INT4/GPTQ/AWQ/GGUF）
- 推理框架对比（vLLM、llama.cpp、Ollama、TGI）
- 连续批处理与推测解码

## 关键连接点

### 预训练模型 → 任务适配

```
通用 LLM（第5章）
    ├─ Prompt 工程 → 改变输入与约束
    ├─ 微调（LoRA）→ 学习稳定任务行为
    ├─ RAG → 在推理时提供外部证据
    └─ Agent → 动态选择工具与行动
```

### Embedding → 外部信息整合

```
第5章：token embedding（离散 token 的向量表示）
    ↓ 共享“映射到向量空间”的思想，但训练目标和粒度不同
RAG：检索 embedding → 向量检索 → 找相关文档
其他应用：相似度、聚类、跨模态对齐和信息路由
```

**启示：** RAG 是 Embedding 和外部知识整合的一个代表性案例，但 Embedding 的应用范围不止检索；学习时应先掌握表示空间和相似度，再决定是否深入 RAG。

---

**下一步：** 阅读 [6.1 Prompt 工程](01_prompt_engineering.md)
