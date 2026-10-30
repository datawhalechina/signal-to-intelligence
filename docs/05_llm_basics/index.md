<!-- AUTO-GENERATED from README.md. Do not edit index.md directly. -->

# 第5章：LLM基础

## 章节概览

本章回答一个核心问题：**LLM 是怎么从文本数据训练成可用助手的？**

![LLM 从预测到生成的统一链路](/diagrams/ch05-llm-prediction-generation.svg)

*章节主图：LLM 的统一工作链路。训练和生成共享“表示—注意力—概率预测”主干，只是在监督信号和下一步输入来源上不同。*

本章主线不是记住模型名称，而是理解一条训练闭环：**分词 → 预训练 → 数据配比 → 基础模型 → 微调 → 对齐 → 评估**。模型家族放在这条链路中作为案例，用来解释不同训练选择如何影响能力、成本和可用性。

## 统一关系：表示、注意力、预测与生成

第 4 章的注意力机制解决“当前表示应该从哪些位置聚合信息”；本章的预训练则解决“如何通过预测目标把这种信息处理能力学出来”。四个概念可以统一为一条链：

```text
信息表示：文本 → token → embedding → 上下文隐藏状态
    ↓
信息交互：注意力根据内容动态聚合上下文
    ↓
概率预测：对下一个 token 的可能性进行估计
    ↓
序列生成：把预测出的 token 放回上下文，重复上述过程
```

因此，LLM 并不是先“理解”再“生成”两个完全分开的模块。它在每个位置都根据已有上下文形成表示、估计下一个 token 的分布，并在自回归循环中把预测转化为输出。训练时，真实下一个 token 提供监督；生成时，模型自己的输出成为下一步输入。

从信息处理角度看：Embedding 负责把离散对象变成可计算表示，Attention 负责选择性聚合，输出层负责概率估计，采样策略负责在不确定性下作出生成决策。这个统一视角也能帮助理解第 6 章的 Prompt、微调、RAG 和 Agent：它们主要改变输入信息、模型参数或外部反馈，而不是改变这条基本生成链。

## 小节目录

### 训练流程（5.1-5.2）

**5.1 预训练** — [📖 阅读](01_pretraining.md)
- Tokenization 与 BPE：文字如何变成数字
- Token Embedding：从离散符号到连续向量
- Decoder-only 架构为什么赢了
- 预训练目标：下一词预测
- Scaling Laws 与非线性评测变化
- GPT 系列演化

**5.2 训练数据** — [📖 阅读](02_training_data.md)
- 数据来源、许可与版本记录
- 去重、质量过滤和评测污染检查
- 用固定预算的消融实验调整数据配比

### 基础模型与适配（5.3-5.4）

**5.3 模型家族与选型** — [📖 阅读](03_model_families.md)
- Decoder-only、Encoder-Decoder、稠密与 MoE 路线
- 基础模型、指令模型和专用模型
- 自部署权重与托管 API 的工程约束

**5.4 微调与 LoRA** — [📖 阅读](04_finetuning.md)
- SFT 的数据与损失
- LoRA 的参数结构与适用边界
- 如何判断 Prompt、RAG 或微调更匹配当前瓶颈

### 对齐与评估（5.5-5.6）

**5.5 偏好对齐** — [📖 阅读](05_rl_alignment.md)
- SFT、奖励模型与策略优化的分工
- RLHF 的 KL 约束与工程成本
- DPO 的假设、流程和评测边界

**5.6 模型评估** — [📖 阅读](06_evaluation.md)
- 为什么评估 LLM 很难
- 主流 Benchmark（MMLU、HumanEval、GSM8K、MT-Bench）
- 人工评估与 Chatbot Arena
- Goodhart's Law：Benchmark 刷分问题

## 核心问题

完成本章后，你应该能回答：

1. Tokenization 解决了什么问题？BPE 的核心思想是什么？
2. 预训练如何让模型从无标注文本中学到知识？
3. Scaling Laws 告诉我们什么？Chinchilla 定律的实践意义？
4. 训练数据的质量和配比如何影响模型能力？
5. 为什么模型家族应该作为训练路线案例，而不是学习结构中心？
6. SFT 学到的是知识还是行为模式？
7. LoRA 减少了哪些参数与存储成本，哪些成本仍然存在？
8. RLHF 和 DPO 有什么区别？
9. 如何评估一个 LLM？Benchmark 的局限是什么？

## 本章学习闭环

1. 用章节主图区分表示、注意力、概率预测和逐 token 生成；
2. 运行 `bpe_tokenization.py`、`autoregressive_generation.py` 或 `lora_visualization.py` 中的一个实验；
3. 完成最低实践任务，用这四个词解释参数变化发生在哪一层；
4. 进入第 6 章时，继续判断 Prompt、微调、RAG、Agent 分别改变输入、参数、上下文或执行流程中的哪一部分。

## 最低实践任务

使用本章的 BPE、采样或 LoRA 可视化实验，改变一个关键参数并记录结果；再用“表示—注意力—预测—生成”四个词解释变化发生在哪一层。完成标准：能区分训练阶段的 next-token prediction 与生成阶段的逐步采样。

## 代码实验

本章共有 **9 个代码脚本**，覆盖 BPE 分词、Scaling Laws、自回归生成、训练数据分析、模型家族演化、LoRA 可视化、RLHF 流程、评估基准对比，以及 LLM API 调用。

| 小节 | 脚本 | 内容 |
|------|------|------|
| 5.1 | [`bpe_tokenization.py`](../../code/ch05_llm_basics/bpe_tokenization.py) | BPE 算法实现，展示词表合并过程 |
| 5.1 | [`scaling_laws.py`](../../code/ch05_llm_basics/scaling_laws.py) | 幂律拟合与离散指标的阈值现象 |
| 5.1 | [`autoregressive_generation.py`](../../code/ch05_llm_basics/autoregressive_generation.py) | 5种采样策略对比（temperature/top-k/top-p） |
| 5.2 | [`training_data_composition.py`](../../code/ch05_llm_basics/training_data_composition.py) | 数据构成饼图、各阶段数据量、Phi 效率对比 |
| 5.3 | [`model_families_evolution.py`](../../code/ch05_llm_basics/model_families_evolution.py) | 模型家族时间线气泡图、开源 vs 闭源对比 |
| 5.4 | [`lora_visualization.py`](../../code/ch05_llm_basics/lora_visualization.py) | LoRA 矩阵分解原理、rank vs 误差、参数量对比 |
| 5.5 | [`rlhf_pipeline.py`](../../code/ch05_llm_basics/rlhf_pipeline.py) | 对齐前后奖励分布、PPO 训练曲线、RLHF vs DPO |
| 5.6 | [`benchmark_comparison.py`](../../code/ch05_llm_basics/benchmark_comparison.py) | 主流模型基准对比、Chatbot Arena ELO 排行 |
| 5.1-5.5 | [`llm_api_demo.py`](../../code/ch05_llm_basics/llm_api_demo.py) | API 调用、ICL 演示、Prompt 对比（需 API Key） |

## 扩展内容

### LLM 训练深度细节 — [📖 阅读](extensions/llm_training_details.md)
- 预训练的具体过程（数据准备、分词、训练配置）
- Scaling Laws 的数学基础
- 规模效应、指标阈值与 LLM 的局限

### 对齐训练深度细节 — [📖 阅读](extensions/alignment_details.md)
- PPO 在 RLHF 中的 4 步循环与 clip 操作
- KL 散度约束的数学含义与 β 系数
- DPO 完整数学推导（Bradley-Terry 模型）
- GRPO 与 o1 风格推理强化学习

### 微调方法调研 — [📖 阅读](extensions/finetuning_survey.md)
- LoRA 变体（QLoRA、AdaLoRA、DoRA）
- Prefix Tuning、Adapter、Prompt Tuning
- 各方法对比表与选择指南

### 评估方法调研 — [📖 阅读](extensions/evaluation_survey.md)
- 主流 Benchmark 详解
- 人工评估流程与 Chatbot Arena 机制
- 中文评估（C-Eval、CMMLU）

### 模型推理优化 — [📖 阅读](extensions/inference_optimization.md)
- KV Cache：为什么自回归生成需要缓存
- Flash Attention：IO 复杂度分析
- 量化原理（INT8/INT4）

### 信息论基础 — [📖 阅读](extensions/information_theory_basics.md)
- 熵、交叉熵、KL 散度的数学基础
- 与 LLM 训练目标的联系

## 关键连接点

### Transformer → LLM 训练流程

```
第4章：Transformer 架构
    ↓ decoder-only + 因果掩码
预训练（5.1）：在海量文本上预测下一词
    ↓ 数据质量与配比（5.2）+ 计算预算
基础模型（5.3）：不同训练路线和工程取舍
    ↓ SFT / LoRA / 任务适配（5.4）
对齐后的模型
    ↓ RLHF/DPO（5.5）
可用的 LLM 助手
    ↓ 评估（5.6）
验证效果，指导下一轮训练
```

---

**下一章：** [第6章：LLM应用](../06_llm_applications/README.md)
