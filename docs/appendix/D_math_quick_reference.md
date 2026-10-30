# 附录 D：数学前置导航

本附录回答“为了读某一章，我需要先补什么”。它不再维护第二套完整公式：统一符号、公式和适用条件以[附录 A：数学与符号参考](A_math_reference.md)为准，术语边界见[附录 G：术语表](G_glossary.md)。

使用建议：先在“章节前置总览”定位最低前置，再按主题查看“需要掌握到什么程度”。不必在开始教程前一次学完所有数学。

## 目录

- [D.1 章节前置总览](#d1-章节前置总览)
- [D.2 复数与信号变换](#d2-复数与信号变换)
- [D.3 数值方法与优化](#d3-数值方法与优化)
- [D.4 向量空间与图结构](#d4-向量空间与图结构)
- [D.5 概率与信息论](#d5-概率与信息论)
- [D.6 随机过程与线性系统](#d6-随机过程与线性系统)
- [D.7 按问题快速查找](#d7-按问题快速查找)

---

## D.1 章节前置总览

| 章节 | 最低前置 | 能继续学习的判断标准 | 公式入口 |
| --- | --- | --- | --- |
| 第 1 章 DSP | 复数、三角函数、求和与卷积 | 能解释幅度/相位、时域/频域和卷积输出 | [A.3](A_math_reference.md#a-3-复数基础)、[A.4](A_math_reference.md#a4-信号处理) |
| 第 2 章优化 | 导数、梯度、概率均值 | 能读懂目标函数、梯度更新和训练/验证划分 | [A.1](A_math_reference.md#a1-线性代数与微积分)、[A.5](A_math_reference.md#a5-优化与统计学习) |
| 第 3 章深度学习 | 矩阵乘法、链式法则、卷积 | 能追踪张量维度并说明反向传播更新什么 | [A.1](A_math_reference.md#a1-线性代数与微积分)、[A.6](A_math_reference.md#a6-深度学习) |
| 第 4 章 Transformer | 点积、softmax、矩阵维度 | 能检查 Q/K/V 的形状和 softmax 的归一化轴 | [A.7](A_math_reference.md#a7-transformer) |
| 第 5 章 LLM | 条件概率、交叉熵 | 能区分训练目标、下一 token 分布和生成循环 | [A.2](A_math_reference.md#a2-概率统计与信息论)、[A.8](A_math_reference.md#a8-llm-训练与生成) |
| 第 6 章 LLM 应用 | 第 5 章最低前置 | 能区分模型概率输出、外部证据和系统评价指标 | [A.8](A_math_reference.md#a8-llm-训练与生成)、[A.10](A_math_reference.md#a10-数值计算与评价指标) |
| 第 7 章多模态 | 向量相似度、交叉熵、注意力 | 能解释“编码—投影—对齐—融合”各自改变什么 | [A.9](A_math_reference.md#a9-多模态表示与对齐) |
| 第 8 章工程 | 数值精度、复杂度、评价口径 | 能同时报告质量、时延、显存和吞吐量 | [A.10](A_math_reference.md#a10-数值计算与评价指标) |

---

## D.2 复数与信号变换

### 学到什么程度

- 能在 $a+jb$ 与 $re^{j\theta}$ 之间转换。
- 知道复指数同时携带幅度与相位，而不是“虚构的频率”。
- 能读懂 DFT 的求和下标、正负号和归一化约定。
- 能区分信号能量、平均功率和 dB 形式的 SNR。

### 学习入口

- 公式速查：[A.3 复数基础](A_math_reference.md#a-3-复数基础)、[A.4 信号处理](A_math_reference.md#a4-信号处理)
- 正文实践：[第 1 章傅里叶变换](../01_dsp/02_fourier.md)、[滤波器与卷积](../01_dsp/03_filters.md)
- 进阶：[随机过程](../01_dsp/extensions/stochastic_processes.md)、[线性系统](../01_dsp/extensions/linear_systems.md)

---

## D.3 数值方法与优化

### 学到什么程度

- 能用链式法则解释梯度从损失传播到参数的路径。
- 知道学习率、梯度尺度和浮点精度会共同影响优化。
- 知道中心差分步长不是越小越好；截断误差与舍入误差需要平衡。
- 能区分凸优化的全局保证与深度网络的非凸训练。

### 关键边界

- 浮点误差模型 $\operatorname{fl}(x)=x(1+\delta)$ 需要正常舍入且不发生上溢或下溢。
- 中心差分在无量纲理想模型中的步长量级为 $h \sim \epsilon^{1/3}$，实际值仍依赖变量和函数尺度。
- 凸问题在满足相应条件时局部最优即全局最优；非凸问题通常缺少通用的全局最优保证。

### 学习入口

- 公式速查：[A.1 线性代数与微积分](A_math_reference.md#a1-线性代数与微积分)、[A.5 优化](A_math_reference.md#a5-优化与统计学习)、[A.10 数值计算](A_math_reference.md#a10-数值计算与评价指标)
- 正文实践：[第 2 章数值方法](../02_optimization/04_numerical_methods.md)
- 进阶：[凸分析](../02_optimization/extensions/convex_analysis.md)、[高级优化](../02_optimization/extensions/advanced_optimization.md)

---

## D.4 向量空间与图结构

### 学到什么程度

- 能把矩阵乘法解释为投影、相似度计算或加权聚合。
- 能区分点积、余弦相似度和欧氏距离的适用口径。
- 能检查注意力分数矩阵的两个序列轴，以及 softmax 沿哪个轴归一化。
- 能把注意力看作数据依赖的消息聚合，但不把注意力权重直接当作因果解释。

### 学习入口

- 公式速查：[A.1 线性代数](A_math_reference.md#a1-线性代数与微积分)、[A.7 Transformer](A_math_reference.md#a7-transformer)
- 正文实践：[第 4 章自注意力](../04_transformer/01_attention.md)
- 进阶：[向量空间基础](../04_transformer/extensions/vector_space_basics.md)、[图论基础](../04_transformer/extensions/graph_theory_basics.md)

---

## D.5 概率与信息论

### 学到什么程度

- 能区分概率、logit、损失和评价指标。
- 能把交叉熵解释为目标分布对预测分布的平均负对数概率。
- 能说明自回归模型如何把联合分布分解为一串条件概率。
- 知道 KL 散度不对称，困惑度比较依赖 tokenizer、数据和计分口径。

### 关键边界

- 交叉熵使用自然对数时，困惑度为 $\exp(H)$；使用以 2 为底的对数时为 $2^H$。
- 温度改变采样分布的尖锐程度，不是事实可靠性开关。
- 批内对比学习把其他样本当作负例，可能遇到假负例。

### 学习入口

- 公式速查：[A.2 概率与信息论](A_math_reference.md#a2-概率统计与信息论)、[A.8 LLM](A_math_reference.md#a8-llm-训练与生成)、[A.9 多模态对齐](A_math_reference.md#a9-多模态表示与对齐)
- 正文实践：[第 5 章预训练](../05_llm_basics/01_pretraining.md)、[第 7 章视觉—语言](../07_multimodal_llm/01_vision_language.md)
- 进阶：[信息论基础](../05_llm_basics/extensions/information_theory_basics.md)、[多模态训练细节](../07_multimodal_llm/extensions/multimodal_training_details.md)

---

## D.6 随机过程与线性系统

这两部分是第 1 章的推荐扩展，不是进入 Transformer 和 LLM 的硬性前置。

### 需要掌握的边界

- 宽平稳只约束均值和二阶相关结构，不等同于严格平稳。
- 一阶马尔可夫性表示给定当前状态后，下一状态与更早历史条件独立。
- 离散时间 LTI 系统的 BIBO 稳定条件是冲激响应绝对可和；“极点都在单位圆内”的写法还要求因果有理系统。
- RNN 的隐藏状态可以与状态空间模型类比，但非线性门控网络不等于线性系统。

### 学习入口

- 公式速查：[A.2 概率统计](A_math_reference.md#a2-概率统计与信息论)、[A.4 信号处理](A_math_reference.md#a4-信号处理)
- 进阶：[随机过程](../01_dsp/extensions/stochastic_processes.md)、[线性系统](../01_dsp/extensions/linear_systems.md)

---

## D.7 按问题快速查找

| 我想理解的问题 | 先看 | 再做 |
| --- | --- | --- |
| 傅里叶变换为什么使用复指数？ | [A.3 复数](A_math_reference.md#a-3-复数基础) | 第 1 章傅里叶实验 |
| CNN 与 DSP 卷积有什么差别？ | [A.4 卷积](A_math_reference.md#a4-信号处理)、[A.6 CNN](A_math_reference.md#a6-深度学习) | 第 3 章 CNN 实验 |
| 为什么注意力要除以 $\sqrt{d_k}$？ | [A.7 注意力](A_math_reference.md#a7-transformer) | 缩放点积注意力实验 |
| 训练和生成为什么都叫下一 token 预测？ | [A.8 自回归目标](A_math_reference.md#a8-llm-训练与生成) | 第 5 章采样实验 |
| LoRA 到底减少了什么？ | [A.8 LoRA](A_math_reference.md#a8-llm-训练与生成) | LoRA rank 可视化实验 |
| CLIP 如何实现跨模态对齐？ | [A.9 InfoNCE](A_math_reference.md#a9-多模态表示与对齐) | 第 7 章相似度矩阵实验 |
| 高分辨率为什么迅速增加成本？ | [A.9 patch 与计算量](A_math_reference.md#a9-多模态表示与对齐) | 第 7 章分辨率实验 |
| 如何避免只报告一个“准确率”？ | [A.10 评价指标](A_math_reference.md#a10-数值计算与评价指标) | 按任务补充误报、漏报和资源指标 |
