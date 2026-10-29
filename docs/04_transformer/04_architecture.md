# 4.4 完整架构

**核心问题：** Transformer 的完整架构是什么？为什么这些组件要这样组合？

---

## 技术背景

### 核心问题

前面几节分别介绍了注意力、多头机制和位置编码，但真正可训练、可扩展的 Transformer，不只是把这些组件简单堆在一起。

这一节回答三个问题：
- 为什么注意力后面还要接 FFN？
- 为什么每个子层都配残差连接和 LayerNorm？
- 为什么 encoder、decoder 和 decoder-only 会分化成不同结构？

目标很简单：把 Transformer 从“组件列表”收成“设计逻辑完整的系统结构”。

### 关键概念

- **Transformer Block**：注意力 + FFN + 残差 + 归一化的基本计算单元
- **Residual Connection**：保留原始输入并叠加子层输出
- **LayerNorm**：对单个样本的特征维度做归一化
- **Feed-Forward Network (FFN)**：对每个 token 独立做非线性变换
- **Encoder-only / Encoder-Decoder / Decoder-only**：三种主流 Transformer 结构范式

可以把这一节理解成一张对应表：
- 注意力：token 之间如何交换信息
- FFN：单个 token 如何进一步加工
- 残差和 LayerNorm：深层训练为什么还能稳定
- decoder-only：生成式 LLM 为什么最终沿着自回归路线收敛

---

## Transformer 块

一个标准 Transformer block 可以概括为：

```text
输入
  ↓
多头自注意力
  ↓
残差连接 + 层归一化
  ↓
前馈网络（FFN）
  ↓
残差连接 + 层归一化
  ↓
输出
```

这个结构看上去像“两个子层串起来”，但每个部分都承担了不同角色：
- **注意力层**：负责在 token 之间交换信息
- **FFN 层**：负责对每个 token 做更强的非线性表示变换
- **残差连接**：负责让深层网络更容易训练
- **LayerNorm**：负责稳定数值分布和优化过程

---

## 为什么注意力后面还要接 FFN？

原因是：**注意力负责信息路由，FFN 负责局部特征加工。**

### 注意力做什么？

注意力回答的是：
- 我应该从哪些位置获取信息？
- 每个位置的信息应该占多大权重？

它更像一个动态的信息聚合器。

### FFN 做什么？

前馈网络对每个 token 的表示做独立的非线性变换：

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

它的作用是：
- 提升单个 token 表示的非线性表达能力
- 把注意力融合后的结果进一步变换成更适合下层使用的特征
- 让模型不仅会“看谁”，还会“怎么加工看到的信息”

可以把它理解为：
- 注意力层负责“通信”
- FFN 层负责“计算”

两者缺一不可。

如果只有注意力，模型更像在搬运信息；加上 FFN，模型才开始真正“加工信息”。

---

## 为什么要用残差连接？

残差连接的形式是：

$$\text{output} = \text{sublayer}(x) + x$$

它看起来只是把输入再加回来，但作用非常关键。

### 1. 保留原始信息通路
如果某一层暂时学不到有用变换，残差路径能提供一条更直接的信息通路，降低表示被层层变换破坏的风险。

### 2. 让深层网络更容易优化
深层模型常见的问题是梯度传递困难、训练不稳定。残差连接相当于给网络提供了更短的优化路径，使很多层堆叠后仍然可训练。

### 3. 更适合逐层 refinement
Transformer 的很多层并不是每层都“重写全部表示”，而是基于已有表示做渐进式修正。残差结构很适合这种“增量更新”的方式。

这也是 Transformer 能堆很深而不至于失控的关键原因之一。

---

## 为什么要用 LayerNorm？

LayerNorm 的形式是：

$$\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sigma} + \beta$$

它和 BatchNorm 不同，不依赖 batch 维度统计，而是对单个样本的特征维度做归一化。

### LayerNorm 的作用

#### 1. 稳定数值范围
深层网络中，不同层输出的数值尺度可能不断漂移。LayerNorm 可以把表示拉回更稳定的范围。

#### 2. 让训练更平滑
当不同 token、不同层的激活尺度差异太大时，优化会变得困难。LayerNorm 能缓解这种问题。

#### 3. 更适合序列模型
因为序列长度和 batch 组织方式经常变化，LayerNorm 比依赖 batch 统计的归一化更适合 Transformer 这类结构。

所以，LayerNorm 的核心不是“提高表达能力”，而是**提高可训练性和数值稳定性**。

这和第 2 章里讲的数值问题是一脉相承的：模型越大，越需要把数值分布控制住。

---

## 为什么 Transformer 可以层层堆叠？

单层注意力只进行一次信息聚合。堆叠多个 block 后，每层都在上一层表示的基础上继续计算注意力和前馈变换，因此信息可以经过多步组合。不同层具体编码什么需要通过实验分析，不能固定解释为“低层局部、高层语义”。

---

## 三种主流 Transformer 架构

### 1. Encoder-only

```text
输入序列
  ↓
多层 Transformer 编码器
  ↓
上下文化表示
```

**特点：**
- 每个位置都可以看整个输入
- 更适合理解任务，而不是逐词生成

**典型用途：**
- 文本分类
- 序列标注
- 表征学习

**代表模型：** BERT

---

### 2. Encoder-Decoder

```text
输入序列
  ↓
编码器：理解输入
  ↓
解码器：逐步生成输出
  ↓
输出序列
```

解码器包含两类注意力：
- **Masked Self-Attention**：只看已生成内容
- **Cross-Attention**：看编码器输出

**特点：**
- 适合输入和输出都很重要的映射任务
- 能显式地把“理解输入”和“生成输出”分开

**典型用途：**
- 机器翻译
- 摘要生成
- 图文描述

**代表模型：** 原始 Transformer、T5

---

### 3. Decoder-only

```text
已出现的 token
  ↓
Masked Self-Attention
  ↓
逐步预测下一个 token
```

**特点：**
- 只保留自回归生成路径
- 使用 causal mask，禁止看到未来 token
- 结构更统一，训练目标更简单

**典型用途：**
- next-token prediction
- 开放式文本生成
- 大语言模型预训练

**代表模型：** GPT 系列

---

## 为什么现代 LLM 大多采用 decoder-only？

这是 Chapter 4 最重要的桥接点之一。

### 1. 训练目标自然统一
decoder-only 直接对应 next-token prediction：给定前文，预测下一个 token。这种目标简单、统一，而且能直接利用海量无标注文本。

### 2. 生成过程天然一致

训练时学的是“根据前文预测后文”，推理时做的也是同样的事：不断追加新 token 并继续预测。

### 3. 架构更简洁，易于规模化
相比 encoder-decoder，decoder-only 少了一套编码器和 cross-attention 路径，整体结构更统一，在大规模预训练中更容易扩展。

### 4. 更适合开放式生成

LLM 不只是做固定输入到输出的映射，而是要持续生成自然语言、代码、推理过程甚至多轮对话。decoder-only 在这种开放式自回归任务里非常自然。

因此，现代 LLM 可以理解为：**把 Transformer 的自注意力与深层 block 结构，集中用于因果生成这条路线。**

这也是为什么 decoder-only 不只是一个“简化版结构”，而是和预训练目标、推理方式、扩展路径一起协同设计出来的结果。

## 完整流程示例：机器翻译 vs 语言模型

### 机器翻译（Encoder-Decoder）

```text
输入：英文句子
  ↓
编码器：理解英文上下文
  ↓
解码器：逐个生成法文单词
  ├─ 看已生成的法文（masked self-attention）
  └─ 看英文表示（cross-attention）
  ↓
输出：法文句子
```

### 语言模型（Decoder-only）

```text
输入：已经出现的文本
  ↓
多层 masked self-attention
  ↓
预测下一个 token
  ↓
把新 token 接回输入
  ↓
继续生成
```

## 实验结果：因果掩码

为了直观看到 decoder-only 结构与普通双向注意力的区别，本章提供了因果掩码实验：

- **代码实验：** [`code/ch04_transformer/causal_mask_demo.py`](../../code/ch04_transformer/causal_mask_demo.py)
- **作用：** 对比双向注意力和 causal mask 如何改变每个位置可见的信息范围

![Causal Mask](/assets/ch04_causal_mask.png)

*图4.4：因果掩码会屏蔽未来位置，使每个 token 只能利用自己和历史上下文。这正是生成式 LLM 的关键约束。*

**代码文件：** [`code/ch04_transformer/causal_mask_demo.py`](../../code/ch04_transformer/causal_mask_demo.py)  
**运行方式：** `python code/ch04_transformer/causal_mask_demo.py`

---

## 本节小结

Transformer 的完整架构之所以成功，不是因为它只有 attention，而是因为它把多个关键设计拼成了一个可训练、可扩展的系统：
- 注意力负责 token 间的信息交换
- FFN 负责单个 token 的非线性加工
- 残差连接和 LayerNorm 负责深层训练稳定
- 多层堆叠负责逐步形成更抽象的表示
- decoder-only 则把这些能力收束到自回归生成上

所以，现代 LLM 不是“在 Transformer 上多叠几层”这么简单，而是把架构、目标函数和推理方式一起统一起来。

---

**下一章：** [第5章：LLM基础](../05_llm_basics/README.md)
