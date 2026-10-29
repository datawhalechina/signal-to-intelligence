# 第3章扩展：深度学习理论

本文档包含第3章的深度扩展内容，适合想深入理解深度学习理论的读者。

---

## E3.1 通用逼近定理

### 定理陈述

**在紧致定义域上，很多连续函数可以用足够宽的单隐层神经网络以任意精度逼近。**

$$f(x) \approx \sum_{i=1}^{n} w_i \sigma(v_i^T x + b_i)$$

其中 $\sigma$ 是激活函数（如ReLU或sigmoid）。

### 含义

- 单隐层网络有很强的函数逼近能力
- 但需要的神经元数量可能很多
- 深度网络用更少的参数达到相同的表达能力

### 启示

**深度 vs 宽度的权衡：**
- 浅层网络：需要指数多的神经元
- 深层网络：用多层堆叠，参数数量多项式增长

---

## E3.2 偏差-方差权衡

### 定义

**总误差 = 偏差² + 方差 + 不可约误差**

$$E[(y - \hat{f}(x))^2] = \text{Bias}^2(\hat{f}) + \text{Var}(\hat{f}) + \sigma^2$$

### 偏差（Bias）

**定义：** 模型预测的期望与真实值的差距。

**含义：** 模型的表达能力不足。

**例子：** 用直线拟合曲线数据，偏差大。

### 方差（Variance）

**定义：** 模型对不同训练集的敏感性。

**含义：** 模型过拟合。

**例子：** 用高阶多项式拟合少量数据，方差大。

### 权衡

```
模型复杂度
  |     偏差
  |    /
  |   /
  |  /_____ 总误差
  | /       \
  |/         \ 方差
  |_____________
```

**最优点：** 偏差和方差的平衡。

---

## E3.3 表示学习

### 核心思想

**深度学习通过多层非线性变换学习数据的分层表示。**

```
原始数据
  ↓
第1层：低级特征（边界、纹理）
  ↓
第2层：中级特征（形状、部分）
  ↓
第3层：高级特征（物体、概念）
  ↓
输出
```

### 为什么有效

1. **自然的分层结构**
   - 现实世界的数据有分层结构
   - 深度学习自然地学习这种结构

2. **特征重用**
   - 低级特征被多个高级特征共享
   - 减少参数数量

3. **逐步抽象**
   - 每层学习一个抽象级别
   - 最后一层的特征最适合任务

### 与迁移学习的联系

**预训练 + 微调：**
1. 在大数据集上预训练，学习通用表示
2. 在小数据集上微调，适应特定任务

**优势：** 利用大数据集学到的表示，减少小数据集的过拟合。

---

## E3.4 CNN的数学基础

### 卷积的性质

**交换律：**
$$f * g = g * f$$

**结合律：**
$$(f * g) * h = f * (g * h)$$

**分配律：**
$$f * (g + h) = f * g + f * h$$

### 卷积定理

**时域卷积 = 频域乘法：**
$$\mathcal{F}(f * g) = \mathcal{F}(f) \cdot \mathcal{F}(g)$$

**应用：** 用FFT快速计算卷积。

### 感受野（Receptive Field）

**定义：** 输出神经元能"看到"的输入区域。

**计算：**
- 第1层：卷积核大小（如3×3）
- 第2层：第1层感受野 + 卷积核大小 - 1

**例子：**
```
3×3卷积 → 感受野 3×3
3×3卷积 → 感受野 5×5
3×3卷积 → 感受野 7×7
```

**启示：** 深层网络有更大的感受野，能看到更大的上下文。

---

## E3.5 RNN的数学分析

### 梯度流

**反向传播通过时间（BPTT）：**

$$\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial y_t} \frac{\partial y_t}{\partial h_t} + \frac{\partial L}{\partial h_{t+1}} \frac{\partial h_{t+1}}{\partial h_t}$$

**问题：** 梯度通过多个时刻相乘。

### 梯度消失/爆炸

**梯度范数：**
$$\left\|\frac{\partial L}{\partial h_1}\right\| = \left\|\prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}}\right\| \left\|\frac{\partial L}{\partial h_T}\right\|$$

**如果 $\left\|\frac{\partial h_t}{\partial h_{t-1}}\right\| < 1$：** 梯度消失

**如果 $\left\|\frac{\partial h_t}{\partial h_{t-1}}\right\| > 1$：** 梯度爆炸

### LSTM的解决方案

**细胞状态（Cell State）：**
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**优势：** 梯度可以直接通过细胞状态，避免消失/爆炸。

---

## E3.6 Transformer的优势分析

### 计算复杂度

**RNN：**
- 时间复杂度：$O(T \cdot d^2)$（T是序列长度）
- 无法并行化

**Transformer：**
- 时间复杂度：$O(T^2 \cdot d)$（注意力计算）
- 序列位置可以并行计算

**权衡：** 对于长序列，Transformer的并行优势弥补了二次复杂度。

### 长期依赖

**RNN：** 信息通过隐状态传递，距离为序列长度

**Transformer：** 任意两个位置之间的距离都是1（通过注意力）

**启示：** Transformer更容易学习长期依赖。

### 可扩展性

**Transformer的优势：**
1. 序列位置可以并行计算 → 可以用大批大小
2. 参数量可任意增加 → 可以扩展到大模型
3. 性能随参数量单调增长 → 缩放律

---

## E3.7 深度学习的局限

### 数据需求

**深度学习需要大量数据：**
- 浅层模型：可能只需几千个样本
- 深层模型：通常需要数百万个样本

**原因：** 参数多，容易过拟合。

### 可解释性

**深度学习是"黑盒"：**
- 难以理解模型为什么做出某个决定
- 难以调试模型的错误

**研究方向：** 可解释AI（XAI）

### 计算成本

**训练大模型很贵：**
- GPT-3：数百万美元
- 需要大量GPU/TPU

**环境影响：** 高能耗

---

## E3.8 推荐论文

### CNN的经典论文

1. **LeCun et al. (1998)** - "Gradient-Based Learning Applied to Document Recognition"
   - LeNet论文
   - CNN的开创性工作

2. **Krizhevsky et al. (2012)** - "ImageNet Classification with Deep Convolutional Neural Networks"
   - AlexNet论文
   - 深度学习复兴的标志

### RNN的经典论文

1. **Hochreiter & Schmidhuber (1997)** - "Long Short-Term Memory"
   - LSTM论文
   - 解决梯度消失问题

2. **Cho et al. (2014)** - "Learning Phrase Representations using RNN Encoder-Decoder"
   - GRU论文
   - LSTM的简化版本

### Transformer的论文

1. **Vaswani et al. (2017)** - "Attention Is All You Need"
   - Transformer论文
   - 深度学习的新时代

---

## E3.9 进一步学习

### 书籍

- **"Deep Learning" by Goodfellow, Bengio, Courville**
  - 深度学习的综合教科书
  - 理论和实践并重

- **"Neural Networks and Deep Learning" by Nielsen**
  - 在线免费书籍
  - 很好的入门资料

### 在线资源

- **Stanford CS231n** - Convolutional Neural Networks for Visual Recognition
- **MIT 6.S191** - Introduction to Deep Learning
- **Fast.ai** - Practical Deep Learning for Coders

### 实践项目

1. **实现CNN**
   - 从零开始实现卷积层
   - 在MNIST上训练

2. **实现RNN**
   - 实现基础RNN
   - 实现LSTM

3. **Transformer实验**
   - 理解自注意力机制
   - 在序列任务上训练

---

**返回：** [第3章：深度学习快速通道](../README.md)
