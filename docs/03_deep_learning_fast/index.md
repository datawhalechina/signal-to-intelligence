<!-- AUTO-GENERATED from README.md. Do not edit index.md directly. -->

# 第3章：深度学习快速通道

## 章节概览

本章不打算罗列模型名，而是用一条主线说明深度学习为什么会成为“表示学习”的核心框架：模型如何从数据中自动提取特征、如何把视觉任务从分类推进到检测和分割、以及如何把一维序列建模推进到 Transformer。

![从 DSP 到 Transformer 的信息处理桥接](/diagrams/ch01-03-information-processing-bridge.svg)

*章节主图：从手工变换到可学习局部特征、递推状态和全局注意力的演化。它说明第 3 章不是孤立的模型速览，而是进入 Transformer 的过渡层。*

## 本章的过渡作用：从特征工程到动态信息处理

第 1 章中的滤波器和第 2 章中的优化，分别回答了“如何提取结构”和“如何学习参数”。本章进一步说明：模型不必完全依赖人工设计的特征，而可以通过多层变换学习任务相关的表示。

```text
DSP：手工设计滤波器，提取已知结构
  ↓
CNN：学习局部滤波器，提取空间模式
  ↓
RNN：用状态递推保留序列历史
  ↓
Transformer：按内容动态选择全局信息
```

这不是“旧算法被新算法替代”的简单故事。每一步都在改变信息处理的范围、表示方式和计算代价：CNN 偏向局部平移结构，RNN 偏向顺序状态，Transformer 则把关系建模变成可并行的动态加权聚合。第 4 章会把这种聚合机制写成注意力公式，第 5 章再说明它如何通过预测目标学习语言表示。

这一章既是对前面DSP、优化、传统机器学习内容的延伸，也是进入Transformer和LLM之前的重要桥梁。

本章重点关注：
- 为什么浅层模型会在复杂任务上遇到表达瓶颈
- CNN 如何把卷积滤波器变成可学习的特征提取器
- 目标检测和图像分割如何把视觉任务粒度继续推深
- RNN、1D CNN 和 Transformer 在序列建模上分别解决什么问题
- 第 3 章如何自然过渡到第 4 章的 Transformer 主线

## 小节目录

### 基础与视觉（3.1-3.4）

**3.1 为什么需要深度学习** — [📖 阅读](01_why_deep_learning.md)
- 从手工特征到自动表示学习，浅层模型的表达瓶颈

**3.2 CNN的本质** — [📖 阅读](02_cnn_essence.md)
- 卷积神经网络与 DSP 滤波器的联系，可学习的局部特征提取

**3.3 从分类到检测：YOLO的核心思想** — [📖 阅读](03_yolo_detection.md)
- CNN 如何同时回答“有什么”和“在哪里”

**3.4 从检测到像素级理解：图像分割** — [📖 阅读](04_image_segmentation.md)
- 从边界框走向像素级预测与空间恢复

### 序列建模（3.5-3.6）

**3.5 序列建模与一维信号处理** — [📖 阅读](05_sequence_models_and_1d_signals.md)
- RNN、1D CNN 和状态空间模型如何处理时序信号

**3.6 为什么Transformer适合大规模序列建模** — [📖 阅读](06_why_transformer_better.md)
- Transformer 相比 RNN 的优势，以及它为什么会成为 LLM 的前置主线

## 核心问题

完成本章后，你应该能回答：

### 原理理解部分
1. 为什么浅层模型在复杂任务上会遇到表达瓶颈？
2. CNN 中的卷积核为什么可以看作可学习的滤波器？
3. 为什么目标检测和图像分割都比分类更难？
4. RNN、1D CNN 和 Transformer 分别适合什么样的序列问题？
5. 为什么 RNN 在长序列上容易遇到训练和并行化瓶颈？
6. 为什么第 3 章的内容会自然通向 Transformer 和 LLM？

### 应用实践部分
7. 如何把手工特征工程切换成表示学习？
8. CNN 如何从分类扩展到检测与分割？
9. 如何用深度学习处理音频、传感器和时间序列？
10. 什么时候该从第 3 章过渡到第 4 章的 Transformer 视角？

## 本章学习闭环

1. 用章节主图比较手工变换、局部可学习特征、递推状态和全局注意力的处理范围；
2. 运行 `detection_segmentation_demo.py` 或 `sequence_models_1d_signal.py`，观察任务输出和建模假设；
3. 完成最低实践任务，说明模型结构改变了哪一种依赖关系；
4. 需要进入真实检测比较时，查看 [视觉目标检测方法对比项目](../../projects/vision_detection_comparison/README.md) 的统一评估口径。

## 最低实践任务

选择一个同时包含局部模式和长距离依赖的合成序列，运行或修改 MLP、1D CNN、RNN 的实验，比较它们保留信息的方式。完成标准：画出至少一组对比结果，并说明为什么局部卷积、递推状态和全局注意力适合不同的依赖结构。

## 代码实验

本章共有 **5 个代码脚本**，生成 **6 张图片**，覆盖深度学习从表示学习到视觉、序列建模的核心思想。

| 小节 | 脚本 | 生成图片 | 文档位置 |
|------|------|---------|---------|
| 3.1 为什么需要深度学习 | [`polynomial_vs_mlp.py`](../../code/ch03_deep_learning_fast/polynomial_vs_mlp.py) | `ch03_polynomial_vs_mlp.png` | [3.1](01_why_deep_learning.md) / [README](README.md) |
| 3.2 CNN的本质 | [`mnist_cnn.py`](../../code/ch03_deep_learning_fast/mnist_cnn.py) | `ch03_mnist_cnn.png` | [3.2](02_cnn_essence.md) / [README](README.md) |
| 3.3/3.4 检测与分割 | [`detection_segmentation_demo.py`](../../code/ch03_deep_learning_fast/detection_segmentation_demo.py) | `ch03_yolo_vs_segmentation.png` `ch03_segmentation_mask_overlay.png` | [3.3](03_yolo_detection.md) / [3.4](04_image_segmentation.md) |
| 3.5 序列建模 | [`rnn_structure.py`](../../code/ch03_deep_learning_fast/rnn_structure.py) | `ch03_rnn_structure.png` | [3.5](05_sequence_models_and_1d_signals.md) / [README](README.md) |
| 3.5 序列建模 | [`sequence_models_1d_signal.py`](../../code/ch03_deep_learning_fast/sequence_models_1d_signal.py) | `ch03_sequence_models.png` | [3.5](05_sequence_models_and_1d_signals.md) |

### 实验1：多项式拟合 vs MLP
- **文件：** [`code/ch03_deep_learning_fast/polynomial_vs_mlp.py`](../../code/ch03_deep_learning_fast/polynomial_vs_mlp.py)
- **内容：** 对比传统多项式拟合和神经网络的表达能力
- **运行：** `python code/ch03_deep_learning_fast/polynomial_vs_mlp.py`
- **输出：** 拟合曲线对比、误差分析、表达能力对比

![Polynomial vs MLP](/assets/ch03_polynomial_vs_mlp.png)

*图3.1：多项式拟合与MLP的对比。展示深度学习相比传统方法的表达能力优势。*

**代码文件：** [`code/ch03_deep_learning_fast/polynomial_vs_mlp.py`](../../code/ch03_deep_learning_fast/polynomial_vs_mlp.py)
**运行方式：** `python code/ch03_deep_learning_fast/polynomial_vs_mlp.py`

### 实验2：CNN结构与特征提取
- **文件：** [`code/ch03_deep_learning_fast/mnist_cnn.py`](../../code/ch03_deep_learning_fast/mnist_cnn.py)
- **内容：** 展示CNN如何从图像中逐层提取特征并完成分类
- **运行：** `python code/ch03_deep_learning_fast/mnist_cnn.py`
- **输出：** CNN结构图、分类结果、特征提取过程

![MNIST CNN](/assets/ch03_mnist_cnn.png)

*图3.2：CNN在手写数字识别中的结构示意。展示卷积层如何逐层提取视觉特征。*

**代码文件：** [`code/ch03_deep_learning_fast/mnist_cnn.py`](../../code/ch03_deep_learning_fast/mnist_cnn.py)
**运行方式：** `python code/ch03_deep_learning_fast/mnist_cnn.py`

### 实验3：RNN结构与序列处理
- **文件：** [`code/ch03_deep_learning_fast/rnn_structure.py`](../../code/ch03_deep_learning_fast/rnn_structure.py)
- **内容：** 展示RNN如何处理序列数据以及隐状态如何随时间演化
- **运行：** `python code/ch03_deep_learning_fast/rnn_structure.py`
- **输出：** RNN结构图、隐状态演化、序列处理过程

![RNN Structure](/assets/ch03_rnn_structure.png)

*图3.5：RNN的结构与序列处理。展示循环连接如何使模型能处理可变长度序列。*

**代码文件：** [`code/ch03_deep_learning_fast/rnn_structure.py`](../../code/ch03_deep_learning_fast/rnn_structure.py)
**运行方式：** `python code/ch03_deep_learning_fast/rnn_structure.py`

### 实验4：目标检测与图像分割演示
- **文件：** [`code/ch03_deep_learning_fast/detection_segmentation_demo.py`](../../code/ch03_deep_learning_fast/detection_segmentation_demo.py)
- **内容：** 对比目标检测与图像分割的任务形式和输出差异
- **运行：** `python code/ch03_deep_learning_fast/detection_segmentation_demo.py`
- **输出：** 检测框示意、分割掩码示意、任务对比图

![YOLO vs Segmentation](/assets/ch03_yolo_vs_segmentation.png)

*图3.4：目标检测（边界框）与图像分割（像素级mask）的输出形式对比。*

**代码文件：** [`code/ch03_deep_learning_fast/detection_segmentation_demo.py`](../../code/ch03_deep_learning_fast/detection_segmentation_demo.py)
**运行方式：** `python code/ch03_deep_learning_fast/detection_segmentation_demo.py`

### 实验5：序列模型与一维信号处理
- **文件：** [`code/ch03_deep_learning_fast/sequence_models_1d_signal.py`](../../code/ch03_deep_learning_fast/sequence_models_1d_signal.py)
- **内容：** 对比1D CNN、RNN与Transformer风格模型在时序信号上的建模方式
- **运行：** `python code/ch03_deep_learning_fast/sequence_models_1d_signal.py`
- **输出：** 时序信号建模对比、局部模式/长期依赖分析、结构示意图

## 扩展内容

### 深度学习理论基础 — [📖 阅读](extensions/deep_learning_theory.md)
- 通用逼近定理：神经网络为什么能拟合任意函数
- 偏差-方差权衡：过拟合与欠拟合的数学分析
- 表示学习：深度网络如何逐层抽象特征
- CNN/RNN 的数学基础与 Transformer 的优势分析
- 推荐论文（LeNet、AlexNet、LSTM、Transformer、YOLO、U-Net）

## 关键连接点

### CNN → DSP

```
DSP中的卷积：y[n] = Σ x[m] * h[n-m]
                    ↓
CNN中的卷积核：可学习的滤波器
```

**启示：** CNN就是让滤波器不再手工设计，而是通过数据自动学习。

### YOLO / 分割 → 视觉理解

```
分类：图里有什么
    ↓
检测：图里有什么 + 在哪里
    ↓
分割：每个像素属于什么
```

**启示：** 视觉模型的发展，可以理解为在不断提升空间理解的粒度。

### RNN / 1D CNN → 时序信号

```
1D CNN：局部模式检测
RNN：状态递推记忆
Transformer：直接建模长距离依赖
```

**启示：** 处理序列不只有一种方法，不同模型对应不同的依赖结构和计算权衡。

### Transformer → LLM

```
Transformer：通用序列建模架构
                    ↓
LLM：Transformer + 大规模预训练
```

**启示：** LLM并不是凭空出现的，而是深度学习、序列建模和规模化训练共同演化的结果。

---

**下一步：** 阅读 [3.1 为什么需要深度学习](01_why_deep_learning.md)
