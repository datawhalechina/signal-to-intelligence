<!-- AUTO-GENERATED from README.md. Do not edit index.md directly. -->

# 第1章：传统数字信号处理

## 章节概览

本章介绍传统数字信号处理（DSP）的核心概念，为后续章节提供统一的数学底座。这里的重点不是把所有经典算法都铺开，而是建立一条清晰的主线：**信号表示 → 频域分析 → 滤波/卷积 → 随机建模 → 检测/估计 → 子空间方法**。

这条主线既适用于通信接收机，也适用于图像、音频和时间序列；后续章节中的卷积、位置编码、序列建模和表示学习，都能在这一章里找到对应的数学原型。

![从 DSP 到 Transformer 的信息处理桥接](/diagrams/ch01-03-information-processing-bridge.svg)

*章节主图：从 DSP 到 Transformer 的信息处理桥接。后续模型没有放弃表示、聚合和抑制无关信息这些基本问题，而是在扩大它们可学习、可适应的范围。*

## 本章与大模型的思想桥接

本章不是把 DSP 当作大模型的直接实现方式，而是用它建立一组可迁移的问题意识。信息处理系统通常都在做几件事：**选择表示、提取结构、抑制无关成分、从观测中估计未知量，并在不确定性下作出判断**。Transformer 和 LLM 换了数据形态与参数化方式，但仍然在处理这些问题。

| DSP 中的问题 | 后续章节中的对应问题 | 需要保留的判断 |
|---|---|---|
| 如何表示信号 | Token、Embedding、隐藏状态如何表示文本 | 表示决定后续可学习的结构 |
| 如何提取结构 | 卷积、注意力如何聚合局部或全局信息 | 聚合范围影响表达能力与计算成本 |
| 如何抑制无关成分 | 滤波、Mask、检索和路由 | “看什么”与“忽略什么”同样重要 |
| 如何从观测中估计参数 | 训练、微调和概率预测 | 估计结果依赖目标、数据和噪声 |
| 如何在不确定性下判决 | 分类、生成、评估和拒答 | 阈值、代价和错误类型必须一起考虑 |

因此，后续看到“注意力”“Embedding”或“生成”时，可以先追问三个问题：它表示了什么？它选择性地聚合了什么？它如何衡量输出是否可靠？这些问题比寻找一一对应的算法名称更重要。

## 小节目录

### 基础概念（1.1-1.4）

**1.1 信号的三种视角** — [📖 阅读](01_signals.md)
- 时域、频域、时频，理解信号的多个角度

**1.2 傅里叶变换与频谱分析** — [📖 阅读](02_fourier.md)
- 从时域到频域的转换，位置编码的数学基础

**1.3 滤波器与卷积** — [📖 阅读](03_filters.md)
- 信号处理的基本操作，CNN 卷积核的原型

**1.4 时频分析** — [📖 阅读](04_time_freq.md)
- 同时看到时间和频率信息，STFT 与小波变换

### 理论与应用（1.5-1.8）

**1.5 随机信号理论** — [📖 阅读](05_random_signals.md)
- 统计特性、平稳性、高斯过程、噪声模型

**1.6 信号检测** — [📖 阅读](06_signal_detection.md)
- 假设检验、似然比检验、ROC曲线、检测器设计

**1.7 信号估计** — [📖 阅读](07_signal_estimation.md)
- 参数估计、MLE、LSE、贝叶斯估计、Cramér-Rao界

**1.8 矩阵分解应用** — [📖 阅读](08_matrix_decomposition.md)
- SVD、EVD、子空间方法、MUSIC、PCA、去噪

## 核心问题

完成本章后，你应该能回答：

**基础部分（1.1-1.4）：**
1. 信号可以从哪些角度理解？
2. 傅里叶变换的本质是什么？
3. 卷积如何实现滤波？
4. CNN中的卷积核与DSP中的滤波器有什么关系？

**进阶部分（1.5-1.8）：**
5. 什么是随机信号？如何分析其统计特性？
6. 如何在噪声中检测信号？什么是ROC曲线？
7. 如何从观测中估计信号参数？什么是Cramér-Rao界？
8. 矩阵分解如何应用于信号处理？什么是MUSIC算法？

## 本章学习闭环

1. 先用章节主图定位“表示、聚合、抑制无关成分、判决”四个问题；
2. 运行 `fft_spectrum.py` 与 `signal_detection.py`，分别观察表示变化和检测结果；
3. 完成最低实践任务，解释噪声或滤波参数为何同时改变频谱与判决；
4. 需要进一步比较方法与失败案例时，进入 `signal_detection` 项目。

## 最低实践任务

生成一个包含两个频率成分和噪声的信号，分别观察时域与频域表示；再调整噪声强度或滤波器参数，记录频谱峰值和检测结果如何变化。完成标准：能用自己的话解释一个参数变化为什么会同时影响“看见什么”和“判断是否存在”。

完成后可进入章节项目：[信号检测与估计](../../projects/signal_detection/README.md)。它在固定虚警率下比较能量检测与匹配滤波，并通过模板失配案例讨论“更强先验为何既能提高灵敏度，也会缩小适用范围”。

## 代码实验

本章共有 **12 个代码脚本**，生成 **16 张图片**，覆盖所有 8 个小节。

| 小节 | 脚本 | 生成图片 | 文档位置 |
|------|------|---------|---------|
| 1.1 信号的三种视角 | [`signal_dimensions.py`](../../code/ch01_dsp/signal_dimensions.py) | `ch01_signal_dimensions.png` | [1.1](01_signals.md) |
| 1.1 信号的三种视角 | [`signal_three_views.py`](../../code/ch01_dsp/signal_three_views.py) | `ch01_three_views.png` | [1.1](01_signals.md) |
| 1.2 傅里叶变换 | [`fft_spectrum.py`](../../code/ch01_dsp/fft_spectrum.py) | `ch01_fft_spectrum.png` | [README](README.md) |
| 1.2 傅里叶变换 | [`fourier_2d.py`](../../code/ch01_dsp/fourier_2d.py) | `ch01_fourier_2d.png` | [1.2](02_fourier.md) |
| 1.2 傅里叶变换与位置编码 | [`positional_encoding.py`](../../code/ch01_dsp/positional_encoding.py) | `ch01_positional_encoding.png` | [1.2](02_fourier.md) |
| 1.3 滤波器与卷积 | [`random_signals.py`](../../code/ch01_dsp/random_signals.py) | `ch01_convolution_effect.png` | [1.3](03_filters.md) |
| 1.4 时频分析 | [`time_freq_analysis.py`](../../code/ch01_dsp/time_freq_analysis.py) | `ch01_time_freq_music.png` | [1.4](04_time_freq.md) |
| 1.4 时频分析 | [`time_freq_analysis.py`](../../code/ch01_dsp/time_freq_analysis.py) | `ch01_time_freq_comparison.png` | [1.4](04_time_freq.md) |
| 1.5 随机信号 | [`random_signals.py`](../../code/ch01_dsp/random_signals.py) | `ch01_random_signals.png` | [1.5](05_random_signals.md) |
| 1.6 信号检测 | [`signal_detection.py`](../../code/ch01_dsp/signal_detection.py) | `ch01_detection_performance.png` | [1.6](06_signal_detection.md) |
| 1.6 信号检测 | [`signal_detection.py`](../../code/ch01_dsp/signal_detection.py) | `ch01_signal_detection.png` | [1.6](06_signal_detection.md) |
| 1.7 信号估计 | [`parameter_estimation.py`](../../code/ch01_dsp/parameter_estimation.py) | `ch01_parameter_estimation.png` | [1.7](07_signal_estimation.md) |
| 1.8 矩阵分解 | [`music_algorithm.py`](../../code/ch01_dsp/music_algorithm.py) | `ch01_matrix_decomposition.png` | [1.8](08_matrix_decomposition.md) |
| 1.8 矩阵分解 | [`music_algorithm.py`](../../code/ch01_dsp/music_algorithm.py) | `ch01_matrix_decomposition_performance.png` | [1.8](08_matrix_decomposition.md) |
| 1.8 矩阵分解 | [`feature_extraction_comparison.py`](../../code/ch01_dsp/feature_extraction_comparison.py) | `ch01_feature_extraction.png` | [1.8](08_matrix_decomposition.md) |
| 扩展：随机过程 | [`stochastic_processes_demo.py`](../../code/ch01_dsp/stochastic_processes_demo.py) | `ch01_stochastic_processes.png` | [extensions](extensions/stochastic_processes.md) |

**运行所有实验：**
```bash
python code/ch01_dsp/fft_spectrum.py
python code/ch01_dsp/signal_three_views.py
python code/ch01_dsp/time_freq_analysis.py
# ... 其余脚本见上表
```

![FFT Spectrum Analysis](/assets/ch01_fft_spectrum.png)

*图1.1：FFT频谱分析。展示信号在频域的表示，不同频率分量的幅度和相位。*

**代码文件：** [`code/ch01_dsp/fft_spectrum.py`](../../code/ch01_dsp/fft_spectrum.py)
**运行方式：** `python code/ch01_dsp/fft_spectrum.py`

## 扩展内容

### 高级DSP话题 — [📖 阅读](extensions/advanced_topics.md)
- 窗函数详解（Hann、Hamming、Blackman 的频谱特性）
- 多速率信号处理（上采样、下采样、滤波器组）
- 滤波器设计（Butterworth、Chebyshev、椭圆滤波器）
- 与深度学习的连接（卷积、注意力、位置编码）

### 线性系统理论 — [📖 阅读](extensions/linear_systems.md)
- 线性时不变系统（LTI）：冲激响应与卷积
- 频率响应与系统稳定性分析
- 状态空间表示、可控性与可观测性
- 与深度学习的连接（RNN 状态方程、CNN 滤波器）

### 随机过程进阶 — [📖 阅读](extensions/stochastic_processes.md)
- 平稳过程、马尔可夫链、高斯过程
- AR/MA/ARMA 模型与谱分析
- 随机过程的滤波（Wiener 滤波、Kalman 滤波）
- 与 LLM 的连接（语言模型作为序列随机过程）

## 关键连接点

### DSP → CNN

```
DSP中的卷积：y[n] = Σ x[m] * h[n-m]
                    ↓
CNN中的卷积核：可学习的滤波器
```

**启示：** CNN可以看作是从数据中学习滤波器参数，而不是手工设计滤波器。

### 傅里叶变换 → 位置编码

```
傅里叶变换：用不同频率的正弦波表示信号
                    ↓
位置编码：用不同频率的正弦波表示位置
```

**启示：** 位置编码把顺序信息编码进向量表示，使 Transformer 能区分不同位置上的 token。

---

**下一步：** 阅读 [1.1 信号的三种视角](01_signals.md)
