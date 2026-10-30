# 扩展：线性系统理论

**难度：** 高级  
**前置知识：** 线性代数、微积分、信号处理基础（第1章）

---

## 为什么需要理解线性系统

在信号处理和控制中：

- **系统稳定性**决定了滤波器是否可用
- **频率响应**决定了系统对不同频率的处理
- **状态空间表示**是现代控制理论的基础
- **RNN 和 LSTM** 可以看作非线性动态系统

---

## 线性时不变系统（LTI）

### 定义

**线性系统**：满足叠加原理
$$T(a x_1(t) + b x_2(t)) = a T(x_1(t)) + b T(x_2(t))$$

**时不变系统**：系统特性不随时间变化
$$y(t - \tau) = T(x(t - \tau))$$

### 系统的表示

**差分方程（离散时间）：**
$$y[n] = \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{N} a_k y[n-k]$$

**微分方程（连续时间）：**
$$\sum_{k=0}^{N} a_k \frac{d^k y(t)}{dt^k} = \sum_{k=0}^{M} b_k \frac{d^k x(t)}{dt^k}$$

---

## 冲激响应和卷积

### 冲激响应

**定义**：系统对单位冲激的响应
$$h[n] = T(\delta[n])$$

其中 $\delta[n]$ 是 Kronecker delta 函数。

### 系统的输出

对于 LTI 系统，任意输入的输出可以表示为：
$$y[n] = \sum_{k=-\infty}^{\infty} x[k] h[n-k] = (x*h)[n]$$

**直观理解**：输出是输入与冲激响应的卷积。

### 因果性

**因果系统**：输出只依赖于当前和过去的输入
$$h[n] = 0, \quad n < 0$$

---

## 频率响应

### 频率响应函数

**定义**：系统对复指数输入 $e^{j\omega n}$ 的响应
$$H(e^{j\omega}) = \sum_{n=-\infty}^{\infty} h[n] e^{-j\omega n}$$

**性质：**
- 幅度响应：$|H(e^{j\omega})|$
- 相位响应：$\angle H(e^{j\omega})$

### 频率响应与冲激响应的关系

$$H(e^{j\omega}) = \text{DTFT}(h[n])$$

**应用**：通过频率响应设计滤波器。

---

## 系统稳定性

### BIBO 稳定性

**定义**：任意一致有界输入都产生一致有界输出
$$\sup_n|x[n]|<\infty\Rightarrow\sup_n|y[n]|<\infty$$

### 稳定性条件

**充要条件**：冲激响应绝对可和
$$\sum_{n=-\infty}^{\infty} |h[n]| < \infty$$

### 极点和零点

**传递函数：**
$$H(z) = \frac{B(z)}{A(z)} = \frac{\sum_{k=0}^{M} b_k z^{-k}}{1 + \sum_{k=1}^{N} a_k z^{-k}}$$

**稳定性条件**：对因果有理离散时间 LTI 系统，BIBO 稳定等价于所有极点严格位于单位圆内
$$|p_i| < 1$$

---

## 状态空间表示

### 状态空间模型

**离散时间：**
$$\mathbf{x}[n+1] = \mathbf{A}\mathbf{x}[n] + \mathbf{B}\mathbf{u}[n]$$
$$\mathbf{y}[n] = \mathbf{C}\mathbf{x}[n] + \mathbf{D}\mathbf{u}[n]$$

**连续时间：**
$$\dot{\mathbf{x}}(t) = \mathbf{A}\mathbf{x}(t) + \mathbf{B}\mathbf{u}(t)$$
$$\mathbf{y}(t) = \mathbf{C}\mathbf{x}(t) + \mathbf{D}\mathbf{u}(t)$$

其中：
- $\mathbf{x}$：状态向量
- $\mathbf{u}$：输入向量
- $\mathbf{y}$：输出向量
- $\mathbf{A}, \mathbf{B}, \mathbf{C}, \mathbf{D}$：系统矩阵

### 从差分方程到状态空间

**例子**：$y[n] = 0.5 y[n-1] + x[n]$

**状态空间表示：**
$$\mathbf{x}[n+1] = \begin{bmatrix} 0.5 \end{bmatrix} \mathbf{x}[n] + \begin{bmatrix} 1 \end{bmatrix} u[n]$$
$$y[n] = \begin{bmatrix} 1 \end{bmatrix} \mathbf{x}[n]$$

### 传递函数与状态空间的关系

$$H(z) = \mathbf{C}(z\mathbf{I} - \mathbf{A})^{-1}\mathbf{B} + \mathbf{D}$$

---

## 系统的可控性和可观测性

### 可控性

**定义**：能否通过输入将系统从任意初始状态转移到任意目标状态。

**可控性矩阵：**
$$\mathcal{C} = [\mathbf{B}, \mathbf{A}\mathbf{B}, \mathbf{A}^2\mathbf{B}, \ldots, \mathbf{A}^{n-1}\mathbf{B}]$$

**条件**：$\text{rank}(\mathcal{C}) = n$（状态维数）

### 可观测性

**定义**：能否从输出推断出系统的初始状态。

**可观测性矩阵：**
$$\mathcal{O} = \begin{bmatrix} \mathbf{C} \\ \mathbf{C}\mathbf{A} \\ \mathbf{C}\mathbf{A}^2 \\ \vdots \\ \mathbf{C}\mathbf{A}^{n-1} \end{bmatrix}$$

**条件**：$\text{rank}(\mathcal{O}) = n$

---

## 系统的特征值和稳定性

### 特征值与稳定性

**离散时间系统稳定**当且仅当：
$$|\lambda_i| < 1, \quad \forall i$$

其中 $\lambda_i$ 是 $\mathbf{A}$ 的特征值。

**连续时间系统稳定**当且仅当：
$$\text{Re}(\lambda_i) < 0, \quad \forall i$$

### 特征值与系统响应

**特征值决定了系统的动态特性：**
- 实特征值：指数衰减或增长
- 复特征值对：振荡衰减或增长

---

## 从线性系统到反馈控制

线性系统理论不仅用于分析信号和滤波器，也可以作为理解反馈控制的入口。本节只保留和后续 LLM 系统设计有关的直觉，不展开完整控制论课程。控制系统关心的不只是“输入经过系统后输出什么”，还关心“能否根据输出误差调整输入，让系统接近目标状态”。

### 开环与闭环

**开环系统**只按预设输入运行，不根据输出结果修正行为：

```text
目标 → 控制器 → 系统 → 输出
```

**闭环系统**会把输出反馈回来，与目标比较后继续调整：

```text
目标 → 控制器 → 系统 → 输出
 ↑                    ↓
 └────── 误差反馈 ←────┘
```

闭环控制的核心是误差：

$$e[n] = r[n] - y[n]$$

其中 $r[n]$ 是目标，$y[n]$ 是输出，$e[n]$ 是当前偏差。控制器根据误差产生控制输入：

$$u[n] = K(e[n])$$

最简单的比例控制可以写成：

$$u[n] = k_p e[n]$$

比例增益 $k_p$ 太小，系统响应慢；太大，系统可能振荡甚至不稳定。这就是为什么稳定性分析是控制系统的核心。

### 反馈的价值和代价

反馈的价值：

- 能修正扰动和模型误差
- 能适应环境变化
- 能把目标、观测和行动连接成闭环

反馈的代价：

- 会引入延迟
- 可能放大噪声
- 控制增益不合适时会造成震荡
- 系统必须满足稳定性和可观测性要求

### 与 Agent 的连接

第 6 章 Agent 的 ReAct 模式可以看作一种离散反馈闭环：

```text
Observe → Reason → Act → Observe → ...
```

它和传统控制系统不是同一个数学对象，但共享同一个工程思想：系统不能只生成一次动作，还要根据反馈修正后续行动。区别在于，Agent 的反馈来自工具调用、外部 API 或用户环境，噪声更大，延迟更高，错误也更难形式化建模。

因此，反馈控制在本教程中的作用不是替代 LLM 应用设计，而是提供一个辅助判断框架：只要系统会根据外部反馈持续调整行为，就需要考虑稳定性、延迟、噪声、可观测性和失败回退。

---

## 滤波器设计

### IIR 滤波器

**无限脉冲响应**：冲激响应无限长

**优势：**
- 用较少的系数实现陡峭的频率响应
- 计算量小

**劣势：**
- 可能不稳定
- 相位响应非线性

### FIR 滤波器

**有限脉冲响应**：冲激响应有限长

**优势：**
- 系数有限时冲激响应绝对可和，因此满足 BIBO 稳定
- 可以设计线性相位

**劣势：**
- 需要更多系数
- 计算量大

### 滤波器设计方法

1. **模拟滤波器设计**（Butterworth、Chebyshev）
2. **双线性变换**：将模拟滤波器转换为数字滤波器
3. **窗函数法**：设计 FIR 滤波器
4. **最优设计**：Remez 算法

---

## 系统识别

### 参数估计

**问题**：给定输入输出数据，估计系统参数。

**最小二乘法：**
$$\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta}} \sum_{n=1}^{N} (y[n] - \hat{y}[n|\boldsymbol{\theta}])^2$$

### 递归最小二乘（RLS）

**优势：**
- 在线学习
- 适应时变系统

**算法：**
$$\hat{\boldsymbol{\theta}}[n] = \hat{\boldsymbol{\theta}}[n-1] + \mathbf{K}[n] (y[n] - \hat{y}[n])$$

其中 $\mathbf{K}[n]$ 是 Kalman 增益。

---

## 与深度学习的连接

### RNN 作为动态系统

**RNN 的隐状态更新：**
$$\mathbf{h}[n] = \sigma(\mathbf{W}_h \mathbf{h}[n-1] + \mathbf{W}_x \mathbf{x}[n])$$

**类似于非线性状态空间模型：**
$$\mathbf{x}[n+1] = f(\mathbf{A}\mathbf{x}[n] + \mathbf{B}\mathbf{u}[n])$$

### LSTM 和门控机制

**遗忘门、输入门、输出门**可以看作系统的**可控性和可观测性**机制。

### 梯度消失问题

**根本原因**：RNN 的特征值小于 1，导致梯度指数衰减。

**解决方案**：
- 残差连接（类似于系统的稳定性）
- 门控机制（类似于系统的可控性）

---

## 实践建议

### 系统分析步骤

1. **获取系统模型**：差分方程或状态空间
2. **计算冲激响应**：理解系统的时域特性
3. **计算频率响应**：理解系统的频域特性
4. **检查稳定性**：极点位置或特征值
5. **设计控制器**（如需要），并检查反馈是否稳定

### Python 实现

```python
import numpy as np
from scipy import signal

# 定义系统：H(z) = 1 / (1 - 0.5*z^-1)
b = [1]
a = [1, -0.5]

# 计算冲激响应
h = signal.impulse((b, a, 1))[1]

# 计算频率响应
w, H = signal.freqz(b, a)

# 检查稳定性
poles = np.roots(a)
print(f"极点：{poles}")
print(f"稳定：{np.all(np.abs(poles) < 1)}")
```

---

## 关键要点

1. **LTI 系统可以用冲激响应或频率响应完全描述**
2. **稳定性由极点位置决定**
3. **状态空间表示是现代控制理论的基础**
4. **可控性和可观测性决定了系统的可设计性**
5. **反馈控制把目标、观测和行动连接成闭环，但必须考虑稳定性、延迟和噪声**
6. **RNN 可以看作非线性动态系统**

---

## 进一步阅读

- Oppenheim & Schafer, "Discrete-Time Signal Processing"
- Kailath, "Linear Systems"
- Ljung, "System Identification: Theory for the User"
