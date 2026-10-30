# 扩展：随机过程进阶

**难度：** 高级  
**前置知识：** 概率论、统计学、随机信号理论（第1.5章）

---

## 为什么需要理解随机过程

在信号处理和机器学习中：

- **时间序列建模**需要理解随机过程的动态特性
- **马尔可夫链**用于序列模型和强化学习
- **平稳过程**的假设简化了许多算法
- **自相关结构**影响滤波和预测的性能

---

## 随机过程的基本概念

### 定义

**随机过程**：一族随机变量 $\{X(t), t \in T\}$，其中 $t$ 是时间参数。

**分类：**
- **离散时间**：$t \in \{0, 1, 2, \ldots\}$（序列）
- **连续时间**：$t \in \mathbb{R}$（信号）
- **离散值**：$X(t) \in \{0, 1, 2, \ldots\}$
- **连续值**：$X(t) \in \mathbb{R}$

### 随机过程的描述

**一阶统计量：**
$$\mu(t) = E[X(t)]$$

**二阶统计量：**
$$R(t_1, t_2) = E[X(t_1)X(t_2)]$$

**自相关函数：**
$$\rho(t_1, t_2) = \frac{R(t_1, t_2) - \mu(t_1)\mu(t_2)}{\sigma(t_1)\sigma(t_2)}$$

---

## 平稳过程

### 严格平稳性

**定义**：如果对任意 $\tau$ 和任意 $n$，有：

$$P(X(t_1) \leq x_1, \ldots, X(t_n) \leq x_n) = P(X(t_1+\tau) \leq x_1, \ldots, X(t_n+\tau) \leq x_n)$$

**直观理解**：过程的统计性质不随时间变化。

### 宽平稳性（弱平稳性）

**定义**：如果满足以下条件：

1. $E[X(t)] = \mu$（常数）
2. $E[X(t)^2] < \infty$
3. $R(t_1, t_2) = R(t_1 - t_2) = R(\tau)$（只依赖于时间差）

**优势**：比严格平稳性更容易验证。

### 自相关函数（ACF）

对于宽平稳过程：

$$\rho(\tau) = \frac{R(\tau) - \mu^2}{\sigma^2}$$

**性质：**
- $\rho(0) = 1$
- $\rho(\tau) = \rho(-\tau)$（对称性）
- $|\rho(\tau)| \leq 1$

---

## 马尔可夫链

### 定义

**马尔可夫性质**：未来只依赖于现在，不依赖于过去：

$$P(X_{n+1} = x | X_n = x_n, X_{n-1} = x_{n-1}, \ldots) = P(X_{n+1} = x | X_n = x_n)$$

**一阶马尔可夫链**：只依赖于前一个状态。

### 转移矩阵

**定义**：转移概率矩阵 $\mathbf{P}$：

$$P_{ij} = P(X_{n+1} = j | X_n = i)$$

**性质：**
- 每行和为 1：$\sum_j P_{ij} = 1$
- 所有元素非负：$P_{ij} \geq 0$

### 例子：天气模型

```
状态：晴天、阴天、下雨

转移矩阵：
       晴  阴  雨
晴  [ 0.7 0.2 0.1 ]
阴  [ 0.3 0.4 0.3 ]
雨  [ 0.2 0.3 0.5 ]
```

### 平稳分布

**定义**：如果 $\boldsymbol{\pi}$ 满足：

$$\boldsymbol{\pi} = \boldsymbol{\pi} \mathbf{P}$$

则 $\boldsymbol{\pi}$ 是平稳分布。

**直观理解**：长期运行后，系统的状态分布不再变化。

---

## 高斯过程

### 定义

**高斯过程**：任意有限个时刻的联合分布都是高斯分布。

**完全由以下两个函数确定：**
1. 均值函数：$\mu(t) = E[X(t)]$
2. 协方差函数：$K(t_1, t_2) = \text{Cov}(X(t_1), X(t_2))$

### 高斯过程回归

**模型：**
$$y = f(x) + \epsilon, \quad f \sim \text{GP}(\mu, K)$$

**优势：**
- 提供不确定性估计
- 贝叶斯方法
- 灵活的核函数

**应用：**
- 时间序列预测
- 超参数优化（贝叶斯优化）
- 空间插值

---

## 自回归模型（AR）

### AR(p) 模型

**定义：**
$$X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdots + \phi_p X_{t-p} + \epsilon_t$$

其中 $\epsilon_t$ 是白噪声。

### 特征方程

**特征多项式：**
$$1 - \phi_1 z - \phi_2 z^2 - \cdots - \phi_p z^p = 0$$

**平稳性条件**：所有根的模都大于 1。

### 自相关函数

对于 AR(1) 模型 $X_t = \phi X_{t-1} + \epsilon_t$：

$$\rho(\tau) = \phi^{|\tau|}$$

---

## 移动平均模型（MA）

### MA(q) 模型

**定义：**
$$X_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \cdots + \theta_q \epsilon_{t-q}$$

### 自相关函数

MA(q) 模型的 ACF 在 $\tau > q$ 时为 0（截断性）。

---

## ARMA 模型

### ARMA(p,q) 模型

**定义：**
$$X_t = \phi_1 X_{t-1} + \cdots + \phi_p X_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + \cdots + \theta_q \epsilon_{t-q}$$

**优势：**
- 结合 AR 和 MA 的优点
- 用较少的参数描述复杂的相关结构

### 模型选择

使用 ACF 和 PACF（偏自相关函数）选择 $p$ 和 $q$：

| 模型 | ACF | PACF |
|------|-----|------|
| AR(p) | 指数衰减 | 在 p 后截断 |
| MA(q) | 在 q 后截断 | 指数衰减 |
| ARMA(p,q) | 指数衰减 | 指数衰减 |

---

## 谱分析

### 功率谱密度（PSD）

对于平稳过程，PSD 是自相关函数的傅里叶变换：

$$S(f) = \sum_{\tau=-\infty}^{\infty} R(\tau) e^{-j2\pi f\tau}$$

### Wiener-Khinchin 定理

**定理**：平稳过程的功率谱密度等于自相关函数的傅里叶变换。

**应用：**
- 频域滤波
- 信号检测
- 系统识别

---

## 随机过程的滤波

### Wiener 滤波

**问题**：给定观测 $y(t) = s(t) + n(t)$，估计信号 $s(t)$。

在信号与噪声不相关、目标是最小均方误差等假设下，Wiener 滤波器为：
$$H(f) = \frac{S_s(f)}{S_s(f) + S_n(f)}$$

其中 $S_s(f)$ 和 $S_n(f)$ 分别是信号和噪声的 PSD。

### Kalman 滤波

**状态空间模型：**
$$\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t + \mathbf{w}_t$$
$$\mathbf{y}_t = \mathbf{C}\mathbf{x}_t + \mathbf{v}_t$$

**特点：**
- 处理非平稳过程
- 递归算法，计算高效
- 在线性状态空间、已知模型和二阶统计量条件下给出线性最小均方误差估计；在线性高斯条件下还等于条件均值估计

---

## 实践建议

### 时间序列分析步骤

1. **可视化**：绘制时间序列图
2. **检查平稳性**：ADF 检验、KPSS 检验
3. **差分**：如果非平稳，进行差分
4. **选择模型**：查看 ACF/PACF
5. **拟合模型**：AR、MA、ARMA
6. **诊断**：检查残差是否为白噪声
7. **预测**：使用拟合的模型进行预测

### Python 实现

```python
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# 拟合 ARIMA 模型
model = ARIMA(data, order=(p, d, q))
results = model.fit()

# 预测
forecast = results.get_forecast(steps=10)
```

---

## 关键要点

1. **平稳性是时间序列分析的基础**
2. **马尔可夫链用于序列建模和强化学习**
3. **AR、MA、ARMA 模型是经典的时间序列模型**
4. **谱分析提供了频域视角**
5. **Kalman 滤波的最优性依赖线性、模型与噪声统计等假设**

---

## 代码实验

- **代码文件：** [`code/ch01_dsp/stochastic_processes_demo.py`](../../../code/ch01_dsp/stochastic_processes_demo.py)
- **运行方式：** `python code/ch01_dsp/stochastic_processes_demo.py`

![Stochastic Processes](/assets/ch01_stochastic_processes.png)

*图：随机过程实验结果。包含马尔可夫链状态分布收敛、AR(1) 过程时序图、自相关函数（ACF）以及 Kalman 滤波效果对比。*

---

## 进一步阅读

- Hamilton, "Time Series Analysis"
- Brockwell & Davis, "Introduction to Time Series and Forecasting"
- Durbin & Koopman, "Time Series Analysis by State Space Methods"
