# 附录A：数学备忘

本附录收集了全书中常用的数学符号、公式和概念，方便随时查阅。更完整的数学基础主题（图论、信息论、凸分析等）见 [附录D：数学基础速查表](D_math_quick_reference.md)。

---

## 目录

- [A.1 线性代数](#a1-线性代数)
- [A.2 概率与统计](#a2-概率与统计)
- [A.3 复数基础](#a-3-复数基础)
- [A.4 信号处理](#a4-信号处理)
- [A.5 优化](#a5-优化)
- [A.6 激活函数](#a6-激活函数)
- [A.7 常用符号](#a7-常用符号)
- [A.8 常用恒等式](#a8-常用恒等式)

---

## A.1 线性代数

### 向量和矩阵

| 符号 | 含义 | 例子 |
|------|------|------|
| $\mathbf{x}$ | 列向量 | $\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ |
| $\mathbf{x}^T$ | 行向量（转置） | $\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$ |
| $\mathbf{A}$ | 矩阵 | $\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ |
| $\mathbf{A}^T$ | 矩阵转置 | 行列互换 |
| $\mathbf{A}^{-1}$ | 矩阵逆 | $\mathbf{A} \mathbf{A}^{-1} = \mathbf{I}$ |
| $\mathbf{I}$ | 单位矩阵 | 对角线为1，其他为0 |

### 矩阵运算

**矩阵乘法：**
$$(\mathbf{A}\mathbf{B})_{ij} = \sum_k A_{ik} B_{kj}$$

**向量点积：**
$$\mathbf{x}^T \mathbf{y} = \sum_i x_i y_i$$

**矩阵范数（Frobenius范数）：**
$$\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} |A_{ij}|^2}$$

**向量范数（L2范数）：**
$$\|\mathbf{x}\|_2 = \sqrt{\sum_i |x_i|^2}$$

### 矩阵求导

以下公式采用列向量梯度，并假设变量为实数。

**标量对向量求导：**
$$\nabla_{\mathbf{x}} (\mathbf{a}^{\mathsf T} \mathbf{x}) = \mathbf{a}$$

$$\nabla_{\mathbf{x}} (\mathbf{x}^{\mathsf T} \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^{\mathsf T}) \mathbf{x}$$

**链式法则：**
若 $\mathbf{x}\in\mathbb{R}^n$、$\mathbf{y}=f(\mathbf{x})\in\mathbb{R}^m$，且 $L$ 是标量，则

$$
\nabla_{\mathbf{x}}L
=\mathbf{J}_{\mathbf{y}\leftarrow\mathbf{x}}^{\mathsf T}\nabla_{\mathbf{y}}L,
\qquad
[\mathbf{J}_{\mathbf{y}\leftarrow\mathbf{x}}]_{ij}
=\frac{\partial y_i}{\partial x_j}.
$$

---

## A.2 概率与统计

### 高斯分布

**概率密度函数（PDF）：**
$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**多元高斯分布：**
$$p(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\mathbf{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

其中 $\boldsymbol{\mu}$ 是均值，$\mathbf{\Sigma}$ 是协方差矩阵。

### 期望和方差

**期望：**
$$\mathbb{E}[X] = \sum_x x\,p(x) \quad \text{（离散）}$$
$$\mathbb{E}[X] = \int x\,p(x)\,dx \quad \text{（连续）}$$

**方差：**
$$\operatorname{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**协方差：**
$$\operatorname{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]$$

### 常用分布

| 分布 | 参数 | 均值 | 方差 |
|------|------|------|------|
| 高斯 | $\mu, \sigma^2$ | $\mu$ | $\sigma^2$ |
| 均匀 | $a, b$ | $(a+b)/2$ | $(b-a)^2/12$ |
| 伯努利 | $p$ | $p$ | $p(1-p)$ |

---

## A.3 复数基础

复数在信号处理中无处不在，是理解傅里叶变换的关键。

### 定义和表示

**代数形式：**
$$z = a + jb$$

其中 $a$ 是实部，$b$ 是虚部，$j$ 是虚数单位（$j^2 = -1$）。

**极坐标形式：**
$$z = r e^{j\theta} = r(\cos\theta + j\sin\theta)$$

其中 $r = |z| = \sqrt{a^2 + b^2}$ 是模，$\theta = \operatorname{atan2}(b,a)$ 是幅角。使用双参数函数可以保留象限信息，并处理 $a=0$ 的情况。

### 运算

**乘法：**
$$(a + jb)(c + jd) = (ac - bd) + j(ad + bc)$$

**共轭：**
$$z^* = a - jb, \quad |z|^2 = z \cdot z^*$$

### 欧拉公式

$$e^{j\theta} = \cos\theta + j\sin\theta$$

**推论：**
$$\cos\theta = \frac{e^{j\theta} + e^{-j\theta}}{2}, \quad \sin\theta = \frac{e^{j\theta} - e^{-j\theta}}{2j}$$

复指数 $e^{j2\pi ft}$ 同时编码幅度和相位，这是傅里叶变换使用复指数的原因。

---

## A.4 信号处理

### 傅里叶变换

**离散傅里叶变换（DFT）：**
$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}$$

**逆变换：**
$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{j2\pi kn/N}$$

### 卷积

**离散卷积：**
$$y[n] = \sum_{m=-\infty}^{\infty} x[m] h[n-m]$$

**性质：**
- 交换律：$x * h = h * x$
- 结合律：$(x * h_1) * h_2 = x * (h_1 * h_2)$
- 频域乘法：$\mathcal{F}(x * h) = \mathcal{F}(x) \cdot \mathcal{F}(h)$

### 能量和功率

**信号能量：**
$$E = \sum_{n=-\infty}^{\infty} |x[n]|^2$$

**信号功率：**
$$P = \lim_{N \to \infty} \frac{1}{2N+1} \sum_{n=-N}^{N} |x[n]|^2$$

---

## A.5 优化

### 梯度下降

**更新规则：**
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla L(\mathbf{w}_t)$$

其中 $\alpha$ 是学习率，$\nabla L$ 是损失函数的梯度。

### 常用损失函数

**均方误差（MSE）：**
$$L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

**交叉熵（分类）：**
$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log(\hat{y}_{ic})$$

**L2正则化：**
$$L_{\text{reg}} = L + \lambda \sum_i w_i^2$$

### 优化器

**SGD（随机梯度下降）：**
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla L(\mathbf{w}_t)$$

**Adam：**
令 $\mathbf{g}_t=\nabla L(\mathbf{w}_t)$，平方、除法和开方均按元素计算：

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \mathbf{g}_t$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) \mathbf{g}_t^{\odot 2}$$
$$\widehat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \qquad \widehat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \frac{\widehat{\mathbf{m}}_t}{\sqrt{\widehat{\mathbf{v}}_t} + \epsilon}$$

---

## A.6 激活函数

| 函数 | 公式 | 导数 | 用途 |
|------|------|------|------|
| ReLU | $\max(0, x)$ | $\mathbb{1}_{x>0}$ | 隐层 |
| Sigmoid | $\frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | 二分类输出 |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2(x)$ | RNN隐层 |
| Softmax | $p_i=\frac{e^{x_i}}{\sum_j e^{x_j}}$ | $\frac{\partial p_i}{\partial x_j}=p_i(\delta_{ij}-p_j)$ | 多分类输出 |

---

## A.7 常用符号

| 符号 | 含义 |
|------|------|
| $\sum$ | 求和 |
| $\prod$ | 求积 |
| $\int$ | 积分 |
| $\partial$ | 偏导数 |
| $\nabla$ | 梯度 |
| $\mathbb{E}$ | 期望 |
| $\mathbb{R}$ | 实数集 |
| $\mathbb{C}$ | 复数集 |
| $j$ 或 $i$ | 虚数单位（$j^2 = -1$） |
| $\approx$ | 近似等于 |
| $\propto$ | 正比于 |

---

## A.8 常用恒等式

**三角恒等式：**
$$\sin^2(x) + \cos^2(x) = 1$$
$$e^{jx} = \cos(x) + j\sin(x)$$

**对数恒等式：**
$$\log(ab) = \log(a) + \log(b)$$
$$\log(a^b) = b\log(a)$$

**指数恒等式：**
$$e^{a+b} = e^a e^b$$
$$(e^a)^b = e^{ab}$$

**常见求导：**
$$\frac{d}{dx} x^n = nx^{n-1}, \quad \frac{d}{dx} e^x = e^x, \quad \frac{d}{dx} \log(x) = \frac{1}{x}$$

**常见积分：**
$$\int x^n dx = \frac{x^{n+1}}{n+1} + C \quad (n \neq -1), \quad \int e^x dx = e^x + C$$
