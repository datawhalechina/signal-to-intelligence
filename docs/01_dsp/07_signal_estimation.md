# 1.7 信号估计

**核心问题：** 如何从噪声中估计信号的参数？如何评估估计器的性能？

---

## 参数估计问题

### 问题设置

**观测模型：**
$$y[n] = s(x[n]; \theta) + w[n], \quad n = 0, 1, \ldots, N-1$$

其中：
- $y[n]$：观测信号
- $s(x[n]; \theta)$：参数化信号模型
- $\theta$：待估计的参数
- $w[n]$：加性高斯白噪声

**估计任务：** 给定观测 $\mathbf{y} = [y[0], y[1], \ldots, y[N-1]]^T$，估计参数 $\theta$。

### 参数估计的例子

**例1：正弦波参数估计**
$$s(t; \theta) = A \sin(2\pi f t + \phi)$$

参数：$\theta = [A, f, \phi]^T$（幅度、频率、相位）

**例2：信道估计**
$$y[n] = h \cdot x[n] + w[n]$$

参数：$\theta = h$（信道增益）

**例3：时延估计**
$$y[n] = s[n - \tau] + w[n]$$

参数：$\theta = \tau$（时延）

---

## 最大似然估计（Maximum Likelihood Estimation, MLE）

### 原理

**似然函数：** 给定参数 $\theta$，观测 $\mathbf{y}$ 的概率。

$$L(\theta; \mathbf{y}) = p(\mathbf{y}|\theta)$$

**最大似然估计：** 选择使似然函数最大的参数。

$$\hat{\theta}_{\text{ML}} = \arg\max_{\theta} L(\theta; \mathbf{y})$$

**对数似然函数：** 为了计算方便，通常最大化对数似然。

$$\hat{\theta}_{\text{ML}} = \arg\max_{\theta} \ln L(\theta; \mathbf{y})$$

### 高斯噪声下的MLE

**假设：** 噪声是AWGN，方差为 $\sigma^2$。

**似然函数：**
$$L(\theta; \mathbf{y}) = \prod_{n=0}^{N-1} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y[n] - s(x[n];\theta))^2}{2\sigma^2}\right)$$

**对数似然函数：**
$$\ln L(\theta; \mathbf{y}) = -\frac{1}{2\sigma^2} \sum_{n=0}^{N-1} (y[n] - s(x[n];\theta))^2 + \text{常数}$$

**最大化对数似然等价于最小化误差平方和：**
$$\hat{\theta}_{\text{ML}} = \arg\min_{\theta} \sum_{n=0}^{N-1} (y[n] - s(x[n];\theta))^2$$

### MLE 的渐近性质

以下性质需要模型可辨识、参数位于合适的内部区域、似然满足正则性条件，以及样本结构允许相应大样本极限：

**一致性：** 当 $N \to \infty$ 时，$\hat{\theta}_{\text{ML}} \to \theta_0$（真实参数）

**渐近正态性：** 当 $N$ 很大时，$\hat{\theta}_{\text{ML}}$ 近似高斯分布

**渐近有效性：** 在正则性条件满足且样本足够多时，MLE 可以逼近 Cramér-Rao 界

---

## 最小二乘估计（Least Squares Estimation, LSE）

### 原理

**最小二乘准则：** 最小化观测与模型的误差平方和。

$$\hat{\theta}_{\text{LS}} = \arg\min_{\theta} \sum_{n=0}^{N-1} (y[n] - s(x[n];\theta))^2$$

**与 MLE 的关系：** 在独立、同方差、零均值高斯加性噪声下，未加权 LSE 等价于 MLE。噪声相关或方差不同时，应使用广义/加权最小二乘口径。

### 线性最小二乘

**线性模型：**
$$y[n] = \mathbf{h}^T \mathbf{x}[n] + w[n]$$

其中 $\mathbf{h}$ 是参数向量，$\mathbf{x}[n]$ 是观测向量。

**矩阵形式：**
$$\mathbf{y} = \mathbf{X} \mathbf{h} + \mathbf{w}$$

其中 $\mathbf{X}$ 是设计矩阵。

若 $\mathbf X$ 满列秩，实数情形的闭式解为

$$
\widehat{\mathbf h}_{\mathrm{LS}}
=(\mathbf X^{\mathsf T}\mathbf X)^{-1}\mathbf X^{\mathsf T}\mathbf y.
$$

一般情形写作 $\widehat{\mathbf h}_{\mathrm{LS}}=\mathbf X^+\mathbf y$，其中 $\mathbf X^+$ 是 Moore–Penrose 伪逆；复数数据使用共轭转置 $\mathsf H$。

**优点：** 有闭式解，计算快速。

### 非线性最小二乘

**非线性模型：** $s(x[n];\theta)$ 对 $\theta$ 非线性。

**求解方法：**
- 梯度下降法
- 牛顿法
- Levenberg-Marquardt算法

**优点：** 适用于复杂模型。

**缺点：** 需要迭代，可能陷入局部最优。

---

## 贝叶斯估计

### 原理

**贝叶斯定理：**
$$p(\theta|\mathbf{y}) = \frac{p(\mathbf{y}|\theta) p(\theta)}{p(\mathbf{y})}$$

其中：
- $p(\theta|\mathbf{y})$：后验分布
- $p(\mathbf{y}|\theta)$：似然函数
- $p(\theta)$：先验分布
- $p(\mathbf{y})$：证据

**贝叶斯估计：** 利用后验分布进行估计。

### 常见的贝叶斯估计器

**1. 最大后验估计（MAP）**
$$\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} p(\theta|\mathbf{y})$$

**2. 最小均方误差估计（MMSE）**
$$\hat{\theta}_{\text{MMSE}} = \mathbb E[\theta|\mathbf{y}] = \int \theta\,p(\theta|\mathbf{y})\,d\theta$$

**3. 最小绝对误差估计（MAE）**
$$\hat{\theta}_{\text{MAE}} = \text{median}(\theta|\mathbf{y})$$

### 贝叶斯 vs 频率学派

**频率学派（MLE）：**
- 参数是固定但未知的常数
- 估计器的性能由重复实验的分布决定

**贝叶斯学派：**
- 参数是随机变量
- 利用先验知识改进估计
- 结果是后验分布，不仅是点估计

**实际应用：** 先验可以在数据有限时提供正则化与不确定性表达，但结果依赖先验和模型是否合理，不能笼统断言一定优于频率学派估计。

---

## 克拉美罗界（Cramér-Rao Bound, CRB）

### 定义

**CRB：** 任何无偏估计器的方差的下界。

$$\text{Var}(\hat{\theta}) \geq \text{CRB}(\theta) = \frac{1}{I(\theta)}$$

其中标量参数时 $I(\theta)$ 是 Fisher 信息；向量参数时应使用 Fisher 信息矩阵，CRB 写成协方差矩阵的不等式。

### Fisher 信息：标量与向量形式

**标量参数：**
$$I(\theta)=\mathbb E\left[\left(\frac{\partial\log p(\mathbf y;\theta)}{\partial\theta}\right)^2\right]$$

**向量参数：** 若得分向量为 $\nabla_{\boldsymbol\theta}\log p(\mathbf y;\boldsymbol\theta)$，则 Fisher 信息矩阵为

$$
\mathbf I(\boldsymbol\theta)
=\mathbb E\!\left[
\nabla_{\boldsymbol\theta}\log p(\mathbf y;\boldsymbol\theta)
\nabla_{\boldsymbol\theta}\log p(\mathbf y;\boldsymbol\theta)^{\mathsf T}
\right].
$$

**性质：**
- 对可辨识且满足正则条件的标量参数，$I(\theta)>0$
- $I(\theta)$ 越大，CRB越小（估计越精确）
- 独立且有信息的新增观测通常使 Fisher 信息相加；重复或退化观测不一定提供新信息

### 高斯噪声下的CRB

**对于独立同方差实高斯噪声和标量参数，Fisher 信息为：**
$$I(\theta) = \frac{1}{\sigma^2} \sum_{n=0}^{N-1} \left(\frac{\partial s(x[n];\theta)}{\partial \theta}\right)^2$$

**CRB：**
$$\text{CRB}(\theta) = \frac{\sigma^2}{\sum_{n=0}^{N-1} \left(\frac{\partial s(x[n];\theta)}{\partial \theta}\right)^2}$$

### 接近CRB的条件

**MLE通常需要在以下条件下才可能逼近CRB：**
1. 模型设定正确，满足常见正则性条件
2. 观测数 $N$ 足够大
3. 参数在内部（不在边界）

---

## 估计器性能指标

### 1. 偏差（Bias）

**定义：**
$$\operatorname{Bias}(\hat{\theta}) = \mathbb E[\hat{\theta}] - \theta$$

**无偏估计器：** $\text{Bias}(\hat{\theta}) = 0$

**有偏估计器：** $\text{Bias}(\hat{\theta}) \neq 0$

**例子：**
- MLE通常是渐近无偏的
- 某些贝叶斯估计器可能有偏

### 2. 方差（Variance）

**定义：**
$$\operatorname{Var}(\hat{\theta}) = \mathbb E[(\hat{\theta} - \mathbb E[\hat{\theta}])^2]$$

**性质：**
- 方差越小，估计越稳定
- 方差受CRB限制

### 3. 均方误差（Mean Squared Error, MSE）

**定义：**
$$\operatorname{MSE}(\hat{\theta}) = \mathbb E[(\hat{\theta} - \theta)^2]$$

**分解：**
$$\text{MSE}(\hat{\theta}) = \text{Bias}^2(\hat{\theta}) + \text{Var}(\hat{\theta})$$

**权衡：** 有时接受小的偏差来换取更小的方差。

### 4. 一致性（Consistency）

**定义：** 当 $N \to \infty$ 时，$\hat{\theta} \xrightarrow{p} \theta$

**含义：** 观测越多，估计越接近真实值。

---

## 实际应用

### 1. 频率估计

**问题：** 从噪声中估计正弦波的频率。

$$y[n] = A \sin(2\pi f n / f_s + \phi) + w[n]$$

**方法：**
- FFT：快速但分辨率有限
- MLE：精确但计算复杂
- 子空间方法（MUSIC）：高分辨率

![Parameter Estimation](/assets/ch01_parameter_estimation.png)

**代码文件：** [`code/ch01_dsp/parameter_estimation.py`](../../code/ch01_dsp/parameter_estimation.py)  
**运行方式：** `python code/ch01_dsp/parameter_estimation.py`

*图1.8：参数估计结果示意。随着观测条件改善或样本增加，估计误差会下降，并逐步逼近理论性能界。*

**应用：** 功率系统监测、音乐分析

### 2. 信道估计

**问题：** 估计通信信道的参数。

$$y[n] = h \cdot x[n] + w[n]$$

**方法：**
- 最小二乘：简单快速
- 贝叶斯：利用信道统计特性
- 自适应滤波：跟踪时变信道

**应用：** 无线通信、雷达

### 3. 时延估计

**问题：** 估计信号的传播时延。

$$y[n] = s[n - \tau] + w[n]$$

**方法：**
- 相关法：计算 $y[n]$ 与 $s[n]$ 的相关性
- MLE：精确估计
- 子空间方法：多信号时延估计

**应用：** 定位、测距、同步

### 4. 医学信号处理

**问题：** 从生物信号中估计参数。

**例子：**
- 心率估计：从ECG估计心跳频率
- 呼吸率估计：从呼吸信号估计呼吸频率
- 脑波频率估计：从EEG估计脑波频率

**方法：** 通常使用频率估计技术

---

## 本节小结

信号估计是信号处理的核心问题：

- **估计方法**：MLE、LSE、贝叶斯估计
- **性能界**：Cramér-Rao界
- **性能指标**：偏差、方差、MSE、一致性
- **实际应用**：频率估计、信道估计、时延估计、医学信号处理

选择合适的估计方法取决于问题的具体特性和可用的先验知识。

---

## 与后续章节的联系

估计理论强调“从有限观测中恢复未知量”，这是后续章节里训练和推理的共同问题：

1. **MLE / LSE**
   - 对应于用数据拟合模型参数和输出分布

2. **贝叶斯估计**
   - 对应于把先验知识与观测结果结合起来做判断

3. **误差分析**
   - 对应于偏差、方差、不确定性和损失函数的分解

4. **性能界**
   - 对应于理解模型在数据和噪声约束下的理论极限

5. **高分辨率方法**
   - 对应于后续章节中的细粒度表示学习与多子空间建模

这些概念会在第 2 章优化、第 5 章训练和第 6 章评估中继续出现。

---

**下一节：** [1.8 矩阵分解应用](08_matrix_decomposition.md)
