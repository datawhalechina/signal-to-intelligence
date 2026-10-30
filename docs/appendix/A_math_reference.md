# 附录 A：数学与符号参考

本附录是全教程数学符号、核心公式和适用条件的统一参考。它回答“公式怎么写、各量维度是什么、在什么条件下成立”。如果需要判断某章应先补哪些数学基础，请查[附录 D：数学前置导航](D_math_quick_reference.md)；术语含义和易混概念见[附录 G：术语表](G_glossary.md)。

## 使用方式

- 正文首次引入公式时负责解释问题背景，本附录负责统一写法和边界。
- 同一符号在局部章节有特殊含义时，以该章紧邻公式的定义为准。
- 概率公式中的对数底必须结合上下文说明：自然对数对应 nat，以 2 为底对应 bit。
- 数值公式默认使用实数；涉及复数时显式使用共轭转置 $\mathsf H$。

## 目录

- [A.0 符号与维度约定](#a0-符号与维度约定)
- [A.1 线性代数与微积分](#a1-线性代数与微积分)
- [A.2 概率、统计与信息论](#a2-概率统计与信息论)
- [A.3 复数基础](#a-3-复数基础)
- [A.4 信号处理](#a4-信号处理)
- [A.5 优化与统计学习](#a5-优化与统计学习)
- [A.6 深度学习](#a6-深度学习)
- [A.7 Transformer](#a7-transformer)
- [A.8 LLM 训练与生成](#a8-llm-训练与生成)
- [A.9 多模态表示与对齐](#a9-多模态表示与对齐)
- [A.10 数值计算与评价指标](#a10-数值计算与评价指标)

---

## A.0 符号与维度约定

### 排版约定

| 写法 | 含义 | 示例 |
| --- | --- | --- |
| $x$、$X$ | 标量或随机变量 | 学习率 $\alpha$、随机变量 $X$ |
| $\mathbf{x}$ | 列向量 | $\mathbf{x}\in\mathbb R^d$ |
| $\mathbf{X}$ | 矩阵 | token 表示 $\mathbf{X}\in\mathbb R^{n\times d}$ |
| $x_i$、$X_{ij}$ | 向量或矩阵元素 | 第 $i$ 个特征、矩阵第 $(i,j)$ 项 |
| $(\cdot)^{\mathsf T}$ | 实数转置 | $\mathbf{x}^{\mathsf T}\mathbf{y}$ |
| $(\cdot)^{\mathsf H}$ | 复共轭转置 | $\mathbf{x}^{\mathsf H}\mathbf{y}$ |
| $\widehat{x}$ | 估计值或预测值 | $\widehat{y}$ |
| $\mathbb E$、$\operatorname{Var}$、$\operatorname{Cov}$ | 期望、方差、协方差 | $\mathbb E[X]$ |

### 常用维度符号

| 符号 | 含义 |
| --- | --- |
| $N$ | 样本数或离散信号长度，具体含义由局部上下文给出 |
| $n$ | 序列中的 token 数 |
| $d$ | 模型隐藏维度 |
| $d_k,d_v$ | 注意力中 key/query 与 value 的维度 |
| $h$ | 注意力头数；数值微分中也可能表示步长 |
| $C$ | 类别数或通道数 |
| $B$ | batch size |

### 基本运算

矩阵乘法与实向量内积：

$$
(\mathbf A\mathbf B)_{ij}=\sum_k A_{ik}B_{kj},
\qquad
\langle\mathbf x,\mathbf y\rangle=\mathbf x^{\mathsf T}\mathbf y.
$$

复向量内积使用共轭转置：

$$
\langle\mathbf x,\mathbf y\rangle=\mathbf x^{\mathsf H}\mathbf y.
$$

常用范数：

$$
\|\mathbf x\|_2=\sqrt{\sum_i|x_i|^2},
\qquad
\|\mathbf A\|_F=\sqrt{\sum_{i,j} |A_{ij}|^2}.
$$

---

## A.1 线性代数与微积分

### 线性映射、投影与相似度

仿射映射：

$$
\mathbf y=\mathbf W\mathbf x+\mathbf b,
\quad
\mathbf W\in\mathbb R^{d_{out}\times d_{in}}.
$$

向量 $\mathbf u$ 在非零向量 $\mathbf v$ 上的投影：

$$
\operatorname{proj}_{\mathbf v}\mathbf u
=\frac{\mathbf v^{\mathsf T}\mathbf u}{\mathbf v^{\mathsf T}\mathbf v}\mathbf v.
$$

余弦相似度：

$$
\operatorname{cos}(\mathbf u,\mathbf v)
=\frac{\mathbf u^{\mathsf T}\mathbf v}{\|\mathbf u\|_2\|\mathbf v\|_2},
\qquad \mathbf u,\mathbf v\neq\mathbf 0.
$$

### 梯度、Jacobian 与链式法则

以下采用列向量梯度，并假设变量为实数：

$$
\nabla_{\mathbf x}(\mathbf a^{\mathsf T}\mathbf x)=\mathbf a,
\qquad
\nabla_{\mathbf x}(\mathbf x^{\mathsf T}\mathbf A\mathbf x)
=(\mathbf A+\mathbf A^{\mathsf T})\mathbf x.
$$

若 $\mathbf y=f(\mathbf x)\in\mathbb R^m$、$\mathbf x\in\mathbb R^n$，且 $L$ 为标量，则

$$
\nabla_{\mathbf{x}}L
=\mathbf{J}_{\mathbf{y}\leftarrow\mathbf{x}}^{\mathsf T}\nabla_{\mathbf{y}}L,
\qquad
[\mathbf J_{\mathbf y\leftarrow\mathbf x}]_{ij}
=\frac{\partial y_i}{\partial x_j}.
$$

常见标量导数：

$$
\frac{d}{dx}x^p=px^{p-1},
\qquad
\frac{d}{dx}e^x=e^x,
\qquad
\frac{d}{dx}\log x=\frac1x\quad(x>0).
$$

---

## A.2 概率、统计与信息论

### 期望、方差与协方差

$$
\mathbb E[X]=\sum_x x\,p(x)\quad\text{（离散）},
\qquad
\mathbb E[X]=\int x\,p(x)\,dx\quad\text{（连续）}.
$$

$$
\operatorname{Var}(X)
=\mathbb E[(X-\mathbb E[X])^2]
=\mathbb E[X^2]-(\mathbb E[X])^2.
$$

$$
\operatorname{Cov}(X,Y)
=\mathbb E[(X-\mathbb E[X])(Y-\mathbb E[Y])].
$$

### 高斯分布

一维高斯分布：

$$
p(x)=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac{(x-\mu)^2}{2\sigma^2}\right].
$$

$d$ 维高斯分布：

$$
p(\mathbf x)=\frac{
\exp\left[-\frac12(\mathbf x-\boldsymbol\mu)^{\mathsf T}
\boldsymbol\Sigma^{-1}(\mathbf x-\boldsymbol\mu)\right]
}{(2\pi)^{d/2}|\boldsymbol\Sigma|^{1/2}}.
$$

该密度写法要求协方差矩阵 $\boldsymbol\Sigma$ 正定；奇异协方差需要使用退化分布的处理方式。

### 熵、交叉熵、KL 散度与互信息

$$
H(P)=-\sum_x P(x)\log P(x),
$$

$$
H(P,Q)=-\sum_xP(x)\log Q(x),
$$

$$
D_{\mathrm{KL}}(P\|Q)
=\sum_xP(x)\log\frac{P(x)}{Q(x)},
$$

$$
I(X;Y)=D_{\mathrm{KL}}(P_{XY}\|P_XP_Y)
=H(X)-H(X\mid Y).
$$

$D_{\mathrm{KL}}$ 一般不对称，因此不是距离度量。

---

## A.3 复数基础

本节编号保持稳定，供第 1 章傅里叶内容直接引用。

### 表示

$$
z=a+jb=re^{j\theta}=r(\cos\theta+j\sin\theta),
$$

其中

$$
r=|z|=\sqrt{a^2+b^2},
\qquad
\theta=\operatorname{atan2}(b,a),
\qquad
j^2=-1.
$$

`atan2` 能保留象限信息，并处理 $a=0$ 的情况。

### 共轭与欧拉公式

$$
z^*=a-jb,
\qquad
|z|^2=zz^*,
\qquad
e^{j\theta}=\cos\theta+j\sin\theta.
$$

$$
\cos\theta=\frac{e^{j\theta}+e^{-j\theta}}2,
\qquad
\sin\theta=\frac{e^{j\theta}-e^{-j\theta}}{2j}.
$$

---

## A.4 信号处理

### 采样与归一化频率

以采样率 $f_s$ 对连续信号 $x_c(t)$ 采样：

$$
x[n]=x_c(nT_s),
\qquad
T_s=\frac1{f_s}.
$$

模拟频率 $f$ 对应数字角频率 $\omega=2\pi f/f_s$。对带限信号，避免频谱混叠通常要求采样率严格高于最高频率的两倍，并配合实际抗混叠滤波器。

### DFT 与逆变换

$$
X[k]=\sum_{n=0}^{N-1}x[n]e^{-j2\pi kn/N},
$$

$$
x[n]=\frac1N\sum_{k=0}^{N-1}X[k]e^{j2\pi kn/N}.
$$

不同库可能把归一化因子放在正变换、逆变换或两侧，比较结果时必须确认约定。

### 卷积与 LTI 系统

$$
y[n]=(x*h)[n]=\sum_{m=-\infty}^{\infty}x[m]h[n-m].
$$

在变换存在且满足相应收敛条件时：

$$
\mathcal F\{x*h\}=\mathcal F\{x\}\,\mathcal F\{h\}.
$$

离散时间 LTI 系统 BIBO 稳定的充要条件是 $\sum_n|h[n]|<\infty$。对因果有理系统，这等价于所有极点严格位于单位圆内。

### 能量、功率与信噪比

$$
E_x=\sum_{n=-\infty}^{\infty}|x[n]|^2,
\qquad
P_x=\lim_{N\to\infty}\frac1{2N+1}\sum_{n=-N}^{N}|x[n]|^2.
$$

若 $P_s$ 和 $P_n$ 使用同一口径估计信号与噪声功率，则

$$
\operatorname{SNR}=\frac{P_s}{P_n},
\qquad
\operatorname{SNR}_{\mathrm{dB}}=10\log_{10}\frac{P_s}{P_n}.
$$

---

## A.5 优化与统计学习

### 经验风险与正则化

$$
\widehat R(\mathbf w)=\frac1N\sum_{i=1}^N
\ell(f_{\mathbf w}(\mathbf x_i),y_i),
$$

$$
L_{\mathrm{reg}}(\mathbf w)=\widehat R(\mathbf w)+\lambda\Omega(\mathbf w).
$$

常见选择为 $\Omega(\mathbf w)=\|\mathbf w\|_2^2$，但权重衰减与“把 L2 项直接加到损失”在所有优化器中并不完全等价。

### 常用损失

均方误差：

$$
L_{\mathrm{MSE}}=\frac1N\sum_{i=1}^N(y_i-\widehat y_i)^2.
$$

$C$ 类交叉熵，其中 $y_{ic}$ 是目标分布、$p_{ic}$ 是预测概率：

$$
L_{\mathrm{CE}}=-\frac1N\sum_{i=1}^N\sum_{c=1}^C
y_{ic}\log p_{ic}.
$$

### 梯度下降与 Adam

$$
\mathbf w_{t+1}=\mathbf w_t-\alpha\nabla L(\mathbf w_t).
$$

对 Adam，令 $\mathbf{g}_t=\nabla L(\mathbf w_t)$，平方、除法和开方均按元素计算：

$$
\mathbf{m}_t=\beta_1\mathbf{m}_{t-1}+(1-\beta_1)\mathbf{g}_t,
$$

$$
\mathbf{v}_t=\beta_2\mathbf{v}_{t-1}+(1-\beta_2)\mathbf{g}_t^{\odot 2},
$$

$$
\widehat{\mathbf{m}}_t=\frac{\mathbf{m}_t}{1-\beta_1^t},
\qquad
\widehat{\mathbf{v}}_t=\frac{\mathbf{v}_t}{1-\beta_2^t},
$$

$$
\mathbf w_{t+1}=\mathbf w_t-
\alpha\frac{\widehat{\mathbf{m}}_t}{\sqrt{\widehat{\mathbf{v}}_t}+\epsilon}.
$$

### 凸性边界

若可行域为凸集、目标为凸函数，则任一局部最优也是全局最优；解仍可能不存在或不唯一。深度网络通常是非凸优化问题，不能沿用这一全局保证。

---

## A.6 深度学习

### 神经元与激活函数

$$
\mathbf h=\phi(\mathbf W\mathbf x+\mathbf b).
$$

| 函数 | 定义 | 导数或说明 |
| --- | --- | --- |
| ReLU | $\max(0,x)$ | $x\neq0$ 时为 $\mathbb 1_{x>0}$；$x=0$ 处实现选择一个次梯度 |
| Sigmoid | $\sigma(x)=1/(1+e^{-x})$ | $\sigma(x)(1-\sigma(x))$ |
| Tanh | $\tanh x$ | $1-\tanh^2x$ |
| Softmax | $p_i=e^{z_i}/\sum_j e^{z_j}$ | $\partial p_i/\partial z_j=p_i(\delta_{ij}-p_j)$ |

### CNN 中的局部运算

许多深度学习库把“卷积层”实现为互相关。以一维输入为例：

$$
y_c[n]=b_c+\sum_{c'=1}^{C_{in}}\sum_{m=0}^{K-1}
w_{c,c'}[m]x_{c'}[n+m].
$$

它与 DSP 卷积的差别是核没有翻转；训练时核由数据学习，因此不影响其作为可学习局部模式提取器的作用。

### 反向传播

反向传播不是另一套求导规则，而是把 A.1 的链式法则按计算图反向复用。若 $L=L(\mathbf h_L)$，则每层都把上游梯度与本层局部 Jacobian 相乘；连乘的尺度解释了梯度消失与梯度爆炸风险。

---

## A.7 Transformer

### 缩放点积注意力

令输入 $\mathbf X\in\mathbb R^{n\times d}$，单头投影为

$$
\mathbf Q=\mathbf X\mathbf W_Q\in\mathbb R^{n\times d_k},
\quad
\mathbf K=\mathbf X\mathbf W_K\in\mathbb R^{n\times d_k},
\quad
\mathbf V=\mathbf X\mathbf W_V\in\mathbb R^{n\times d_v}.
$$

$$
\operatorname{Attention}(\mathbf Q,\mathbf K,\mathbf V)
=\operatorname{softmax}_{\text{key}}
\left(\frac{\mathbf Q\mathbf K^{\mathsf T}}{\sqrt{d_k}}+\mathbf M\right)\mathbf V.
$$

softmax 对每个 query 所对应的 key 轴归一化。允许连接处 $M_{ij}=0$，被屏蔽处通常取 $M_{ij}=-\infty$ 或足够小的可表示数。因果掩码要求位置 $i$ 不能读取 $j>i$ 的位置。

### 多头注意力

$$
\operatorname{head}_r
=\operatorname{Attention}(\mathbf X\mathbf W_Q^{(r)},
\mathbf X\mathbf W_K^{(r)},\mathbf X\mathbf W_V^{(r)}),
$$

$$
\operatorname{MHA}(\mathbf X)
=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)\mathbf W_O.
$$

标准全注意力的分数矩阵为 $n\times n$，其主要时间和显存开销随序列长度呈二次增长；精确复杂度还取决于投影维度和实现。

### 正弦位置编码

$$
\operatorname{PE}(p,2i)=\sin\left(p/10000^{2i/d}\right),
$$

$$
\operatorname{PE}(p,2i+1)=\cos\left(p/10000^{2i/d}\right).
$$

这是固定绝对位置编码的一种形式，不代表现代 LLM 都采用该方案；RoPE 等方法会以不同方式把相对位置信息注入 query 和 key。

### LayerNorm、残差与 FFN

对单个 token 的 $d$ 维表示 $\mathbf x$：

$$
\mu=\frac1d\sum_{i=1}^d x_i,
\qquad
\sigma^2=\frac1d\sum_{i=1}^d(x_i-\mu)^2,
$$

$$
\operatorname{LayerNorm}(\mathbf x)
=\boldsymbol\gamma\odot\frac{\mathbf x-\mu}{\sqrt{\sigma^2+\epsilon}}
+\boldsymbol\beta.
$$

残差连接与逐 token 前馈网络可写为

$$
\mathbf y=\mathbf x+F(\mathbf x),
\qquad
\operatorname{FFN}(\mathbf x)=\phi(\mathbf x\mathbf W_1+\mathbf b_1)\mathbf W_2+\mathbf b_2.
$$

Pre-LN 与 Post-LN 的归一化位置不同，不能只凭上述组件公式判断具体架构。

---

## A.8 LLM 训练与生成

### 自回归分解与训练目标

$$
p_\theta(x_{1:T})=\prod_{t=1}^T p_\theta(x_t\mid x_{<t}).
$$

token 级负对数似然：

$$
L_{\mathrm{NLL}}(\theta)
=-\frac1T\sum_{t=1}^T\log p_\theta(x_t\mid x_{<t}).
$$

训练时通常使用真实历史 token 作为条件；生成时把已选出的 token 追加到上下文，二者共享下一 token 条件分布，但运行过程不同。

### 温度采样与困惑度

$$
p_i(\tau)=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)},
\qquad \tau>0.
$$

较低温度通常使分布更尖锐，较高温度使其更平缓；它改变采样分布，不保证事实正确性。

若平均交叉熵使用自然对数：

$$
\operatorname{PPL}=\exp(L_{\mathrm{NLL}}).
$$

若使用以 2 为底的对数，则 $\operatorname{PPL}=2^{L_{\mathrm{NLL}}}$。只有在相同 tokenizer、数据与计分口径下，困惑度才适合直接比较。

### LoRA

对冻结权重 $\mathbf W_0\in\mathbb R^{d_{out}\times d_{in}}$，低秩更新写为

$$
\mathbf W=\mathbf W_0+\Delta\mathbf W,
\qquad
\Delta\mathbf W=\frac{\alpha}{r}\mathbf B\mathbf A,
$$

其中

$$
\mathbf A\in\mathbb R^{r\times d_{in}},
\qquad
\mathbf B\in\mathbb R^{d_{out}\times r}.
$$

该层新增可训练参数量为 $r(d_{in}+d_{out})$。实际节省比例取决于插入层、是否训练偏置及其他模块，rank 小也不自动保证任务效果。

---

## A.9 多模态表示与对齐

### 归一化表示与相似度

给定两种模态编码器 $f$、$g$：

$$
\mathbf v_i=\frac{f(x_i)}{\|f(x_i)\|_2},
\qquad
\mathbf t_i=\frac{g(y_i)}{\|g(y_i)\|_2},
\qquad
s_{ij}=\mathbf v_i^{\mathsf T}\mathbf t_j.
$$

归一化后点积等于余弦相似度。若模型训练时未采用该归一化，不能默认向量模长没有信息。

### 对称 InfoNCE

对 batch 中 $B$ 个匹配样本，图像或信号到文本方向：

$$
L_{x\to y}=-\frac1B\sum_{i=1}^B
\log\frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^B\exp(s_{ij}/\tau)}.
$$

反方向 $L_{y\to x}$ 对相似度矩阵按列归一化，对称目标为

$$
L_{\mathrm{contrastive}}=\frac12(L_{x\to y}+L_{y\to x}).
$$

批内其他样本被当作负例；重复内容、近义描述或错误配对会产生假负例，因此损失值不能脱离数据构造解释。

### patch、视觉 token 与计算量

对于 $H\times W$ 图像和不重叠的 $P\times P$ patch，若高宽可整除：

$$
N_{patch}=\frac HP\frac WP.
$$

高宽同时放大 $k$ 倍时，patch 数约增至 $k^2$ 倍；若使用全注意力，仅视觉 token 的注意力分数规模约增至 $k^4$ 倍。实际系统还受分块、token 压缩、局部注意力和文本长度影响。

信号模态同样可以把波形窗口、时频块或事件片段映射成 token。对齐公式不要求输入必须是图像，关键是编码器、时间或空间定位以及配对监督是否保留任务所需信息。

---

## A.10 数值计算与评价指标

### 稳定 softmax 与 log-sum-exp

softmax 对整体平移不变，因此实现时通常减去最大 logit：

$$
\operatorname{softmax}(\mathbf z)_i
=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}},
\qquad m=\max_j z_j.
$$

$$
\operatorname{LSE}(\mathbf z)
=m+\log\sum_j e^{z_j-m}.
$$

### 浮点误差与梯度检查

在正常舍入且不发生上溢、下溢的简化模型下：

$$
\operatorname{fl}(x)=x(1+\delta),
\qquad |\delta|\leq\epsilon.
$$

中心差分：

$$
f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}.
$$

其截断误差为 $O(h^2)$、舍入误差约为 $O(\epsilon/h)$，因此无量纲理想模型下常取 $h \sim \epsilon^{1/3}$；实际步长还要结合变量与函数尺度。

### 分类指标

$$
\operatorname{Precision}=\frac{TP}{TP+FP},
\qquad
\operatorname{Recall}=\frac{TP}{TP+FN},
$$

$$
F_1=2\frac{\operatorname{Precision}\cdot\operatorname{Recall}}
{\operatorname{Precision}+\operatorname{Recall}}.
$$

分母为零时必须声明实现约定。多分类任务还需说明 macro、micro 或 weighted 聚合方式；检测任务则必须同时说明匹配阈值和 IoU 等判定规则。
