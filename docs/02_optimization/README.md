# 第 2 章：优化算法与传统机器学习

本章区分模型、目标函数、优化方法和验证指标，并用传统机器学习建立可检查的基线。

![优化系统视角](/diagrams/ch02-optimization-system.svg)

*章节主图：数据经过模型得到预测，目标函数给出误差，优化器据此更新参数；验证集用于检查更新是否改善泛化结果。*

## 小节目录

| 小节 | 内容 | 实验 |
|---|---|---|
| [2.1 梯度下降](01_gradient_descent.md) | 梯度、链式法则、学习率 | MMSE 与神经网络 |
| [2.2 自适应优化器](02_adaptive_optimizers.md) | Momentum、RMSprop、Adam | LMS 与 Adam |
| [2.3 优化器与模型的分工](03_optimization_and_traditional_ml.md) | 模型、损失、更新与训练系统 | MMSE 与神经网络 |
| [2.4 数值方法](04_numerical_methods.md) | 浮点误差、条件数、梯度检查 | 数值检查 |
| [2.5 线性与逻辑回归](05_linear_logistic_regression.md) | 连续值预测与二分类 | 拟合和决策边界 |
| [2.6 SVM 与核方法](06_svm_kernel_methods.md) | 间隔、Hinge 损失、核函数 | 线性核与 RBF 核 |
| [2.7 决策树与随机森林](07_decision_trees_random_forest.md) | 递归分裂、随机化与聚合 | 单树与随机森林 |

## 与后续章节的连接

```text
数据 + 模型 + 目标函数
          ↓ 优化或拟合
        参数
          ↓ 独立验证
    指标与错误样本
```

第 3 章把模型换成多层神经网络，第 4 章换成 Transformer，第 5 章把训练目标扩展到大规模 token 预测。这条训练闭环不变，但模型规模、数据与系统约束会增加。

线性模型、核方法和树模型不是 Transformer 的早期版本。它们使用不同假设和训练机制，在表格任务、小样本任务或需要简单基线时仍有独立价值。

## 本章学习闭环

1. 从主图区分模型、损失、优化器与验证集；
2. 运行同一数据上的两种更新方法；
3. 改变一个参数并保存损失曲线；
4. 用测试指标判断收敛更快是否真的改善结果。

## 最低实践任务

运行：

```bash
python code/ch02_optimization/lms_vs_adam.py
```

固定数据与初始化，改变学习率或优化器，比较至少两条损失曲线。记录：训练是否稳定、达到同一损失所需步数，以及最终测试误差。完成标准是能分别说明目标函数、更新规则和学习率的作用。

### 项目延伸：检测阈值也是优化变量

[信号检测项目](../../projects/signal_detection/README.md)在虚警率约束下选择阈值：

$$
\max_{\tau} P_D(\tau)\quad \text{subject to}\quad P_{FA}(\tau)\leq\alpha
$$

```bash
make run-project-signal-detection
```

分别设置 `--target-pfa 0.01` 和 `--target-pfa 0.10`，比较阈值、检测率与虚警率。这个实验不训练神经网络，但同样需要明确变量、目标、约束和验证数据。

## 代码实验

| 脚本 | 观察内容 |
|---|---|
| [`mmse_vs_nn.py`](../../code/ch02_optimization/mmse_vs_nn.py) | 线性估计与神经网络的假设差异 |
| [`lms_vs_adam.py`](../../code/ch02_optimization/lms_vs_adam.py) | 更新规则与收敛曲线 |
| [`linear_logistic_regression.py`](../../code/ch02_optimization/linear_logistic_regression.py) | 回归拟合与分类边界 |
| [`svm_kernel.py`](../../code/ch02_optimization/svm_kernel.py) | 线性核与 RBF 核边界 |
| [`decision_tree_random_forest.py`](../../code/ch02_optimization/decision_tree_random_forest.py) | 单树和随机森林的泛化间隙 |
| [`convex_analysis_demo.py`](../../code/ch02_optimization/convex_analysis_demo.py) | 凸、非凸与条件数 |

## 扩展内容

- [高级优化](extensions/advanced_optimization.md)：二阶方法、学习率调度、正则化和分布式优化。
- [凸分析](extensions/convex_analysis.md)：凸集、对偶问题与非凸优化。
- [运筹学](extensions/operations_research_basics.md)：规划、排队和服务系统约束。
- [树数据结构](extensions/tree_data_structures.md)：遍历、递归、复杂度和剪枝。

---

**下一步：** [2.1 梯度下降基础](01_gradient_descent.md)
