# 扩展：树数据结构基础

本文档提供树数据结构的详细讲解，供对决策树实现细节感兴趣的读者参考。

---

## 树的基本概念

### 什么是树

**树**是一种分层的数据结构，由节点和边组成。

**树的基本概念：**
- **根节点**：没有父节点的节点（树的起点）
- **内部节点**：有子节点的节点
- **叶子节点**：没有子节点的节点（树的终点）
- **深度**：从根节点到该节点的边数
- **高度**：从该节点到最深叶子节点的边数
- **度数**：节点的子节点个数

**树的性质：**
- 树是连通的无环图
- $n$ 个节点的树有 $n-1$ 条边
- 树中任意两个节点之间有唯一的路径

### 决策树的树结构

**决策树是一种特殊的树：**
- **根节点**：代表整个数据集
- **内部节点**：代表一个特征的判断（如 $x > 0.5$？）
- **边**：代表判断结果（是/否）
- **叶子节点**：代表最终的预测结果（类别或数值）

**例子：**
```
        x₁ > 5?
       /      \
      是        否
     /          \
  x₂ > 3?      类别A
  /    \
 是     否
/       \
类别B   类别C
```

---

## 树的遍历与预测

### 预测过程

**预测过程就是树的遍历：**
1. 从根节点开始
2. 根据特征值选择分支（左或右）
3. 到达下一个节点，重复步骤2
4. 最终到达叶子节点，输出预测结果

**时间复杂度：** $O(h)$，其中 $h$ 是树的高度

### 树的遍历方法

- **前序遍历**：先访问节点，再访问子节点（用于预测）
- **中序遍历**：先访问左子树，再访问节点，最后访问右子树
- **后序遍历**：先访问子节点，再访问节点

---

## 二叉树

### 二叉树的定义

**二叉树**是每个节点最多有两个子节点的树。

**二叉树的性质：**
- 第 $i$ 层最多有 $2^{i-1}$ 个节点
- 深度为 $h$ 的二叉树最多有 $2^h - 1$ 个节点
- 有 $n$ 个节点的二叉树的高度至少为 $\lceil \log_2(n+1) \rceil$

### 决策树与二叉树

决策树通常是二叉树，因为每个分割点都产生两个分支（是/否）。

---

## 树的复杂度分析

### 构建树的时间复杂度

对于 $n$ 个样本、$d$ 个特征的数据集：

- **最坏情况**：$O(n^2 d)$（每层只分割一个样本）
- **平均情况**：$O(n \log n \cdot d)$（平衡树）
- **最好情况**：$O(n d)$（完全平衡树）

### 查询树的时间复杂度

- **最坏情况**：$O(h)$，其中 $h$ 是树的高度
- **平均情况**：$O(\log n)$（平衡树）

### 空间复杂度

- **存储树**：$O(n)$（最多 $n$ 个节点）
- **递归深度**：$O(h)$（最坏情况 $O(n)$）

---

## 树的递归实现

### 节点定义

```python
class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature        # 分割特征的索引
        self.threshold = threshold    # 分割阈值
        self.left = left             # 左子树
        self.right = right           # 右子树
        self.value = value           # 叶子节点的预测值
```

### 递归构建

```python
def build_tree(X, y, depth=0, max_depth=10):
    if depth >= max_depth or len(y) < 2:
        # 停止条件：达到最大深度或样本过少
        return TreeNode(value=np.mean(y))
    
    # 寻找最佳分割
    best_feature, best_threshold = find_best_split(X, y)
    
    if best_feature is None:
        # 无法继续分割
        return TreeNode(value=np.mean(y))
    
    # 分割数据
    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask
    
    # 递归构建左右子树
    left_subtree = build_tree(X[left_mask], y[left_mask], depth+1, max_depth)
    right_subtree = build_tree(X[right_mask], y[right_mask], depth+1, max_depth)
    
    return TreeNode(feature=best_feature, threshold=best_threshold,
                   left=left_subtree, right=right_subtree)
```

### 递归预测

```python
def predict_tree(node, x):
    if node.value is not None:
        # 叶子节点
        return node.value
    
    if x[node.feature] <= node.threshold:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)
```

---

## 树的剪枝

### 为什么需要剪枝

决策树容易过拟合，因为可以无限深地分割，直到每个叶子节点只有一个样本。

### 剪枝方法

**前剪枝（Pre-pruning）：** 在构建过程中停止分割
- 限制树的最大深度
- 设置最小样本数
- 设置最小信息增益阈值

**后剪枝（Post-pruning）：** 先构建完整的树，再删除不必要的节点
- 使用验证集评估每个子树
- 删除不能改进验证集性能的节点

---

## 相关资源

### 文件位置

本文档是第2章的**扩展内容**，位于：
```
docs/02_optimization/extensions/tree_data_structures.md
```

### 相关主文档

- **主文档：** [第2章：优化算法与传统机器学习](../README.md)
- **2.7 决策树与随机森林：** [07_decision_trees_random_forest.md](../07_decision_trees_random_forest.md)
- **2.6 SVM与核方法：** [06_svm_kernel_methods.md](../06_svm_kernel_methods.md)
- **2.5 线性/逻辑回归：** [05_linear_logistic_regression.md](../05_linear_logistic_regression.md)

### 相关扩展文档

- **高级优化话题：** [advanced_optimization.md](./advanced_optimization.md)
- **凸分析基础：** [convex_analysis.md](./convex_analysis.md)

### 代码实验

- **决策树与随机森林：** [`code/ch02_optimization/decision_tree_random_forest.py`](../../../code/ch02_optimization/decision_tree_random_forest.py)

### 后续阅读建议

**如果你对以下话题感兴趣，可以继续阅读：**

1. **决策树的实现**
   - 2.7 决策树与随机森林
   - 了解如何在实践中使用树结构

2. **集成学习**
   - 随机森林、梯度提升等集成方法
   - 了解多个树如何组合提高性能

3. **深度学习中的树结构**
   - 第3章：深度学习基础
   - 了解神经网络与树结构的关系

4. **数学基础**
   - 附录A：数学参考
   - 了解更多关于图论、递归等数学概念

---

**返回：** [第2章：优化与机器学习](../README.md)
