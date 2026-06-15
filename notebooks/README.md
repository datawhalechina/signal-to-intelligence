# Jupyter Notebooks - 交互式学习指南

**版本：** v2.1  
**最后更新：** 2026-06-04

本目录包含每章的 Jupyter notebooks，提供交互式学习体验。Notebook 是教程的交互补充，不是算法实现的第二份源头；核心实验逻辑应优先复用 `code/` 下的脚本。

## 快速开始

### 1. 启动Jupyter

```bash
# 进入项目目录
cd signal-to-intelligence

# 启动Jupyter
jupyter notebook
```

### 2. 打开notebook

在浏览器中打开 `notebooks/` 目录，选择要学习的章节。

### 3. 运行代码

- 点击代码单元格
- 按 `Shift + Enter` 运行
- 修改参数，重新运行观察结果

## Notebooks 列表

### 第1章：DSP基础
- **文件**: `ch01_dsp_interactive.ipynb`
- **内容**: FFT频谱分析、位置编码交互式实验
- **学习时间**: 30分钟
- **交互元素**: 频率调整滑块、实时频谱更新

### 第2章：优化与梯度下降
- **文件**: `ch02_optimization_interactive.ipynb`
- **内容**: 梯度下降可视化、优化器对比
- **学习时间**: 30分钟
- **交互元素**: 学习率调整、实时收敛曲线

### 第3章：深度学习快速通道
- **文件**: `ch03_deep_learning_interactive.ipynb`
- **内容**: CNN/RNN结构、模型训练过程
- **学习时间**: 40分钟
- **交互元素**: 网络参数调整、实时训练监控

### 第4章：Transformer详解
- **文件**: `ch04_transformer_interactive.ipynb`
- **内容**: 自注意力机制、多头注意力可视化
- **学习时间**: 40分钟
- **交互元素**: 注意力权重热力图、位置编码分析

### 第5章：LLM基础
- **文件**: `ch05_llm_interactive.ipynb`
- **内容**: Scaling Laws、Tokenization、Prompt工程实验
- **学习时间**: 50分钟
- **交互元素**: Prompt模板、实时API调用

### 第6章：LLM应用与微调
- **文件**: `ch06_llm_applications_interactive.ipynb`
- **内容**: RAG系统演示、Agent框架
- **学习时间**: 60分钟
- **交互元素**: 文档检索、Agent决策过程

### 第7章：多模态LLM
- **文件**: `ch07_multimodal_interactive.ipynb`
- **内容**: 图像处理、视觉-语言对齐
- **学习时间**: 50分钟
- **交互元素**: 图像上传、实时分析

### 第8章：LLM工程实践
- **文件**: `ch08_engineering_interactive.ipynb`
- **内容**: 模型部署、成本优化
- **学习时间**: 40分钟
- **交互元素**: 配置调整、性能基准测试

## 使用技巧

### 0. 复用项目代码
Notebook 通过 `notebooks/project.py` 加载 `code/` 下的脚本：

```python
from pathlib import Path
import os
import sys
import subprocess

if not Path("notebooks/bootstrap.py").exists():
    root = Path("/content/signal-to-intelligence")
    if not root.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/datawhalechina/signal-to-intelligence.git", str(root)],
            check=True,
        )
    os.chdir(root)

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from notebooks.bootstrap import load_code_module

fft_spectrum = load_code_module("code/ch01_dsp/fft_spectrum.py")
```

维护时优先修改 `code/` 中的函数，再让 notebook 调用这些函数；不要在 notebook 中复制一份 LMS、Attention、MLP 等核心算法。

### 1. 修改参数
大多数notebooks都有可调参数。修改后重新运行单元格查看结果变化。

```python
# 例如，修改这些参数
LEARNING_RATE = 0.01  # 改为 0.001 或 0.1
BATCH_SIZE = 32       # 改为 16 或 64
EPOCHS = 100          # 改为 50 或 200
```

### 2. 保存结果
```python
# 保存图表
plt.savefig('my_result.png', dpi=150, bbox_inches='tight')

# 保存数据
np.save('my_data.npy', data)
```

### 3. 调试代码
```python
# 添加调试输出
print(f"变量值: {variable}")
print(f"形状: {array.shape}")

# 使用 pdb 调试
import pdb; pdb.set_trace()
```

### 4. 性能优化
- 减少数据量进行快速实验
- 使用 `%timeit` 测量代码性能
- 使用 `%matplotlib notebook` 获得交互式图表

## 常见问题

### Q1: 如何安装依赖？

见 [附录 B：环境配置](../docs/appendix/B_cloud_deployment.md) 和根目录 `pyproject.toml`。

### Q2: 如何在远程服务器上运行？
```bash
# 在服务器上启动
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# 在本地浏览器中访问
http://server_ip:8888
```

### Q3: 如何导出notebook为PDF？
```bash
# 需要先安装 nbconvert 和 pandoc
pip install nbconvert pandoc

# 导出为PDF
jupyter nbconvert --to pdf ch01_dsp_interactive.ipynb
```

### Q4: 如何在notebook中使用GPU？
```python
import torch
print(torch.cuda.is_available())  # 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## 学习路径建议

### 快速入门（2小时）
1. 运行 `ch01_dsp_interactive.ipynb`
2. 修改参数，观察结果
3. 阅读对应的文档

### 标准学习（6小时）
1. 按顺序运行所有notebooks
2. 完成每个notebook中的练习题
3. 修改参数进行实验

### 深度学习（12小时）
1. 运行所有notebooks
2. 修改代码实现新功能
3. 结合文档和论文深入理解

## 贡献指南

如果你创建了新的 notebook 或改进了现有的，欢迎提交 PR。

### Notebook编写规范
- 使用清晰的Markdown标题组织内容
- 为每个代码单元格添加说明
- 包含交互式元素（滑块、下拉菜单等）
- 添加练习题和思考题
- 提供进一步学习的资源链接
- 核心算法和数据生成逻辑放在 `code/`，Notebook 只做参数调整、可视化和演示编排

## 相关资源

- **文档**: `docs/` 目录中的详细说明
- **代码**: `code/` 目录中的完整实现
- **测试**: `tests/` 目录中的单元测试
- **论文**: 各章节推荐的学术论文

---

**下一步**: 打开 `ch01_dsp_interactive.ipynb` 开始学习！
