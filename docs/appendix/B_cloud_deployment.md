# 云端部署指南

**版本：** v1.0  
**最后更新：** 2026-06-03

本指南说明如何在云端运行所有实验，适合没有本地 GPU 或不想配置本地环境的用户。

> **优势：** 免费 GPU、无需配置、一键启动

---

## Google Colab（免费，推荐）

### 特点

- ✅ 免费 T4 GPU
- ✅ 每天 4-6 小时使用时间
- ✅ 无需本地配置
- ✅ 适合学习第 1-6 章

### 快速开始

1. 访问 [Google Colab](https://colab.research.google.com)
2. 新建 Notebook
3. 复制下方代码到第一个 Cell 并运行：

```python
# 克隆仓库
!git clone https://github.com/datawhalechina/signal-to-intelligence.git
%cd signal-to-intelligence

# 安装依赖
!pip install -e . -q

# 运行第一个实验
!python code/ch01_dsp/fft_spectrum.py
```

### 完整工作流

```python
# Cell 1: 环境设置
!git clone https://github.com/datawhalechina/signal-to-intelligence.git
%cd signal-to-intelligence
!pip install -e . -q

# Cell 2: 运行实验
!python code/ch01_dsp/fft_spectrum.py

# Cell 3: 查看生成的图表
from IPython.display import Image, display
display(Image('assets/ch01_fft_spectrum.png'))

# Cell 4: 运行测试
!pytest tests/test_ch01_dsp.py -v
```

### 注意事项

- 文件保存在 Colab 会话中，关闭后丢失
- 要永久保存，下载文件或用 Google Drive 挂载
- 长期运行可能被中断（有超时限制）

---

## Kaggle Notebooks（免费，备选）

### 特点

- ✅ 免费 P100 GPU
- ✅ 每周 30 小时使用时间
- ✅ 无需本地配置
- ✅ 比 Colab 时间更长

### 快速开始

1. 访问 [Kaggle](https://www.kaggle.com)
2. 创建新 Notebook
3. 复制下方代码：

```python
# 克隆仓库
!git clone https://github.com/datawhalechina/signal-to-intelligence.git
%cd signal-to-intelligence

# 安装依赖
!pip install -e . -q

# 运行实验
!python code/ch01_dsp/fft_spectrum.py
```

---

## CNB 云原生开发环境（推荐，即将开放）

### 特点

- ✅ 预配置的 Linux + NVIDIA GPU 环境
- ✅ 已安装所有依赖
- ✅ 持久化存储
- ✅ 无需配置

### 待补充内容

CNB 项目空间配置完成后，将提供：
- 一键启动链接
- 环境快照
- 存储配置指南

敬请期待。

---

## Docker（本地隔离环境）

如果想在本地用隔离环境，可用 Docker。

### CPU 版本

```bash
make docker-up
```

### GPU 版本

```bash
make docker-up-gpu
```

### 详见

- `deploy/docker-compose.yml` （CPU）
- `deploy/docker-compose.gpu.yml` （GPU）

---

## 对比表

| 方案 | 免费 GPU | 时限 | 配置 | 推荐指数 |
|------|---------|------|------|--------|
| **Google Colab** | ✅ T4 | 4-6h/天 | 无需 | ⭐⭐⭐⭐⭐ |
| **Kaggle** | ✅ P100 | 30h/周 | 无需 | ⭐⭐⭐⭐ |
| **CNB** | ✅ 多种 | 无限 | 一键 | ⭐⭐⭐⭐⭐ |
| **Docker** | ❌ | 无限 | 中等 | ⭐⭐⭐ |

---

## 常见问题

### Colab 超时怎么办？

Colab 会在无操作 90 分钟后断开连接。
- 定期交互（点击、输入）保持连接
- 改用 Kaggle（30h/周，较少中断）
- 等待 CNB 开放

### 如何在云端保存文件？

**Google Colab：**
```python
# 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 保存文件
!cp assets/ch01_fft_spectrum.png /content/drive/MyDrive/
```

**Kaggle：**
文件自动保存到 Notebook 输出。

### 能运行第 7 章的重度实验吗？

**Google Colab：** 
- T4 GPU 显存 16GB，可运行 7B 模型的量化版（INT4）
- 建议用 Colab 的 A100 升级版（付费）

**Kaggle：**
- P100 显存 16GB，同样限制
- 大模型推理建议用 CNB

### 云端访问本地文件？

不支持。需要：
- 上传文件到云端
- 或用 Google Drive / S3 等存储

---

## 工作流示例：Colab 中学习第 5 章

```python
# Cell 1: 初始化
!git clone https://github.com/datawhalechina/signal-to-intelligence.git
%cd signal-to-intelligence
!pip install -e ".[llm]" -q

# Cell 2: 设置 API（如果用 API）
import os
os.environ['DEEPSEEK_API_KEY'] = 'your-key-here'  # 或其他 API

# Cell 3: 运行离线实验
!python code/ch05_llm_basics/bpe_tokenization.py

# Cell 4: 查看图表
from IPython.display import Image, display
display(Image('assets/ch05_bpe_tokenization.png'))

# Cell 5: 运行 API 实验
!python code/ch05_llm_basics/llm_api_demo.py

# Cell 6: 运行测试
!pytest tests/test_ch05_llm_basics.py -v
```

---

## 获取帮助

- 📖 [快速开始](B_environment_setup.md)
- 🎮 [本地推理指南](B_local_inference.md)
- 📡 [API 配置指南](B_api_configuration.md)
- 🧪 [代码运行指南](C_code_guide.md)
- 📚 [完整导航](INDEX.md)
