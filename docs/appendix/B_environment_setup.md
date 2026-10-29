# 环境配置指南 — 快速开始

本文档覆盖能够完成首次运行的核心路径。

**其他场景？选择下方导航：**
- 📡 **需要 API（DeepSeek/OpenAI/阿里云）** → [API 配置指南](B_api_configuration.md)
- 🎮 **有 GPU / 要本地推理** → [本地推理指南](B_local_inference.md)  
- ☁️ **用云端（Colab/CNB/Docker）** → [云端部署指南](B_cloud_deployment.md)
- 📚 **完整导航** → [附录索引](INDEX.md)

---

## 系统要求

**操作系统** — Linux / macOS / Windows (WSL2)  
**Python** — 3.10+ （推荐 3.11）  
**磁盘** — 500 MB（最小）

检查 Python：
```bash
python3 --version
```

---

## 快速开始

### 步骤 1：选择环境

**A. conda（推荐）**
```bash
conda create -n sti python=3.11 --no-default-packages -y
conda activate sti
pip install --upgrade pip setuptools wheel
```

**B. venv（标准 Python）**
```bash
git clone https://github.com/datawhalechina/signal-to-intelligence.git
cd signal-to-intelligence
python3 -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\Activate.ps1   # Windows
pip install --upgrade pip setuptools wheel
```

### 步骤 2：安装依赖
```bash
pip install -e .
```
✅ **完成** — 可以运行第 1-6 章全部 + 第 7-8 章部分实验

### 步骤 3：验证
```bash
python3 code/ch01_dsp/fft_spectrum.py
make test
```

---

## 接下来

- 📖 **开始学习** → [项目 README](https://github.com/datawhalechina/signal-to-intelligence/blob/main/README.md)
- 🧪 **运行实验** → [代码运行指南](C_code_guide.md)
- ❓ **遇到问题** → [常见问题](#常见问题) / [完整导航](INDEX.md)

---

## 常见问题

### 如何退出虚拟环境？
```bash
deactivate
```

### Windows 权限错误？
以管理员身份运行 PowerShell。

### 重新安装依赖？
```bash
pip install --force-reinstall -e .
```

### 有 GPU / 要用 API / 要云端？
见上方导航链接。

---

## 获取帮助

- 📖 [完整导航](INDEX.md)
- 💻 [本地推理指南](B_local_inference.md)
- 📡 [API 配置指南](B_api_configuration.md)
- ☁️ [云端部署指南](B_cloud_deployment.md)
- 🧪 [代码运行指南](C_code_guide.md)
