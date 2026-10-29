# API 配置指南

本指南说明如何配置 LLM API，以运行第 5 章的 `llm_api_demo.py`。

> **前置条件：** 已完成 [快速开始](B_environment_setup.md)  
> **需要的命令：** `pip install -e ".[llm]"`

---

## API 提供商对比

| 提供商 | 模型 | 特点 | 地址 | 国内访问 |
|--------|------|------|------|--------|
| **DeepSeek** | 以控制台当前列表为准 | OpenAI 兼容接口 | [platform.deepseek.com](https://platform.deepseek.com) | 以所在地区实测为准 |
| **阿里云百炼** | 以控制台当前列表为准 | Qwen 等模型服务 | [bailian.aliyun.com](https://bailian.aliyun.com) | 以所在地区实测为准 |
| **智谱 AI** | 以控制台当前列表为准 | GLM 等模型服务 | [open.bigmodel.cn](https://open.bigmodel.cn) | 以所在地区实测为准 |
| Anthropic | 以控制台当前列表为准 | 托管 API | [console.anthropic.com](https://console.anthropic.com) | 核对服务地区与条款 |
| OpenAI | 以控制台当前列表为准 | 托管 API | [platform.openai.com](https://platform.openai.com) | 核对服务地区与条款 |

选择前核对模型版本、价格、数据保留政策、服务地区和代码示例所需的接口格式。

---

## 获取 API 密钥

### DeepSeek

1. 访问 [platform.deepseek.com](https://platform.deepseek.com)
2. 注册账户
3. 创建 API Key
4. 复制密钥

### 阿里云百炼

1. 访问 [bailian.aliyun.com](https://bailian.aliyun.com)
2. 注册或登录
3. 创建 API Key
4. 复制密钥

### 智谱 AI

1. 访问 [open.bigmodel.cn](https://open.bigmodel.cn)
2. 注册或登录
3. 创建 API Key
4. 复制密钥

### OpenAI / Anthropic

按照服务商当前支持的地区、账户和支付要求配置。

---

## 设置环境变量

### 方式 1：临时设置（当前终端）

```bash
# DeepSeek
export DEEPSEEK_API_KEY="your-key-here"

# 阿里云百炼
export DASHSCOPE_API_KEY="your-key-here"

# 智谱 AI
export ZHIPUAI_API_KEY="your-key-here"

# OpenAI
export OPENAI_API_KEY="your-key-here"

# Anthropic
export ANTHROPIC_API_KEY="your-key-here"
```

然后运行实验：
```bash
python code/ch05_llm_basics/llm_api_demo.py
```

### 方式 2：永久设置

**Linux / macOS：**
```bash
# 编辑 ~/.bashrc 或 ~/.zshrc
echo 'export DEEPSEEK_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**Windows (PowerShell)：**
```powershell
$env:DEEPSEEK_API_KEY="your-key-here"
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "your-key-here", "User")
```

### 方式 3：.env 文件（推荐）

在项目根目录创建 `.env`：
```bash
DEEPSEEK_API_KEY=your-key-here
DASHSCOPE_API_KEY=your-key-here
ZHIPUAI_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
```

然后在 Python 中加载：
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 运行第 5 章 API 实验

### 使用 DeepSeek

```bash
export DEEPSEEK_API_KEY="your-key-here"
python code/ch05_llm_basics/llm_api_demo.py
```

### 使用阿里云百炼

```bash
export DASHSCOPE_API_KEY="your-key-here"
python code/ch05_llm_basics/llm_api_demo.py
```

### 使用智谱 AI

```bash
export ZHIPUAI_API_KEY="your-key-here"
python code/ch05_llm_basics/llm_api_demo.py
```

### 使用 OpenAI

```bash
export OPENAI_API_KEY="your-key-here"
python code/ch05_llm_basics/llm_api_demo.py
```

---

## 代码示例

### 调用 DeepSeek API

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-deepseek-key",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "你好，请介绍一下 DSP"}
    ]
)

print(response.choices[0].message.content)
```

### 调用阿里云百炼 API

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-dashscope-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "user", "content": "你好，请介绍一下 DSP"}
    ]
)

print(response.choices[0].message.content)
```

---

## 常见问题

### API 调用超时？

检查网络连接。国内用户可以尝试：
- 使用国内 API（DeepSeek / 阿里云百炼 / 智谱）
- 检查是否需要代理
- 增加超时时间

### API Key 无效？

- 确认 Key 已复制完整（无多余空格）
- 确认 Key 未过期
- 在提供商网站验证 Key 是否有效

### 没有 API 预算？

尝试免费方案：
- **阿里云百炼：** 有新用户免费额度
- **智谱 AI：** 有免费额度
- **本地 Ollama：** 完全免费（见 [本地推理指南](B_local_inference.md)）

### 想用本地模型而不是 API？

见 [本地推理指南](B_local_inference.md) 的 Ollama 部分。

---

## 获取帮助

- 📖 [快速开始](B_environment_setup.md)
- 🎮 [本地推理指南](B_local_inference.md)
- 🧪 [代码运行指南](C_code_guide.md)
- 📚 [完整导航](INDEX.md)
