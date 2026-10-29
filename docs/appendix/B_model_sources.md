# 附录 B：模型与数据集获取

本指南说明如何选择来源、固定版本、下载并校验模型或数据集。平台、模型标签和镜像状态会变化，具体可用项以服务页面和项目锁定版本为准。

---

## 下载前记录什么

```text
仓库或数据集标识
revision / commit hash / tag
需要的文件与格式
许可证和使用限制
预计磁盘空间
缓存目录
文件校验或完整性检查方式
```

不要用“最新版本”作为可复现标识，也不要只根据模型系列名判断许可证。

## 常用来源

### Hugging Face

Hugging Face Hub 提供模型、数据集、配置和版本历史。下载时固定 `revision`：

```bash
pip install huggingface_hub

huggingface-cli download ORG/REPO \
  --revision COMMIT_OR_TAG \
  --local-dir models/REPO
```

Python 方式：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ORG/REPO",
    revision="COMMIT_OR_TAG",
    local_dir="models/REPO",
)
```

需要授权的模型应先在模型页接受条款，并使用个人访问令牌。不要把令牌写入脚本或提交到仓库。

### ModelScope

ModelScope 提供模型和数据集仓库。使用前核对仓库维护者、revision、许可证和文件列表：

```bash
pip install modelscope
```

```python
from modelscope import snapshot_download

model_dir = snapshot_download(
    "ORG/REPO",
    revision="REVISION",
    cache_dir="models",
)
```

不同平台上的同名模型不一定是同一个 revision，也可能使用不同目录结构，不能只比较名称。

### 其他来源

项目官网、云模型服务和本地模型管理工具也可以提供权重或 API。采用前检查：

- 是否为模型作者或可信镜像；
- 文件是否完整，是否修改过配置或量化方式；
- 是否能固定版本并追溯来源；
- 许可证是否允许当前用途。

---

## 模型文件格式

| 格式 | 常见用途 | 检查项 |
|---|---|---|
| `safetensors` | Transformers 等框架加载 | 分片索引、dtype、配置文件 |
| PyTorch checkpoint | 训练或历史模型 | 来源可信度、加载方式 |
| GGUF | llama.cpp 等本地推理 | 量化类型、上下文与 tokenizer 元数据 |
| ONNX | 跨框架或运行时部署 | opset、动态轴、算子支持 |

模型目录通常还需要 `config.json`、tokenizer 文件、generation 配置和分片索引。只有权重文件未必足以正确加载模型。

## 快速开始：下载模型

### Ollama

Ollama 适合快速试运行其模型库中已有的标签：

```bash
ollama list
ollama pull <model-tag>
ollama show <model-tag>
```

记录实际标签和摘要信息。标签可能更新，不应只在报告中写系列名。

### Hugging Face CLI

下载前先查看文件列表，只获取需要的分片或配置：

```bash
huggingface-cli download ORG/REPO \
  --revision COMMIT_OR_TAG \
  --include "*.json" "*.safetensors" \
  --local-dir models/REPO
```

大文件下载完成后，检查目录大小、分片索引和加载 smoke test。

---

## 数据集获取

Hugging Face Datasets 示例：

```python
from datasets import load_dataset

dataset = load_dataset(
    "ORG/DATASET",
    revision="COMMIT_OR_TAG",
    split="train",
)
print(dataset.features)
print(dataset.num_rows)
```

下载数据时还应记录：

- 数据集 card、citation 与许可证；
- split、configuration 和 revision；
- 样本数、字段 schema 与语言分布；
- 是否包含个人信息、受限内容或评测题；
- 本地清洗和过滤脚本版本。

数据集可以下载不代表可以任意训练、再发布或用于商业产品。

## 量化模型

量化文件名通常包含位宽、分组或方法信息，但命名并不完全统一。选择前确认：

- 推理框架支持的格式和量化类型；
- 模型架构、tokenizer 与 chat template 是否匹配；
- 权重大小和运行时峰值内存；
- 目标任务上的质量回归；
- 文件来自模型作者、量化作者还是第三方镜像。

较低位宽通常减少权重存储，不保证按同比例减少全部显存，也不保证提高当前硬件上的吞吐。

## 下载后的最小校验

```text
1. 保存来源 URL、revision 和许可证
2. 核对下载文件数量与目录大小
3. 加载 tokenizer 与 config
4. 用短输入完成一次推理 smoke test
5. 记录框架版本、dtype、设备和峰值内存
6. 将模型写入项目 registry，而不是提交权重文件
```

本项目的模型与数据集登记约定见[项目维护说明](E_project_maintenance.md)。

## 常见问题

### 下载中断

优先使用支持缓存和断点续传的官方客户端，并保持相同 revision 与目标目录。不要把多个不明来源的分片直接混合。

### 磁盘空间不足

先确认权重分片、下载缓存和加载时临时副本的空间需求。可只下载所需文件、选择经验证的量化版本或清理明确不再使用的缓存。

### 无法访问来源

检查服务地区、账户权限和网络策略。使用镜像时记录镜像地址，并核对 commit、文件清单或哈希；不要默认镜像与上游完全同步。

### 加载失败

依次检查模型架构支持、配置与权重是否同一 revision、分片是否完整、依赖版本和 tokenizer 文件。错误日志应保留具体文件名与版本。

---

**相关文档：** [本地推理](B_local_inference.md) · [环境配置](B_environment_setup.md) · [项目维护](E_project_maintenance.md)
