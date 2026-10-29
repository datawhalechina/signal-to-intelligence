# 附录B：Docker 与 CNB 容器化部署指南

**对标：** 第0章第4小节 — [工具链选择决策表](../00_introduction/04_tools_and_infrastructure.md#工具链选择决策表)

---

## 快速导航

| 我想... | 跳转位置 | 预计时间 |
|--------|---------|---------|
| 理解 Docker vs CNB | [Docker vs CNB 对比](#docker-vs-cnb-对比) | 10 min |
| 快速容器化 LLM 应用 | [快速开始：Docker 化](#快速开始docker-化-llm-应用) | 20 min |
| 完整 Dockerfile 教程 | [Dockerfile 详细讲解](#dockerfile-详细讲解) | 30 min |
| 部署到云平台 | [云平台部署](#云平台部署) | 30 min |
| CNB 使用指南 | [CNB（Cloud Native Buildpacks）](#cnb-cloud-native-buildpacks) | 15 min |
| 常见问题 | [常见问题与解决](#常见问题与解决) | As needed |

---

## 前置说明

> **参考第0章第4小节：** [工具链决策表](../00_introduction/04_tools_and_infrastructure.md#工具链选择决策表)
>
> 本章补充：如何将 LLM 应用容器化，部署到云平台，以及 Docker 和 CNB 的详细讲解。

---

## Docker vs CNB 对比

### 核心区别

| 维度 | Docker | CNB |
|------|--------|-----|
| **学习难度** | 中等 | 低 |
| **配置方式** | Dockerfile | 自动检测 |
| **控制力** | 高（手动配置） | 低（自动化） |
| **镜像大小** | 可控（取决于Dockerfile） | 通常较大 |
| **速度** | 快（手工优化） | 中等（自动优化） |
| **维护成本** | 高（需维护Dockerfile） | 低（自动更新buildpack） |
| **云平台支持** | 通用 | Heroku / Google Cloud / IBM |
| **生产就绪** | ✅ | ✅ |

### 选择建议

```
我想要什么？

├─ 完全控制，自己优化
│  └─ 选择：Docker
│
├─ 快速部署，不想配置
│  └─ 选择：CNB
│
├─ 部署到特定云平台（Heroku/GCP）
│  └─ 选择：CNB
│
└─ 需要在多个平台运行
   └─ 选择：Docker（更通用）
```

---

## Docker 基础

### Docker 是什么？

Docker = **轻量级虚拟机**，包含应用所有依赖

```
传统部署：
代码 → 依赖 → 环境变量 → 手动配置
问题：依赖版本不一致，"在我机器上能跑"

Docker 部署：
代码 + 依赖 + 环境变量 → 容器镜像
好处：任何地方都能一致运行
```

### Docker 核心概念

| 概念 | 说明 | 类比 |
|------|------|------|
| **Image（镜像）** | 应用的蓝图 | 蓝图 |
| **Container（容器）** | 运行的实例 | 房子 |
| **Registry（仓库）** | 镜像存储地 | 图书馆 |
| **Dockerfile** | 镜像构建脚本 | 施工说明 |

---

## Dockerfile 详细讲解

### 基础 Dockerfile 结构

```dockerfile
# 1. 基础镜像（必须）
FROM python:3.11-slim

# 2. 工作目录
WORKDIR /app

# 3. 复制文件
COPY requirements.txt .
COPY . .

# 4. 安装依赖
RUN pip install -r requirements.txt

# 5. 暴露端口
EXPOSE 8000

# 6. 启动命令
CMD ["python", "app.py"]
```

### 各指令详解

#### FROM（基础镜像）

```dockerfile
# 选择基础镜像的原则：
# 1. 使用官方镜像（安全）
# 2. 使用 -slim 版本（更小）
# 3. 指定具体版本号（可重现）

FROM python:3.11-slim      # 推荐
FROM python:3.11           # 可以，但镜像更大
FROM python                 # 不推荐（版本不固定）
FROM nvidia/cuda:12.1       # GPU 支持
```

#### WORKDIR（工作目录）

```dockerfile
# 所有后续命令都在这个目录下执行
WORKDIR /app

# 等价于
RUN mkdir -p /app && cd /app
```

#### COPY vs ADD

```dockerfile
# COPY：推荐用于复制本地文件
COPY requirements.txt .
COPY src/ ./src/

# ADD：支持远程 URL，但通常不推荐
ADD https://example.com/file.tar.gz .
```

#### RUN（执行命令）

```dockerfile
# 安装依赖
RUN pip install -r requirements.txt

# 多个命令用 && 连接（减少镜像层数）
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean

# 不好的做法：每个 RUN 创建一个新层
RUN apt-get update
RUN apt-get install -y git
RUN apt-get clean
```

#### EXPOSE（暴露端口）

```dockerfile
# 声明应用使用的端口（仅文档作用）
EXPOSE 8000

# 运行时实际需要 -p 参数映射
# docker run -p 8000:8000 myapp
```

#### CMD vs ENTRYPOINT

```dockerfile
# CMD：默认启动命令（可被覆盖）
CMD ["python", "app.py"]

# 运行时可以覆盖
docker run myapp /bin/bash

# ENTRYPOINT：强制启动命令（不可被覆盖）
ENTRYPOINT ["python", "app.py"]

# 结合使用
ENTRYPOINT ["python"]
CMD ["app.py"]
# 可以 docker run myapp other_script.py
```

#### ENV（环境变量）

```dockerfile
# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV API_KEY=default_value

# 在容器中可以访问
# 也可以在 docker run -e API_KEY=xxx 时覆盖
```

---

## LLM 应用 Docker 化实例

### 示例1：Ollama 服务容器化

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# 安装 Ollama
RUN apt-get update && apt-get install -y curl && \
    curl https://ollama.ai/install.sh | sh

EXPOSE 11434

# 启动 Ollama 服务
CMD ["ollama", "serve"]
```

**构建和运行：**

```bash
# 构建镜像
docker build -t ollama-server .

# 运行容器（GPU 支持）
docker run --gpus all -p 11434:11434 ollama-server

# 从容器内拉取模型
docker exec -it <container_id> ollama pull qwen:7b
```

### 示例2：Python LLM 应用

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean

# 复制代码
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models

# 暴露 API 端口
EXPOSE 8000

# 启动应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**requirements.txt：**

```txt
fastapi==0.104.1
uvicorn==0.24.0
torch==2.1.0
transformers==4.34.0
ollama==0.0.20
```

**main.py：**

```python
from fastapi import FastAPI
import ollama

app = FastAPI()

@app.post("/chat")
async def chat(prompt: str):
    response = ollama.generate(
        model="qwen:7b",
        prompt=prompt,
        stream=False
    )
    return {"response": response["response"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**构建和测试：**

```bash
# 构建镜像
docker build -t llm-api .

# 运行容器
docker run -p 8000:8000 llm-api

# 测试 API
curl -X POST http://localhost:8000/chat -d "prompt=Hello"
```

---

## Docker 最佳实践

### 1. 多阶段构建（减小镜像大小）

```dockerfile
# 阶段1：构建
FROM python:3.11 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# 阶段2：运行（只复制必要的文件）
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

CMD ["python", "app.py"]
```

### 2. 缓存优化

```dockerfile
# 不好：每次修改代码都重新安装依赖
COPY . .
RUN pip install -r requirements.txt

# 好：依赖改变才重新安装
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

### 3. .dockerignore 文件

```
# .dockerignore
__pycache__
*.pyc
.git
.env
node_modules
.pytest_cache
models/  # 不要包含大模型文件
```

---

## CNB（Cloud Native Buildpacks）

### CNB 是什么？

CNB = **自动化的容器构建系统**

特点：
- 无需编写 Dockerfile
- 自动检测应用类型
- 自动应用最佳实践

### CNB 快速开始

```bash
# 1. 安装 pack
curl https://github.com/buildpacks/pack/releases/download/v0.33.0/pack-v0.33.0-linux.tgz | tar xvz

# 2. 构建应用（自动）
pack build llm-app --builder heroku/builder:22

# 3. 运行容器
docker run -p 8000:8000 llm-app
```

### 对比：Docker vs CNB

**Docker 方式：**
```bash
# 需要写 Dockerfile
docker build -t myapp .
docker run myapp
```

**CNB 方式：**
```bash
# 无需 Dockerfile，自动检测
pack build myapp
docker run myapp
```

---

## 云平台部署

### Heroku（支持 CNB）

```bash
# 1. 安装 Heroku CLI
curl https://cli.heroku.com/install.sh | sh

# 2. 登录
heroku login

# 3. 创建应用
heroku create my-llm-app

# 4. 部署
git push heroku main

# 5. 查看日志
heroku logs --tail
```

### Google Cloud Run（支持 Docker）

```bash
# 1. 构建镜像
docker build -t gcr.io/PROJECT_ID/llm-app .

# 2. 推送到 Google Container Registry
docker push gcr.io/PROJECT_ID/llm-app

# 3. 部署到 Cloud Run
gcloud run deploy llm-app \
    --image gcr.io/PROJECT_ID/llm-app \
    --platform managed \
    --region us-central1 \
    --memory 4Gi \
    --cpu 2
```

### AWS（支持 Docker）

```bash
# 1. 登录到 ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# 2. 构建并推送
docker build -t llm-app .
docker tag llm-app:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/llm-app:latest
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/llm-app:latest

# 3. 部署到 ECS 或 Lambda
# （配置过程较复杂，建议查看官方文档）
```

---

## Docker 常用命令

```bash
# 构建镜像
docker build -t myapp:1.0 .

# 列出镜像
docker images

# 运行容器
docker run -p 8000:8000 myapp:1.0

# 进入容器
docker exec -it <container_id> /bin/bash

# 查看容器日志
docker logs -f <container_id>

# 停止容器
docker stop <container_id>

# 删除容器
docker rm <container_id>

# 推送镜像到仓库
docker push myrepo/myapp:1.0

# 拉取镜像
docker pull myrepo/myapp:1.0
```

---

## 常见问题与解决

### 问题1：镜像太大

**症状：** 构建的镜像 > 5GB

**解决方案：**

```dockerfile
# 1. 使用 -slim 基础镜像
FROM python:3.11-slim  # 而不是 python:3.11

# 2. 清理包管理器缓存
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3. 使用多阶段构建
# 见上面的示例

# 4. 不要复制大文件到镜像
# 模型文件应该在运行时下载或挂载
```

### 问题2：容器无法访问 GPU

**症状：** `nvidia-smi` 在容器内不可用

**解决方案：**

```bash
# 1. 确保安装了 nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2

# 2. 重启 Docker
sudo systemctl restart docker

# 3. 使用 --gpus 参数运行
docker run --gpus all myapp
```

### 问题3：容器启动缓慢

**症状：** 第一次启动需要 5+ 分钟

**原因和解决方案：**

```dockerfile
# 原因1：每次启动都下载模型
# 解决：在 Dockerfile 中预下载模型
RUN python -c "from transformers import AutoModel; \
               AutoModel.from_pretrained('model_name')"

# 原因2：每次都重新编译依赖
# 解决：使用预编译的 wheel 包
RUN pip install --no-build-isolation \
    --find-links /wheels -r requirements.txt
```

---

## 安全最佳实践

```dockerfile
# 1. 不要以 root 运行
RUN useradd -m appuser
USER appuser

# 2. 不要包含敏感信息
# 错误做法：
# ENV API_KEY=12345

# 正确做法：运行时注入
# docker run -e API_KEY=12345 myapp

# 3. 扫描镜像漏洞
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy image myapp:latest
```

---

## 相关链接

**第0章参考：**
- [第0章：工具链决策表](../00_introduction/04_tools_and_infrastructure.md)

**其他附录：**
- [附录B：本地推理指南](B_local_inference.md)
- [附录B：模型获取指南](B_model_sources.md)
- [附录B：环境配置](B_environment_setup.md)

**外部资源：**
- Docker 官方文档：https://docs.docker.com
- CNB 官方文档：https://buildpacks.io
- Heroku 部署：https://devcenter.heroku.com
