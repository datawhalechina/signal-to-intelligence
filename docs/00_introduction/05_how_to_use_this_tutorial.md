# 如何使用本教程

本教程有四种使用方式。默认入口是在线阅读 VitePress；代码、Notebook 和云端环境用于补充实验和复现。

---

## 推荐入口

| 使用方式 | 适合场景 | 主要入口 |
|---------|----------|----------|
| 在线阅读 VitePress | 系统学习正文、查阅章节结构 | `docs/` 发布后的文档站 |
| 本地运行代码 | 复现实验、生成图片、修改参数 | `code/`、`tests/` |
| Notebook 交互实验 | 教学演示、边改参数边观察结果 | `notebooks/` |
| CNB 云端环境 | 不想配置本地环境、需要云端算力 | 云端配置说明 |

正文以 VitePress 文档为准；核心代码以 `code/` 为准；Notebook 是交互补充。

---

## 如何理解本教程的严谨性

本教程不是严格数学教材，而是面向工程师的概念迁移与实践教程。

正文追求工程自洽：不要求给出所有定理证明，但要求概念不误导、类比有边界、工程结论说明适用条件。扩展章节用于补充理论背景、公式推导和进一步阅读。代码实验用于验证关键现象，帮助读者把抽象概念落到可运行的实现上。

| 层级 | 作用 | 阅读方式 |
|------|------|----------|
| 正文 | 建立直觉、概念迁移和工程判断 | 先读，抓住主线 |
| 扩展 | 补充数学背景、理论细节和局限 | 按需深入 |
| 代码实验 | 验证现象、观察参数影响 | 边运行边理解 |

---

## 章节定位与运行边界

本教程按章节的默认设计是“正文可读、代码可跑、重型内容可选”。默认实验尽量保持离线、轻量和可复现，API、本地推理和大模型权重只作为按需增强能力。

| 章节 | 默认定位 | 说明 |
|------|----------|------|
| 第1-4章 | 离线基础实验 | 以 `numpy` / `matplotlib` / `scipy` 和模拟数据为主 |
| 第5章 | 离线基础 + 可选 API / 本地推理 | 默认 8 个离线脚本；API 演示按需启用 |
| 第6章 | 离线应用模拟 | 默认和扩展脚本都偏轻量，不依赖真实 LLM |
| 第7章 | 离线多模态入门 + 可选重型实验 | 默认 6 个离线脚本；重度脚本需要模型权重 |
| 第8章 | 标准库工程实践 | 不默认依赖 API、GPU 或模型下载 |

更细的脚本清单、依赖和运行命令见 [附录 C：代码运行指南](../appendix/C_code_guide.md)。模型、数据集和维护约定见 [附录 E：项目维护说明](../appendix/E_project_maintenance.md)。

---

## 方式一：在线阅读 VitePress

适合：

- 只想系统学习内容
- 不急着运行代码
- 用浏览器、手机或平板阅读

在线文档由 GitHub Pages workflow 构建。正文来自 `docs/`，图片来自 `assets/` 同步到 `docs/public/assets/` 后的发布产物。

本地预览：

```bash
npm run docs:dev
```

构建检查：

```bash
npm run docs:build
```

---

## 方式二：本地运行代码

适合：

- 复现实验结果
- 重新生成图表
- 修改脚本参数
- 检查教程中的代码是否仍然可运行

基本命令：

```bash
pip install -e .
python code/ch01_dsp/fft_spectrum.py
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -q
```

更完整的代码运行说明见 [附录 C：代码运行指南](../appendix/C_code_guide.md)。

---

## 方式三：Notebook 交互实验

Notebook 的定位：

- 是交互实验补充
- 不作为教程正文的第二份源头
- 不复制完整章节解释
- 核心算法优先调用 `code/` 中的函数

适合：

- 教学演示
- 边改参数边看图
- 在本地 Jupyter、Colab 或 CNB 中逐步执行

启动：

```bash
jupyter notebook
```

Notebook 通过 `notebooks/project.py` 加载 `code/` 下的脚本，避免一份算法同时维护在 `.py`、测试和 `.ipynb` 里。

### 章节 Notebook 索引

每章都有对应的交互式 notebook。推荐先读正文，再打开对应 notebook 做交互实验。

| 章节 | Notebook 文件 | Colab 入口 |
|------|--------------|-----------|
| 第1章 | `notebooks/ch01_dsp_interactive.ipynb` | [打开](https://colab.research.google.com/github/datawhalechina/signal-to-intelligence/blob/main/notebooks/ch01_dsp_interactive.ipynb) |
| 第2章 | `notebooks/ch02_optimization_interactive.ipynb` | [打开](https://colab.research.google.com/github/datawhalechina/signal-to-intelligence/blob/main/notebooks/ch02_optimization_interactive.ipynb) |
| 第3章 | `notebooks/ch03_deep_learning_interactive.ipynb` | [打开](https://colab.research.google.com/github/datawhalechina/signal-to-intelligence/blob/main/notebooks/ch03_deep_learning_interactive.ipynb) |
| 第4章 | `notebooks/ch04_transformer_interactive.ipynb` | [打开](https://colab.research.google.com/github/datawhalechina/signal-to-intelligence/blob/main/notebooks/ch04_transformer_interactive.ipynb) |
| 第5章 | `notebooks/ch05_llm_interactive.ipynb` | [打开](https://colab.research.google.com/github/datawhalechina/signal-to-intelligence/blob/main/notebooks/ch05_llm_interactive.ipynb) |
| 第6章 | `notebooks/ch06_llm_applications_interactive.ipynb` | [打开](https://colab.research.google.com/github/datawhalechina/signal-to-intelligence/blob/main/notebooks/ch06_llm_applications_interactive.ipynb) |
| 第7章 | `notebooks/ch07_multimodal_interactive.ipynb` | [打开](https://colab.research.google.com/github/datawhalechina/signal-to-intelligence/blob/main/notebooks/ch07_multimodal_interactive.ipynb) |
| 第8章 | `notebooks/ch08_engineering_interactive.ipynb` | [打开](https://colab.research.google.com/github/datawhalechina/signal-to-intelligence/blob/main/notebooks/ch08_engineering_interactive.ipynb) |

---

## 方式四：CNB 云端环境

适合：

- 不想配置本地 Python 环境
- 临时运行代码或 Notebook
- 没有本地 GPU
- 想快速复现实验

当前默认原则：

- 第 1-6 章默认不依赖 GPU
- 第 7 章多模态实验以轻量 NumPy 演示为主
- 第 8 章偏工程配置和成本分析，不默认依赖 GPU

云端环境配置后续统一放在附录 B 或独立部署说明中，避免 README、Notebook 和正文重复维护。

---

## 如何选择

| 目标 | 推荐方式 |
|------|----------|
| 先理解整体脉络 | 在线阅读 VitePress |
| 验证公式和图表 | 本地运行 `code/` |
| 课堂演示或调参观察 | Notebook |
| 免配置体验 | CNB 云端环境 |
| 维护项目内容 | 阅读 [附录 E：项目维护说明](../appendix/E_project_maintenance.md) |

如果不确定从哪里开始，先阅读 [学习路径](02_learning_paths.md)，再按章节顺序阅读正文。
