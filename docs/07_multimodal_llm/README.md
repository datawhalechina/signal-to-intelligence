# 第7章：多模态LLM

## 章节概览

本章回答一个核心问题：**如何把图像、音频和传感器等不同信号表示，与语言建立可验证的对齐关系？**

![多模态 LLM 系统主线](/diagrams/ch07-multimodal-system-map.svg)

*章节主图：图像、音频、振动或其他传感器观测先经过与任务匹配的表示和编码，再与语言对齐并进入检索、预测或生成接口；表示质量与部署成本需要联合验收。*

本章主线是跨模态表示设计：**信号表示 → 模态编码 → 跨模态对齐 → 融合与语言接口 → 任务验收**。ViT、CLIP、LLaVA 和 Qwen-VL 用于解释视觉主干；音频、振动和传感器信号用于说明同一套思想如何扩展到非视觉观测。具体模型只用于分析设计取舍，不作为固定推荐列表。

## 小节目录

### 表示与对齐（7.1-7.2）

**7.1 视觉-语言对齐** — [📖 阅读](01_vision_language.md)
- ViT：图像分块 + Transformer 编码
- CLIP：对比学习实现图像-文本对齐
- 对齐的几何直观：相似对靠近，不相似对远离
- 对齐失败：细粒度文字、空间关系和长尾视觉概念

**7.2 非视觉信号—语言对齐** — [📖 阅读](02_signal_language_alignment.md)
- 从波形、频谱、时频图和传感器窗口得到信号 token
- 双编码器、序列接口和确定性工具三种接入方式
- 信号侧、对齐侧、语言侧和系统侧的分层验收

### 架构与技术细节（7.3-7.4）

**7.3 高分辨率多模态架构** — [📖 阅读](02_qwen_vl.md)
- LLaVA 架构（单编码器 + 投影层 + LLM）
- 以 Qwen2.5-VL 为例理解高分辨率、多图像、中文优化等设计
- 混合融合策略 vs 早期/晚期融合
- 模型案例背后的通用问题：视觉 token 如何进入语言模型

**7.4 高分辨率图像处理** — [📖 阅读](03_high_resolution.md)
- 为什么标准 224×224 不够用
- 三种方案：动态分辨率、图像分块、自适应采样
- 方案权衡：细节保留、token 成本、延迟和上下文占用

### 应用与评测（7.5-7.6）

**7.5 多模态应用** — [📖 阅读](04_applications.md)
- 图像问答、文档理解、图表分析
- 模型选择框架
- 任务边界、输入质量和失败模式

**7.6 实战案例** — [📖 阅读](05_case_studies.md)
- 文档理解、图表分析、多语言应用的完整示例
- 从任务约束反推模型、分辨率和后处理策略

## 核心问题

完成本章后，你应该能回答：

1. ViT 如何将图像转换为 Transformer 可处理的序列？
2. CLIP 的对比学习如何实现视觉—语言对齐？
3. 音频、振动和传感器信号如何形成 token 并与文本对齐？
4. 双编码器、序列接口和确定性工具分别适合什么任务？
5. 高分辨率和长信号输入怎样影响表示质量与 token 成本？
6. 如何同时评估信号保真、跨模态对齐、语言输出和部署资源？

## 本章学习闭环

1. 先根据章节主图定位观测、表示、编码、对齐和语言接口；
2. 用 ViT patch 或第 1 章时频实验比较视觉与非视觉信号怎样形成表示单元；
3. 完成最低实践任务，记录表示数量、保留的信息、对齐目标和计算代价；
4. 根据实验结果说明输入细节、对齐质量与端侧资源之间的取舍。

## 最低实践任务

从图像 patch 或一维信号时频窗口中任选一种表示，改变 patch / 窗口大小，比较表示单元数量和信息损失；再为输入定义文本标签或事实描述，并给出信号侧、对齐侧和系统侧各一个指标。完成标准：能说明观测如何变成 token、怎样与语言对齐，以及压缩表示会牺牲什么。

## 代码实验

本章复用第 1 章时频实验，并提供 **9 个第 7 章代码脚本**，覆盖 ViT、CLIP、高分辨率处理、多模态应用和案例评测。

| 小节 | 脚本 | 生成图表 | 内容 |
|------|------|---------|------|
| 7.1 | [`architecture_diagrams.py`](../../code/ch07_multimodal_llm/architecture_diagrams.py) | `ch07_vit_cnn_comparison.png`<br>`ch07_temperature_effect.png` | CNN vs ViT 感受野对比、温度参数效果 |
| 7.2 | [`time_freq_analysis.py`](../../code/ch01_dsp/time_freq_analysis.py) | `ch01_time_frequency.png` | 非视觉信号的时频表示与窗口选择 |
| 7.3/7.4 | [`explainer_diagrams.py`](../../code/ch07_multimodal_llm/explainer_diagrams.py) | `ch07_architecture_comparison.png`<br>`ch07_fusion_strategies.png`<br>`ch07_dynamic_resolution.png`<br>`ch07_image_tiling.png` | LLaVA vs Qwen2.5-VL 对比、融合策略、动态分辨率、分块示意 |
| 7.1 | [`vit_patches.py`](../../code/ch07_multimodal_llm/vit_patches.py) | `ch07_vit_patches.png` | ViT 图像分块可视化 |
| 7.1 | [`clip_similarity.py`](../../code/ch07_multimodal_llm/clip_similarity.py) | `ch07_clip_similarity.png` | CLIP 图像-文本相似度矩阵 |
| 7.1 | [`clip_alignment_demo.py`](../../code/ch07_multimodal_llm/clip_alignment_demo.py) | `ch07_clip_alignment.png` | CLIP 对齐演示（需 `openai-clip`） |
| 7.3 | [`qwen_vl_analysis.py`](../../code/ch07_multimodal_llm/qwen_vl_analysis.py) | `ch07_qwen_vl_analysis.png` | 图像尺寸与视觉 token 预算 |
| 7.4 | [`high_resolution_processing.py`](../../code/ch07_multimodal_llm/high_resolution_processing.py) | `ch07_high_resolution_processing.png` | 三种高分辨率方案对比 |
| 7.5 | [`multimodal_applications.py`](../../code/ch07_multimodal_llm/multimodal_applications.py) | `ch07_multimodal_applications.png` | 应用指标与失败模式 |
| 7.6 | [`case_studies.py`](../../code/ch07_multimodal_llm/case_studies.py) | `ch07_case_studies.png` | 三个案例的处理流程与验收指标 |

**运行方式：**
```bash
python code/ch07_multimodal_llm/architecture_diagrams.py
python code/ch07_multimodal_llm/explainer_diagrams.py
python code/ch01_dsp/time_freq_analysis.py
python code/ch07_multimodal_llm/vit_patches.py
python code/ch07_multimodal_llm/clip_similarity.py
python code/ch07_multimodal_llm/qwen_vl_analysis.py
python code/ch07_multimodal_llm/high_resolution_processing.py
python code/ch07_multimodal_llm/multimodal_applications.py
python code/ch07_multimodal_llm/case_studies.py
```

> `clip_alignment_demo.py` 需额外安装 `openai-clip` 包（见 [附录B：环境配置](../appendix/B_environment_setup.md)）。


## 扩展内容

### 多模态对齐训练深度细节 — [📖 阅读](extensions/multimodal_training_details.md)
- InfoNCE 损失完整推导与温度参数 τ 的分析
- 批大小对负样本数量的影响（256 vs 4096 vs 32768）
- 投影层设计：MLP vs Q-Former 对比
- 三阶段训练策略（冻结 ViT → 指令微调 → 全参数微调）

### 多模态应用进阶 — [📖 阅读](extensions/multimodal_applications_advanced.md)
- 多模态 RAG：双路检索（CLIP 图像向量 + 文本向量）与 RRF 融合
- OCR + LLM 混合文档理解技术栈
- 视觉 Agent 工具调用模式
- 模型选择决策树与常见坑（CLIP 模板、OOM、多图混乱）

## 关键连接点

### 第4章 Transformer → 第7章 ViT

```
第4章：Transformer 处理文本序列（token 序列）
    ↓ 同样的架构
第7章：ViT 处理图像（patch 序列）
    图像 → 分块 → Patch 嵌入 → Transformer
```

### 第5章 Embedding → 第7章 CLIP

```
第5章：token embedding（离散符号 → 连续向量）
    ↓ 同样的思想，扩展到跨模态
第7章：CLIP（图像 → 向量，文本 → 向量，对齐到同一空间）
```

### 第1章信号表示 → 第7章信号—语言对齐 → 第8章端侧验收

```text
第1章：采样、时频变换、检测与估计
    ↓ 把任务所需结构变成可学习表示
第7章：信号编码、跨模态对齐与语言接口
    ↓ 在资源约束下压缩表示和模型
第8章：用信号、对齐、任务和系统指标联合验收
```

---

**下一步：** 阅读 [7.1 视觉-语言对齐](01_vision_language.md)
