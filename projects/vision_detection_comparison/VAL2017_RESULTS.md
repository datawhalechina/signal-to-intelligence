# COCO val2017 固定 50 张教学基准

这份记录用于验证 YOLO 与 DETR 的统一对比方法，并给教程提供可复现的案例。它是单类别、小样本、单次 CPU 运行的教学基准，不是 COCO 官方排行榜结果。

## 实验配置

- 日期：2026-10-01
- 数据：按图像 ID 升序选择的前 50 张含 `person` 的 COCO val2017 图片
- 标签：Ultralytics 官方 `coco2017labels.zip` 中的检测标签
- 真实框：249 个 `person` 框
- 模型：YOLOv8n、`facebook/detr-resnet-50@main`
- 权重：官方预训练权重；DETR 使用 safetensors
- 设备：CPU
- 输入：YOLO 固定 `imgsz=640`；DETR 使用 checkpoint 自带预处理配置
- 评价：单类别 AP@0.5 与 PR 曲线；候选收集下限 0.05
- 计时：排除磁盘读取和模型加载，包含预处理、前向与后处理；第一张图片预热一次
- 环境：Python 3.11.5、PyTorch 2.5.1、Transformers 4.57.6、Ultralytics 8.4.170

## 本机结果

| 模型 | AP@0.5 | 最佳 F1 | 最佳点 P / R | 平均延迟 | 中位延迟 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 0.724 | 0.739 | 0.764 / 0.715 | 17.3 ms | 15.5 ms | 26.3 ms |
| DETR ResNet-50 | 0.814 | 0.770 | 0.787 / 0.755 | 721.5 ms | 726.4 ms | 903.2 ms |

最佳 F1 对应阈值分别为 YOLO 0.226、DETR 0.850。这个差异再次说明：不同模型的原始置信度不能直接横向比较。

在 Recall 约为 0.5 的可比工作点上：

| 模型 | 阈值 | Precision | Recall | TP / FP |
|---|---:|---:|---:|---:|
| YOLOv8n | 0.545 | 0.926 | 0.502 | 125 / 10 |
| DETR ResNet-50 | 0.979 | 0.955 | 0.514 | 128 / 6 |

在当前 0.05 候选收集下限内，YOLO 最高 Recall 为 0.799，未达到脚本定义的 Recall=0.8 工作点；DETR 在阈值 0.756 时达到 Recall 0.807、Precision 0.703。

## 如何解释

1. 在这 50 张图片、249 个目标上，DETR 的 AP 与最佳 F1 较高，PR 曲线在大部分召回区间也更靠上。
2. YOLOv8n 的本机平均延迟约为 DETR ResNet-50 的 1/42，体现了小模型 CNN 检测器在 CPU 部署上的明显速度优势。
3. 这不是“谁绝对更好”的结论，而是精度—速度权衡的实例：模型选择取决于漏检成本、误检成本、硬件预算和延迟目标。
4. 0.05 下的最终 Precision 主要描述候选池，不是推荐部署精度；模型比较应优先看完整 PR 曲线或同 Recall 工作点。

## 可复现性与边界

- 子集由 `prepare_coco_val.py` 确定性生成，选中图像 ID 和全部 PR 点记录在运行产生的 `metrics.json` 中。
- 本次原始 `metrics.json` 的 SHA-256 为 `701b9545817a32ec86bed83a5752cbcbbd2761f29fb8d73e902537524762b7ff`。
- Ultralytics YOLO 标签不包含原始 COCO JSON 的全部元数据；本实验不能替代 COCO API 的多 IoU、多尺度 mAP。
- 两次运行得到相同质量指标；平均延迟分别为 YOLO 16.4/17.3 ms、DETR 658.0/721.5 ms，说明系统负载会带来可见波动。发布稳定性能结论前仍应增加轮次并报告分布。
- 固定取最低图像 ID 有利于复现，但不保证覆盖小目标、遮挡、密集场景等各类难例。

复现数据准备与运行命令见项目 [`README.md`](README.md)。每次运行会在输出目录生成完整 `metrics.json`、`precision_recall_curve.png`、`latency_comparison.png` 和两组错误案例图；错误图使用各模型最佳 F1 工作点，而不是 0.05 候选收集下限。
