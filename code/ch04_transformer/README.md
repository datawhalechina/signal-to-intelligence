# 第4章代码结构

本目录将 Transformer 教学代码分为两层：

```text
attention_core.py          纯数值计算：softmax、注意力、Mask、归一化、FFN
self_attention.py          教学演示：构造输入、运行实验、打印摘要、生成图表
*_demo.py                  其他独立概念演示：缩放、因果 Mask、图论等
```

`attention_core.py` 是第 4 章共享计算逻辑的源头，不包含绘图、打印或文件写入。现有 `self_attention.py` 保留原有公开函数和运行路径，以兼容文档、Notebook 和已有测试；它内部调用核心模块。

后续章节项目应优先调用 `attention_core.py` 的函数，并把数据准备、评估和可视化放在项目自己的目录中。

试点质量检查：

```bash
make lint-ch04
make test
```
