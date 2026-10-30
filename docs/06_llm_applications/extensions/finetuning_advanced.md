# 扩展：微调工程实践

**所属章节：** [第6章：LLM应用](../README.md)  
**前置阅读：** [6.2 微调：让 LLM 适配你的任务](../02_finetuning.md)  
**原理层内容：** 见 [第5章 E5.x 微调方法调研](../../05_llm_basics/extensions/finetuning_survey.md)

---

## 概览

6.2 节介绍了微调的原理和 LoRA 的数学基础。本节聚焦**工程实践**：如何准备数据、选择工具、配置训练、评估效果。

---

## 数据准备

### 数据格式

SFT 数据最终需要转换成目标模型 chat template 所需的 token 序列。下面两种 JSON 只是常见的中间数据约定，不是所有训练框架都遵循的统一标准：

**Alpaca 格式（单轮）：**
```json
{
  "instruction": "将下面的句子翻译成英文",
  "input": "今天天气很好",
  "output": "The weather is very nice today."
}
```

**ShareGPT 格式（多轮对话）：**
```json
{
  "conversations": [
    {"from": "human", "value": "你好，帮我写一首关于春天的诗"},
    {"from": "gpt", "value": "春风轻抚柳丝长..."},
    {"from": "human", "value": "能改成七言绝句吗？"},
    {"from": "gpt", "value": "春风拂柳绿如烟..."}
  ]
}
```

**选择建议：** 根据训练框架的数据加载器和目标模型 chat template 选择或转换格式；训练前抽查转换后的完整文本、角色标记和 loss mask，而不是只检查 JSON 外观。

### 数据质量检查

微调数据的质量比数量重要，上线前检查：

```
数据质量检查清单：
  ✓ 格式一致（字段、角色和 chat template 转换符合当前数据加载器约定）
  ✓ 检查重复（记录精确重复与近重复比例，并抽样判断是否需要去除）
  ✓ 长度分布合理（过长样本截断，过短样本过滤）
  ✓ 无有害内容（色情、暴力、歧视）
  ✓ 标注一致性（同类问题回答风格统一）
  ✓ 覆盖目标场景（不同难度、不同子任务均有样本）
```

### 数据增强

数据量不足时，可考虑以下方式扩充，但合成数据必须经过规则、模型或人工验证，避免把错误和单一表达方式放大：

- **指令改写**：用 LLM 对同一任务生成多种表述方式（"翻译" → "请将…译为英文" / "用英语表达…"）
- **回译**：中文 → 英文 → 中文，产生语义等价但表述不同的样本
- **难度分层**：在已有样本基础上，让 LLM 生成更难/更简单的变体

**混合任务时的数据配比：** 多任务微调时，各任务样本量差异过大会导致模型偏向大任务。常见做法是按任务数量取平方根后归一化（temperature sampling），或直接手动设定各任务上限。

### 数据规模实验

数据规模应通过学习曲线决定，而不是套用固定条数。可以从一组覆盖核心场景的高质量样本开始，按倍数增加数据量，并固定基础模型、训练预算和验证集，观察：

- 目标任务指标是否继续改善；
- 边界场景和安全回归是否恶化；
- 新增样本是否带来新的输入覆盖，而不只是近重复；
- 训练收益是否值得新增标注与计算成本。

---

## 主流微调工具

### PEFT（Hugging Face）

最基础的 LoRA 实现，与 Transformers 生态无缝集成：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                          # LoRA rank
    lora_alpha=16,                # 缩放系数
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 3,407,872 || all params: 7,241,732,096 || trainable%: 0.047
```

### LLaMA-Factory

一站式微调框架，支持 100+ 模型，提供 Web UI：

```bash
pip install llamafactory

# 启动 Web UI
llamafactory-cli webui

# 命令行训练
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset alpaca_zh \
  --finetuning_type lora \
  --lora_rank 8 \
  --output_dir ./output
```

**适合：** 快速上手，不想写训练代码。  
**局限：** 高度封装，复杂实验（自定义 loss、多阶段训练）难以实现。

### Unsloth

下面给出一个面向低比特 LoRA 的实现示例。速度和显存收益依赖软件版本、模型架构、序列长度、batch、算子和硬件，应在目标环境中与基础实现实测：

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,   # QLoRA
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
```

**适合：** 当前版本明确支持的模型，希望快速尝试低比特 LoRA。
**局限：** 模型、平台和自定义训练逻辑支持范围会随版本变化，选型前应核对官方兼容列表并保留可复现环境。

### 工具选型建议

| 场景 | 推荐工具 | 局限 |
|------|---------|------|
| 快速验证，不想写代码 | LLaMA-Factory（Web UI） | 复杂实验难以定制 |
| 显存受限的单卡实验 | 当前环境支持的 QLoRA 实现 | 需要验证算子、模型与平台兼容性 |
| 需要自定义训练逻辑 | PEFT + Transformers | 需要手写训练循环 |
| 大规模分布式训练 | DeepSpeed + PEFT | 配置复杂，调试成本高 |

---

## 训练配置参考

### 关键超参数

```yaml
# LoRA 实验起点示例；不是跨模型推荐值
model: Qwen2.5-7B-Instruct
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules: [q_proj, v_proj, k_proj, o_proj]

learning_rate: 2e-4
lr_scheduler: cosine
warmup_ratio: 0.03
num_epochs: 3
batch_size: 4
gradient_accumulation_steps: 4   # 等效 batch_size = 16
max_seq_length: 2048

# 运行前应分别估算并实测：基础权重、可训练参数、
# 优化器状态、梯度、激活、临时 buffer 和框架开销。
```

### 训练监控

关注以下指标判断训练是否正常：

```
正常训练曲线：
  训练 loss：持续下降，最终趋于平稳
  验证 loss：下降后趋于平稳（若上升则过拟合）
  梯度范数：观察趋势、尖峰和非有限值；没有跨模型通用正常区间

异常信号：
  loss 不下降 → 学习率太小或数据格式错误
  loss 突然变 NaN → 学习率太大或数值溢出
  验证 loss 上升 → 过拟合，减少 epoch 或增加 dropout
```

---

## 评估与合并

### 评估微调效果

**定性评估**（必做）：对比微调前后的输出，覆盖目标场景的典型样本和边界情况。

```python
# 定性评估：对比微调前后的输出
prompts = ["你的测试问题1", "你的测试问题2"]
for prompt in prompts:
    base_output = base_model.generate(prompt)
    ft_output = finetuned_model.generate(prompt)
    print(f"Base: {base_output}")
    print(f"Finetuned: {ft_output}")
```

**定量评估**：在保留的测试集上计算任务指标。

| 任务类型 | 推荐指标 | 注意事项 |
|---------|---------|---------|
| 分类/抽取 | 准确率、F1 | 结合类别不平衡和错误代价选择 |
| 代码生成 | pass@k（执行通过率） | 比 BLEU 更有意义 |
| 摘要/翻译 | ROUGE-L、BLEU | 与人类判断相关性有限，仅供参考 |
| 开放生成 | 任务评分器 + 人工评估 | 自动评分与人工评分都需校准一致性 |

**人工评估：** BLEU/ROUGE 等表面指标对开放生成的覆盖有限。应设计包含事实性、任务完成度、格式和安全性的评分表，并根据目标置信度、差异大小和预算确定抽样规模，同时记录评审者一致性。

### 合并 LoRA 权重

推理时可以将 LoRA 权重合并到基础模型，消除推理开销：

```python
from peft import PeftModel

# 加载基础模型 + LoRA 权重
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
model = PeftModel.from_pretrained(model, "./lora_output")

# 合并并保存
model = model.merge_and_unload()
model.save_pretrained("./merged_model")
```

合并可以减少运行时单独加载适配器的开销，但模型文件大小、数值误差和推理速度仍取决于保存精度、量化顺序与推理后端，应重新做输出和性能回归。

---

## 常见坑与避坑指南

| 坑 | 现象 | 解决方法 |
|----|------|---------|
| 数据模板与模型不匹配 | loss 异常或输出角色混乱 | 检查数据加载器、chat template、特殊 token 和 loss mask |
| `max_seq_length` 过大 | OOM 或吞吐骤降 | 统计长度分布，在截断损失与资源预算之间做实验 |
| `target_modules` 不合适 | 参数量和效果偏离预期 | 打印模块名与可训练参数，并比较多组目标模块消融 |
| 基础模型选择不匹配 | 指令行为或领域能力不足 | 比较 base / instruct 起点，记录 chat template 和初始基线 |
| 学习率或精度不稳定 | loss 尖峰、NaN | 检查数据、loss scale、梯度和精度配置，再缩小搜索范围 |
| 过拟合 | 训练指标改善但验证回退 | 使用独立验证集、早停并检查能力与安全回归 |

---

## 后续步骤

微调完成后的典型流程：

```
1. 合并 LoRA 权重（见上方代码）
   ↓
2. 量化（可选）
   INT4/INT8 量化减少显存，适合部署
   工具：llama.cpp（GGUF）、AutoGPTQ、AWQ
   ↓
3. 部署推理服务
   本地：Ollama（GGUF）、vLLM（HuggingFace 格式）
   云端：HuggingFace Inference Endpoints、自建 API
   ↓
4. 持续迭代
   收集线上失败案例 → 补充到训练集 → 重新微调
   按数据漂移、失败率、业务风险和发布节奏触发评估与再训练
```

**部署详情**见 [推理部署优化](inference_deployment.md)。

---

**返回：** [第6章：LLM应用](../README.md)
