# 扩展：Agent 框架深度

**所属章节：** [第6章：LLM应用](../README.md)  
**前置阅读：** [6.4 Agent 框架](../04_agent.md)

---

## 概览

6.4 节介绍了 Agent 的基本概念和 ReAct 模式。本节聚焦**工程可靠性**：如何让工具调用更稳定、多步推理更可控、复杂任务更可靠。

---

## 工具定义规范

### 标准格式

工具定义的质量直接决定 LLM 能否正确调用。关键是 `description` 要精确：

```python
tools = [
    {
        "name": "search_web",
        "description": "搜索互联网获取最新信息。适用于：需要实时数据、近期事件、具体数字。不适用于：通用知识、历史事实（模型已知）。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，建议使用英文以获得更好结果"
                },
                "num_results": {
                    "type": "integer",
                    "description": "返回结果数量，默认 5，最大 10",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
]
```

**工具描述的三要素：**
1. **做什么**：工具的功能
2. **什么时候用**：适用场景
3. **什么时候不用**：避免误调用

### 工具调用可靠性

LLM 可能调用错误的工具或传入无效参数。防御策略：

```
LLM 生成工具调用
    │
    ▼
[参数验证]（JSON Schema 校验）
    │
    ├─ 校验失败 → 返回错误信息给 LLM，要求重试
    │
    ▼
[工具执行]
    │
    ├─ 执行失败 → 返回错误信息给 LLM，最多重试 N 次
    │
    ▼
工具结果 → 返回给 LLM
```

参数类型错误（如 LLM 把数字传成字符串）是最常见的失败原因，可以在校验层自动修正：

```python
def validate_and_fix_params(params: dict, schema: dict) -> dict:
    """尝试自动修正类型错误，无法修正时抛出异常让 LLM 重试。"""
    fixed = {}
    for key, prop in schema.get("properties", {}).items():
        val = params.get(key)
        if val is None:
            if key in schema.get("required", []):
                raise ValueError(f"Missing required parameter: {key}")
            continue
        expected_type = prop.get("type")
        if expected_type == "integer" and isinstance(val, str):
            fixed[key] = int(val)   # "5" → 5
        elif expected_type == "number" and isinstance(val, str):
            fixed[key] = float(val)
        else:
            fixed[key] = val
    return fixed
```

工具调用重试建议使用指数退避（exponential backoff），避免在工具服务故障时频繁重试：

```python
import time

def call_tool_with_retry(tool_fn, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            return tool_fn(**params)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)   # 1s, 2s, 4s
```

---

## ReAct 模式详解

ReAct（Reasoning + Acting）是一种常见的 Agent 循环：

```
用户：2024年诺贝尔物理学奖得主是谁？他们的主要贡献是什么？

Thought: 我需要查找2024年诺贝尔物理学奖的信息
Action: search_web(query="2024 Nobel Prize Physics winner")
Observation: 搜索结果显示获奖者是 John Hopfield 和 Geoffrey Hinton...

Thought: 我已经知道获奖者，现在需要了解他们的具体贡献
Action: search_web(query="Hopfield Hinton Nobel Prize contributions neural networks")
Observation: 他们因在人工神经网络领域的基础性发现而获奖...

Thought: 我现在有足够的信息来回答问题了
Final Answer: 2024年诺贝尔物理学奖授予 John Hopfield 和 Geoffrey Hinton，
表彰他们在人工神经网络领域的基础性发现...
```

**关键设计：** 工程日志应记录简短、可审计的动作理由，而不是依赖或暴露模型的完整内部思维链。动作理由本身不能保证决策正确，仍需校验参数、工具返回值和停止条件。

**错误恢复案例：** 当 Thought 推理出错时，将错误反馈给 LLM 让其重新规划：

```
Thought: 我需要查询用户 ID 为 abc123 的订单
Action: query_orders(user_id="abc123")
Observation: Error: user_id must be integer, got string "abc123"

Thought: 参数类型错误，user_id 应该是整数。但我没有整数 ID，
         需要先通过用户名查询 ID
Action: get_user_id(username="abc123")
Observation: {"user_id": 42}

Thought: 获取到整数 ID，现在可以查询订单了
Action: query_orders(user_id=42)
```

---

## 多步推理控制

### 循环终止条件

Agent 循环必须有明确的终止条件，否则可能无限循环：

```python
MAX_STEPS = 10   # 最大步数限制

for step in range(MAX_STEPS):
    response = llm.call(messages)
    
    if response.is_final_answer:
        return response.content
    
    tool_result = execute_tool(response.tool_call)
    messages.append(tool_result)

# 超过最大步数，强制返回
return "任务超过最大步数限制，当前进度：..."
```

### 错误恢复

工具调用失败时，让 LLM 自主决策如何恢复：

```
工具调用失败
    │
    ▼
将错误信息返回给 LLM：
"工具 search_web 调用失败：网络超时。
请选择：1) 重试 2) 使用其他工具 3) 基于已有信息回答"
    │
    ▼
LLM 决策下一步
```

### 并行工具调用

当多个工具调用相互独立且服务端允许并发时，可以并行执行以降低串行等待：

```python
# 串行：总时间 = t1 + t2 + t3
result1 = tool_a(...)
result2 = tool_b(...)
result3 = tool_c(...)

# 理想化并行下界接近 max(t1, t2, t3)，实际还有调度与网络开销
import asyncio
results = await asyncio.gather(
    tool_a(...),
    tool_b(...),
    tool_c(...)
)
```

部分模型接口允许在一次响应中返回多个工具调用。是否可以安全并行还要检查调用之间的数据依赖、限流和副作用。

---

## 记忆系统

Agent 的记忆分为四类：

| 类型 | 存储位置 | 生命周期 | 用途 |
|------|---------|---------|------|
| 工作记忆 | 上下文窗口 | 单次对话 | 当前任务的中间结果 |
| 对话历史 | 数据库 | 跨对话 | 记住用户偏好和历史 |
| 知识库 | 向量数据库 | 持久 | 领域知识（RAG） |
| 程序记忆 | 代码/工具 | 持久 | 固定的操作流程 |

**上下文窗口管理：** 长对话需要压缩历史，常见策略：
- 保留最近 N 轮
- 用 LLM 总结早期对话
- 只保留关键事实（实体、决策）

**对话历史持久化示例：**

```python
import sqlite3, json

def save_turn(session_id: str, role: str, content: str):
    conn = sqlite3.connect("agent_history.db")
    conn.execute(
        "INSERT INTO turns (session_id, role, content, ts) VALUES (?,?,?,datetime('now'))",
        (session_id, role, content)
    )
    conn.commit()

def load_recent_turns(session_id: str, n: int = 20) -> list[dict]:
    conn = sqlite3.connect("agent_history.db")
    rows = conn.execute(
        "SELECT role, content FROM turns WHERE session_id=? ORDER BY ts DESC LIMIT ?",
        (session_id, n)
    ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]
```

**RAG 与 Agent 集成：** 将向量检索封装为一个工具，Agent 按需调用：

```python
tools = [
    {
        "name": "search_knowledge_base",
        "description": "在内部知识库中检索相关文档。适用于：产品文档、政策、FAQ。",
        "parameters": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 3}}
    }
]
# Agent 调用时触发向量检索，结果作为 Observation 返回
```

---

## Agent 框架选型

下表用于识别抽象方式，不代表长期有效的功能排名。框架接口和支持范围变化较快，选型时应核对目标版本、维护状态和迁移成本。

| 框架 | 定位 | 特点 | 局限 |
|------|------|------|------|
| LangChain Agents | 通用 | 工具生态丰富 | 抽象层多，学习曲线陡峭，调试困难 |
| LlamaIndex Agents | RAG + Agent | 与知识库集成好 | 非 RAG 场景功能较弱 |
| AutoGen | 多 Agent | 多个 Agent 协作对话 | 对话可能无限循环，必须设计终止条件 |
| CrewAI | 多 Agent | 角色分工，适合复杂任务 | 配置繁琐，调试多 Agent 交互困难 |
| Semantic Kernel | 企业级 | 微软出品，.NET/Python | 国内用户较少，英文资料为主 |

**选择建议：**

| 场景 | 推荐 | 不推荐 | 理由 |
|------|------|--------|------|
| 快速原型验证 | 原生 function calling | LangChain | 减少抽象层，调试简单 |
| 复杂工具链 | LangChain | 原生 API | 内置工具、解析器丰富 |
| 多 Agent 协作 | AutoGen | LangChain | AutoGen 原生支持对话管理 |
| 受限网络环境 | 原生 API + 自建 | LangChain | 避免外网依赖 |

---

## 成本与延迟控制

Agent 的成本来自多轮模型调用、工具服务、重试、日志和人工接管：

```
单次任务成本
≈ Σ(每步输入 token × 输入单价
   + 每步输出 token × 输出单价
   + 工具调用成本)
  + 重试、存储与人工处理成本
```

**成本优化策略：**
- 用小模型处理简单步骤（工具选择），大模型处理复杂推理
- 缓存工具调用结果（相同参数的重复调用）
- 设置合理的最大步数限制

---

## 推荐论文

- **Yao et al. (2022)** — "ReAct: Synergizing Reasoning and Acting in Language Models"（ReAct 框架）
- **Schick et al. (2023)** — "Toolformer: Language Models Can Teach Themselves to Use Tools"
- **Wang et al. (2023)** — "Voyager: An Open-Ended Embodied Agent with Large Language Models"
- **Wu et al. (2023)** — "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation"

---

## 常见坑与解决方案

| 常见问题 | 解决方案 |
|---------|---------|
| LLM 无限循环调用同一个工具 | 在 Prompt 中限制："同一工具连续调用不超过 3 次"；代码层面记录调用历史并强制终止 |
| 工具调用参数总是不对 | 优化 `description`，加上正例和反例；对高频错误参数做自动类型修正（见上方代码） |
| 多步推理中"遗忘"早期信息 | 每步将关键信息提取到"工作记忆"摘要，附加到下一步的 Prompt 中 |
| 工具返回结果太长撑爆上下文 | 对工具结果做截断（保留前 N 字符）或摘要（用小模型压缩） |
| AutoGen 多 Agent 对话无限循环 | 设计明确的终止条件（如 `TERMINATE` 关键词）；设置最大对话轮数 |
| Agent 成本失控 | 监控每次任务的步数和 token 消耗；设置单任务 token 上限并告警 |

---

## Agent 质量评估

如何衡量 Agent 做得好不好：

| 指标 | 说明 | 测量方法 |
|------|------|---------|
| 任务成功率 | 端到端完成任务的比例 | 人工标注测试集，判断最终答案是否正确 |
| 平均步数 | 完成任务所需的工具调用次数 | 自动统计，步数越少效率越高 |
| 工具调用准确率 | 工具和参数是否满足当前状态下的约束 | 对比允许的调用集合与参数规则，而非只锁定唯一序列 |
| 错误恢复率 | 遇到工具失败后成功恢复的比例 | 注入故障测试 |
| 人工评估 | 任务质量、动作理由和风险处理 | 使用评分量表并记录评审一致性 |

**最小可行评估方案：** 先构建覆盖正常、工具失败、权限拒绝、重复调用和不可回答场景的小型回归集；样本量按风险和目标置信度扩展。每次迭代固定配置运行，监控任务成功率、成本和副作用。

---

**返回：** [第6章：LLM应用](../README.md)
