"""
第6章代码实验：Agent ReAct 循环演示

演示 ReAct（Reason + Act + Observe）模式：
- 用可审计的简短理由表示下一步工具选择
- 执行工具（Act）获取结果
- 观察结果（Observe）决定是否继续
- 重复直到得出答案或达到最大步数

使用模拟数据，无需 API Key 或外部依赖。
"""

import ast
import operator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_PATH = Path("assets/ch06_agent_error_accumulation.png")

KNOWLEDGE_BASE = {
    "transformer": (
        "Transformer 是基于自注意力机制的神经网络架构，"
        "由 Vaswani 等人于 2017 年提出。"
    ),
    "rag": (
        "RAG（检索增强生成）先从知识库检索相关文档，"
        "再用文档增强 LLM 的生成过程。"
    ),
    "lora": (
        "LoRA 的可训练参数量由 rank、矩阵输入输出维度、层数和目标模块共同决定，"
        "不能用固定百分比代替计算。"
    ),
    "scaling": (
        "Chinchilla 缩放律表明：模型参数量和训练数据量应同步增长，"
        "最优比例约为 20 tokens/参数。"
    ),
}

TOOLS = {}


def search_knowledge(query: str) -> str:
    """在知识库中搜索相关信息（模拟语义检索）"""
    query_lower = query.lower()
    for key, value in KNOWLEDGE_BASE.items():
        if key in query_lower:
            return value
    return "未找到相关信息。"


def calculate(expression: str) -> str:
    """Safely evaluate an arithmetic-only expression."""
    binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
    }
    unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in binary_ops:
            return binary_ops[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_ops:
            return unary_ops[type(node.op)](evaluate(node.operand))
        raise ValueError("只允许数字和基本算术运算")

    try:
        result = evaluate(ast.parse(expression, mode="eval"))
        return str(result)
    except Exception as exc:
        return f"计算错误：{exc}"


TOOLS.update(
    {
        "search": {
            "description": (
                "在知识库中搜索信息。适用于："
                "需要查找概念定义、技术原理。"
            ),
            "func": search_knowledge,
            "param": "query（搜索关键词）",
        },
        "calculate": {
            "description": "计算数学表达式。适用于：数值计算、参数量估算。",
            "func": calculate,
            "param": "expression（Python 数学表达式）",
        },
    }
)


def simulate_llm_reason(question: str, history: list[dict]) -> dict:
    """Simulate auditable action selection, not hidden chain-of-thought."""
    step = len(history)

    if "什么是" in question or "介绍" in question:
        if step == 0:
            keyword = question.replace("什么是", "").replace("？", "").replace("?", "").strip()
            return {
                "rationale": f"定义问题需要先查询本地知识库：{keyword}",
                "action": "search",
                "input": keyword,
            }
        obs = history[-1]["observation"]
        return {
            "rationale": "知识库已返回结果，可以结束工具循环。",
            "action": "answer",
            "input": obs,
        }

    if "参数量" in question or "计算" in question:
        if step == 0:
            return {
                "rationale": "参数量取决于结构，先查询 LoRA 计算条件。",
                "action": "search",
                "input": "lora",
            }
        if step == 1:
            return {
                "rationale": "按 32 层、q/v 两个 4096×4096 矩阵、rank=8 计算。",
                "action": "calculate",
                "input": "32 * 2 * 2 * 4096 * 8",
            }
        calc_result = history[-1]["observation"]
        return {
            "rationale": "结构化参数已计算，可以返回带假设的结果。",
            "action": "answer",
            "input": (
                "在 32 层、隐藏维度 4096、仅对 q/v 投影使用 rank=8 LoRA 的假设下，"
                f"可训练矩阵参数为 {int(float(calc_result)):,}。"
                "更换目标模块或模型结构后必须重新计算。"
            ),
        }

    return {
        "rationale": "当前工具无法提供可靠证据。",
        "action": "answer",
        "input": "这个问题超出了当前知识库范围。",
    }


def solve_question(question: str, max_steps: int = 5) -> dict:
    history = []
    transcript = []

    for step in range(1, max_steps + 1):
        decision = simulate_llm_reason(question, history)
        transcript.append({"step": step, **decision})
        if decision["action"] == "answer":
            return {
                "question": question,
                "answer": decision["input"],
                "history": history,
                "transcript": transcript,
                "finished": True,
            }

        tool_func = TOOLS[decision["action"]]["func"]
        observation = tool_func(decision["input"])
        history.append(
            {
                "step": step,
                "action": decision["action"],
                "input": decision["input"],
                "observation": observation,
            }
        )

    return {
        "question": question,
        "answer": "达到最大步数，未能得出答案。",
        "history": history,
        "transcript": transcript,
        "finished": False,
    }


def react_agent(question: str, max_steps: int = 5) -> str:
    """带打印的 ReAct 循环演示。"""
    result = solve_question(question, max_steps=max_steps)

    print(f"\n{'=' * 55}")
    print(f"问题：{question}")
    print(f"{'=' * 55}")
    for item in result["transcript"]:
        print(f"\n[Step {item['step']}]")
        print(f"  Rationale: {item['rationale']}")
        print(f"  Action  : {item['action']}")
        if item["action"] == "answer":
            print(f"  Answer  : {item['input']}")
            break
        last_obs = next(h["observation"] for h in result["history"] if h["step"] == item["step"])
        print(f"  Input   : {item['input']}")
        print(f"  Observe : {last_obs}")

    return result["answer"]


def demo_error_accumulation():
    """演示多步推理中错误累积的风险"""
    rates = [0.80, 0.90, 0.95]
    steps = list(range(1, 11))
    curves = {rate: [rate**n for n in steps] for rate in rates}
    return {"rates": rates, "steps": steps, "curves": curves}


def plot_results(result, output_path=OUTPUT_PATH):
    chart = demo_error_accumulation()
    print(f"\n{'=' * 55}")
    print("错误累积演示：每步成功率不同时，多步任务的整体成功率")
    print(f"{'=' * 55}")
    for r in chart["rates"]:
        print(f"\n  单步成功率 {r:.0%}：")
        for n in [1, 3, 5, 10]:
            print(f"    {n:2d} 步任务：{r**n:.1%} 成功率")
    print("\n说明：曲线假设各步骤独立、成功率相同且全部必须成功。")
    print("工程实践：固定流程优先用确定性编排；动态步骤需要校验、回退和终止条件。")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#f39c12", "#27ae60"]
    labels = ["80% per step", "90% per step", "95% per step"]

    for rate, color, label in zip(chart["rates"], colors, labels):
        ax.plot(
            chart["steps"],
            [v * 100 for v in chart["curves"][rate]],
            marker="o",
            markersize=5,
            color=color,
            label=label,
            linewidth=2,
        )

    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(10.1, 50, "50%", va="center", fontsize=9, color="gray")
    ax.set_xlabel("Number of Steps", fontsize=12)
    ax.set_ylabel("Overall Success Rate (%)", fontsize=12)
    ax.set_title("Independent-Step Reliability Illustration", fontsize=13, fontweight="bold")
    ax.set_xticks(chart["steps"])
    ax.set_ylim(0, 105)
    ax.legend(title="Per-step success rate", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n图表已保存：{output_path}")
    return output_path


def run_experiment():
    question1 = "什么是 RAG？"
    question2 = "给定结构时，LoRA 可训练参数量如何计算？"
    answer1 = solve_question(question1)
    answer2 = solve_question(question2)
    chart = demo_error_accumulation()
    return {
        "question1": question1,
        "question2": question2,
        "answer1": answer1["answer"],
        "answer2": answer2["answer"],
        "transcript1": answer1["transcript"],
        "transcript2": answer2["transcript"],
        "history1": answer1["history"],
        "history2": answer2["history"],
        "chart": chart,
    }


def main():
    print("第6章：Agent ReAct 循环演示")
    print("（使用模拟数据，无需 API Key）\n")

    run_experiment()
    react_agent("什么是 RAG？")
    react_agent("给定结构时，LoRA 可训练参数量如何计算？")
    plot_results({})

    print(f"\n{'=' * 55}")
    print("演示完成！")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
