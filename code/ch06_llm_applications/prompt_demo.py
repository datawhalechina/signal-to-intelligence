"""Illustrative Prompt strategy visualization, not benchmark results."""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_PATH = Path("assets/ch06_prompt_techniques.png")


def run_experiment():
    # Hand-authored scenario scores used only to explain that strategy ranking
    # can vary by task. They are not measurements from any model or dataset.
    tasks = ["Classification", "Math\nReasoning", "Code\nGeneration", "Summarization"]
    zero_shot = np.array([72, 45, 58, 78], dtype=float)
    few_shot = np.array([85, 62, 74, 83], dtype=float)
    cot = np.array([84, 88, 76, 81], dtype=float)
    methods = ["Zero-shot", "Few-shot", "Chain-of-Thought", "Fine-tuning"]
    effort = np.array([1, 2, 3, 9], dtype=float)
    perf = np.array([65, 78, 85, 92], dtype=float)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    steps = [
        (1.0, "Write\nInitial\nPrompt", "#4C72B0"),
        (3.2, "Run Fixed\nEvaluation\nSet", "#55A868"),
        (5.4, "Analyze\nFailure\nCases", "#C44E52"),
        (7.6, "Refine\nPrompt", "#8172B2"),
    ]

    return {
        "data_kind": "illustrative_scenario",
        "tasks": tasks,
        "zero_shot": zero_shot,
        "few_shot": few_shot,
        "cot": cot,
        "methods": methods,
        "effort": effort,
        "perf": perf,
        "colors": colors,
        "steps": steps,
    }


def plot_technique_comparison(ax, result):
    x = np.arange(len(result["tasks"]))
    width = 0.25
    ax.bar(x - width, result["zero_shot"], width, label="Zero-shot", color="#4C72B0", alpha=0.85)
    ax.bar(x, result["few_shot"], width, label="Few-shot", color="#55A868", alpha=0.85)
    ax.bar(
        x + width,
        result["cot"],
        width,
        label="Chain-of-Thought",
        color="#C44E52",
        alpha=0.85,
    )
    ax.set_xlabel("Task Type", fontsize=11)
    ax.set_ylabel("Illustrative task score", fontsize=11)
    ax.set_title("Illustrative Strategy Differences", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(result["tasks"], fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def plot_cost_vs_effort(ax, result):
    for i, (method, effort, perf, color) in enumerate(
        zip(result["methods"], result["effort"], result["perf"], result["colors"])
    ):
        ax.scatter(effort, perf, s=120, color=color, zorder=3, edgecolors="white", linewidths=1.5)
        offset_x = 0.3 if i < 3 else -1.8
        offset_y = 1.5 if i != 1 else -3.5
        ax.annotate(method, (effort, perf), xytext=(effort + offset_x, perf + offset_y),
                    fontsize=10, color=color)

    rect = mpatches.FancyBboxPatch(
        (1.5, 74),
        2.5,
        14,
        boxstyle="round,pad=0.3",
        linewidth=1.5,
        edgecolor="#55A868",
        facecolor="#55A868",
        alpha=0.08,
    )
    ax.add_patch(rect)
    ax.text(2.75, 89.5, "Example region", ha="center", fontsize=9, color="#55A868", style="italic")
    ax.set_xlabel("Engineering Effort (relative)", fontsize=11)
    ax.set_ylabel("Illustrative quality score", fontsize=11)
    ax.set_title("Illustrative Effort vs Quality", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 11)
    ax.set_ylim(55, 98)
    ax.grid(alpha=0.3)


def plot_iteration_workflow(ax, result):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Iterative Prompt Optimization Workflow", fontsize=13, fontweight="bold")

    for x, label, color in result["steps"]:
        circle = plt.Circle((x, 2), 0.75, color=color, alpha=0.85, zorder=3)
        ax.add_patch(circle)
        ax.text(
            x,
            2,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color="white",
            fontweight="bold",
            zorder=4,
        )

    for i in range(len(result["steps"]) - 1):
        x1 = result["steps"][i][0] + 0.75
        x2 = result["steps"][i + 1][0] - 0.75
        ax.annotate(
            "",
            xy=(x2, 2),
            xytext=(x1, 2),
            arrowprops=dict(arrowstyle="->", lw=2, color="#555555"),
        )

    ax.annotate(
        "",
        xy=(1.0, 1.1),
        xytext=(7.6, 1.1),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#888888", connectionstyle="arc3,rad=0"),
    )
    ax.text(4.3, 0.55, "Iterate until quality target met", ha="center",
            fontsize=9, color="#888888", style="italic")
    ax.annotate(
        "",
        xy=(9.3, 2),
        xytext=(8.35, 2),
        arrowprops=dict(arrowstyle="->", lw=2, color="#555555"),
    )
    ax.text(
        9.6,
        2,
        "Done",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
        fontweight="bold",
    )


def plot_results(result, output_path=OUTPUT_PATH):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Prompt Engineering: Techniques, Trade-offs & Workflow",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plot_technique_comparison(axes[0], result)
    plot_cost_vs_effort(axes[1], result)
    plot_iteration_workflow(axes[2], result)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    result = run_experiment()
    output_path = plot_results(result)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
