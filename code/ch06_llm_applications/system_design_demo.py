"""Illustrative decision framework; scores are not empirical benchmarks."""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_PATH = Path("assets/ch06_system_design.png")


def run_experiment():
    # These profiles make trade-offs visible in one teaching figure. They do
    # not rank technologies universally and must not be used for procurement.
    categories = [
        "Performance",
        "Flexibility",
        "Cost\nEfficiency",
        "Latency\nEfficiency",
        "Maintainability",
        "Data\nEfficiency",
    ]
    approaches = {
        "Prompt": [3, 5, 5, 4, 5, 5],
        "Fine-tuning": [5, 2, 2, 5, 3, 1],
        "RAG": [4, 4, 3, 3, 4, 4],
        "Agent": [5, 5, 1, 1, 2, 3],
    }
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    approaches_order = list(approaches.keys())
    setup_cost = np.array([1, 3, 8, 5], dtype=float)
    run_cost = np.array([3, 4, 2, 7], dtype=float)
    maintain = np.array([1, 3, 4, 6], dtype=float)
    totals = setup_cost + run_cost + maintain

    return {
        "data_kind": "illustrative_profiles",
        "categories": categories,
        "approaches": approaches,
        "approaches_order": approaches_order,
        "colors": colors,
        "setup_cost": setup_cost,
        "run_cost": run_cost,
        "maintain": maintain,
        "totals": totals,
    }


def plot_radar(ax, result):
    categories = result["categories"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=7, color="grey")
    ax.grid(color="grey", alpha=0.3)

    for (name, vals), color in zip(result["approaches"].items(), result["colors"]):
        vals_plot = vals + vals[:1]
        ax.plot(angles, vals_plot, "o-", linewidth=2, color=color, label=name)
        ax.fill(angles, vals_plot, alpha=0.08, color=color)

    ax.set_title("Illustrative Profiles\n(scenario assumptions)", fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)


def plot_decision_tree(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Technology Selection Decision Tree", fontsize=12, fontweight="bold")

    def box(x, y, text, color, width=2.2, height=0.7, fontsize=9):
        rect = mpatches.FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.15",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
            alpha=0.9,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                color="white", fontweight="bold", zorder=4)

    def diamond(x, y, text, color="#E8A838", fontsize=7.5):
        d = 0.55
        xs = [x, x + d * 2.5, x, x - d * 2.5, x]
        ys = [y + d, y, y - d, y, y + d]
        ax.fill(xs, ys, color=color, alpha=0.9, zorder=3)
        ax.plot(xs, ys, color="white", linewidth=1.2, zorder=4)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                color="white", fontweight="bold", zorder=5)

    def arrow(x1, y1, x2, y2, label="", label_side="right"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555"), zorder=2)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            offset = 0.25 if label_side == "right" else -0.25
            ax.text(mx + offset, my, label, fontsize=8, color="#555555",
                    ha="center", va="center")

    box(5, 9.2, "New LLM Task", "#555555", width=2.5)
    diamond(5, 7.8, "Works with\nPrompt?")
    box(8.2, 7.8, "Prompt\nEngineering", "#4C72B0", width=2.0)
    diamond(5, 6.2, "Needs fresh\nknowledge?")
    box(8.2, 6.2, "RAG", "#C44E52", width=2.0)
    diamond(5, 4.6, "Strict format\nor style?")
    box(8.2, 4.6, "Fine-tuning\n(LoRA)", "#55A868", width=2.0)
    diamond(5, 3.0, "Dynamic next\naction needed?")
    box(8.2, 3.0, "Agent", "#8172B2", width=2.0)
    box(5, 1.5, "Combine approaches\nas needed", "#888888", width=3.2)

    arrow(5, 8.85, 5, 8.35)
    arrow(5, 7.25, 5, 6.75)
    arrow(5, 5.65, 5, 5.15)
    arrow(5, 4.05, 5, 3.55)
    arrow(5, 2.45, 5, 1.85)
    arrow(6.12, 7.8, 7.2, 7.8, "Yes", "right")
    arrow(6.12, 6.2, 7.2, 6.2, "Yes", "right")
    arrow(6.12, 4.6, 7.2, 4.6, "Yes", "right")
    arrow(6.12, 3.0, 7.2, 3.0, "Yes", "right")
    for y in [7.8, 6.2, 4.6, 3.0]:
        ax.text(4.55, y - 0.55, "No", fontsize=8, color="#555555", ha="center")


def plot_cost_timeline(ax, result):
    approaches = ["Prompt", "RAG", "Fine-tuning", "Agent"]
    x = np.arange(len(approaches))
    w = 0.5
    ax.bar(x, result["setup_cost"], w, label="Setup Cost", color="#4C72B0", alpha=0.85)
    ax.bar(
        x,
        result["run_cost"],
        w,
        label="Runtime Cost",
        color="#C44E52",
        alpha=0.85,
        bottom=result["setup_cost"],
    )
    ax.bar(
        x,
        result["maintain"],
        w,
        label="Maintenance Cost",
        color="#8172B2",
        alpha=0.85,
        bottom=result["setup_cost"] + result["run_cost"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, fontsize=11)
    ax.set_ylabel("Illustrative effort units", fontsize=10)
    ax.set_title("Scenario Cost Assumptions", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    for i, total in enumerate(result["totals"]):
        ax.text(i, total + 0.3, str(int(total)), ha="center", fontsize=10, fontweight="bold")


def plot_results(result, output_path=OUTPUT_PATH):
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, polar=True)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    fig.suptitle(
        "LLM System Design: Technology Selection Framework",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plot_radar(ax1, result)
    plot_decision_tree(ax2)
    plot_cost_timeline(ax3, result)

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
