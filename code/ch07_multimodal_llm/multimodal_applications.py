"""Evaluation map for captioning, visual QA and image retrieval."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUTPUT_PATH = Path("assets/ch07_multimodal_applications.png")


def run_experiment():
    """Return task inputs, outputs, metrics and common failure modes."""
    tasks = [
        {
            "title": "Image Captioning",
            "input": "image",
            "output": "caption",
            "metrics": "CIDEr / factual checks",
            "failure": "omission or hallucinated detail",
        },
        {
            "title": "Visual Question Answering",
            "input": "image + question",
            "output": "answer",
            "metrics": "task accuracy / error groups",
            "failure": "wrong perception or reasoning",
        },
        {
            "title": "Image Retrieval",
            "input": "text or image query",
            "output": "ranked items",
            "metrics": "Recall@k / latency",
            "failure": "relevant item not retrieved",
        },
    ]
    return {"tasks": tasks, "output_path": OUTPUT_PATH}


def _save_task_map(tasks=None, output_path=OUTPUT_PATH):
    """Draw a task-to-evidence map without built-in benchmark scores."""
    if tasks is None:
        tasks = run_experiment()["tasks"]

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.suptitle(
        "Multimodal applications: match each task with evidence and failures",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")
    colors = ("#DBEAFE", "#EDE9FE", "#DCFCE7")

    for row, (task, color) in enumerate(zip(tasks, colors)):
        y = 4.55 - row * 1.55
        ax.text(
            1.65,
            y + 0.42,
            task["title"],
            ha="right",
            fontsize=10,
            fontweight="bold",
        )
        cells = (
            (3.0, 2.3, "Input -> output", f"{task['input']} -> {task['output']}"),
            (7.2, 3.4, "Evidence", task["metrics"]),
            (12.1, 4.3, "Inspect failures", task["failure"]),
        )
        for x, width, label, value in cells:
            box = FancyBboxPatch(
                (x - width / 2, y),
                width,
                0.88,
                boxstyle="round,pad=0.03,rounding_size=0.12",
                facecolor=color,
                edgecolor="#94A3B8",
                linewidth=1.2,
            )
            ax.add_patch(box)
            ax.text(x, y + 0.60, label, ha="center", fontsize=8, color="#526076")
            ax.text(x, y + 0.29, value, ha="center", fontsize=9)
        for x1, x2 in ((4.2, 5.45), (8.95, 9.9)):
            ax.add_patch(
                FancyArrowPatch(
                    (x1, y + 0.44),
                    (x2, y + 0.44),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.4,
                    color="#64748B",
                )
            )

    ax.text(
        7.5,
        0.12,
        "Use the same dataset, preprocessing and runtime settings when comparing models.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.94))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    output_path = _save_task_map()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
