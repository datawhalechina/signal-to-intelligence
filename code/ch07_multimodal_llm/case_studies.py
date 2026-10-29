"""Evaluation workflows for three multimodal case studies."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUTPUT_PATH = Path("assets/ch07_case_studies.png")


def run_experiment():
    """Return inputs, processing steps and evidence for three cases."""
    cases = [
        {
            "title": "Document Understanding",
            "input": "Page image",
            "pipeline": "OCR / VLM  ->  JSON  ->  validation",
            "metrics": "Field F1 | schema pass | review rate",
        },
        {
            "title": "Chart Analysis",
            "input": "Chart image",
            "pipeline": "axes / legend  ->  data  ->  calculation",
            "metrics": "Value error | consistency | robustness",
        },
        {
            "title": "Multilingual Vision",
            "input": "Image + language",
            "pipeline": "same task  ->  grouped by language",
            "metrics": "OCR CER | task score | language follow",
        },
    ]
    return {"cases": cases, "output_path": OUTPUT_PATH}


def _save_evaluation_map(cases=None, output_path=OUTPUT_PATH):
    """Save the input-to-evidence map without invented benchmark scores."""
    if cases is None:
        cases = run_experiment()["cases"]

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.suptitle(
        "Multimodal case studies: define evidence before comparing models",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")

    colors = ("#DBEAFE", "#EDE9FE", "#DCFCE7")
    for row, (case, color) in enumerate(zip(cases, colors)):
        y = 4.55 - row * 1.55
        ax.text(
            1.55,
            y + 0.42,
            case["title"],
            ha="right",
            fontsize=10,
            fontweight="bold",
        )
        boxes = (
            (2.9, case["input"], "Input"),
            (7.4, case["pipeline"], "Process"),
            (12.25, case["metrics"], "Evidence"),
        )
        widths = (2.5, 4.0, 4.5)
        for (x, value, label), width in zip(boxes, widths):
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
            ax.text(x, y + 0.30, value, ha="center", fontsize=9)
        for x1, x2 in ((4.2, 5.35), (9.45, 9.95)):
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
        "Fix dataset, preprocessing, prompt, model version and hardware; "
        "save raw predictions.",
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
    output_path = _save_evaluation_map()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
