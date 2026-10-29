"""Generate reproducible comparison plots from the structured metrics payload."""

import os
from pathlib import Path


MODEL_COLORS = {
    "yolov8n": "#4f46e5",
    "detr_resnet50": "#9333ea",
}


def write_comparison_plots(result: dict, output_dir: Path) -> dict[str, str]:
    """Write PR and latency plots and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output_dir / ".cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pr_path = output_dir / "precision_recall_curve.png"
    latency_path = output_dir / "latency_comparison.png"

    figure, axis = plt.subplots(figsize=(8, 6))
    for model_name, model_result in result["models"].items():
        points = model_result["pr_curve"]
        recalls = [0.0] + [point["recall"] for point in points]
        precisions = [1.0] + [point["precision"] for point in points]
        axis.step(
            recalls,
            precisions,
            where="post",
            label=f"{model_name} (AP={model_result['metrics']['average_precision']:.3f})",
            color=MODEL_COLORS.get(model_name),
            linewidth=2.2,
        )
        best = model_result["operating_points"]["best_f1"]
        if best:
            axis.scatter(
                best["recall"],
                best["precision"],
                color=MODEL_COLORS.get(model_name),
                edgecolor="white",
                linewidth=1.2,
                s=70,
                zorder=3,
            )
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.02), ylim=(0, 1.02))
    axis.set_title("Person detection precision-recall curve")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(pr_path, dpi=160)
    plt.close(figure)

    names = list(result["models"])
    means = [result["models"][name]["latency"]["mean_ms"] for name in names]
    p95_values = [result["models"][name]["latency"]["p95_ms"] for name in names]
    positions = list(range(len(names)))
    figure, axis = plt.subplots(figsize=(8, 5))
    width = 0.34
    axis.bar([item - width / 2 for item in positions], means, width, label="Mean")
    axis.bar([item + width / 2 for item in positions], p95_values, width, label="P95")
    axis.set_xticks(positions, names)
    axis.set_ylabel("Latency per image (ms)")
    axis.set_title(f"CPU inference latency ({result['config']['images']} images)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(latency_path, dpi=160)
    plt.close(figure)

    return {
        "precision_recall_curve": str(pr_path),
        "latency_comparison": str(latency_path),
    }
