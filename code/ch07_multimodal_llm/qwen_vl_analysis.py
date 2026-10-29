"""Visual-token budget for a dynamic-resolution multimodal model.

The script uses a generic patch size and spatial merge factor to show how image
dimensions affect the number of visual tokens. It does not report model quality.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = Path("assets/ch07_qwen_vl_analysis.png")


def run_experiment(patch_size=14, merge_size=2):
    """Calculate visual-token counts for representative image shapes."""
    labels = ["224x224", "448x448", "896x896", "1024x1536"]
    image_shapes = np.array(
        [[224, 224], [448, 448], [896, 896], [1024, 1536]], dtype=int
    )
    patch_grids = np.ceil(image_shapes / patch_size).astype(int)
    patch_tokens = np.prod(patch_grids, axis=1)
    merged_grids = np.ceil(patch_grids / merge_size).astype(int)
    visual_tokens = np.prod(merged_grids, axis=1)

    return {
        "labels": labels,
        "image_shapes": image_shapes,
        "patch_size": patch_size,
        "merge_size": merge_size,
        "patch_grids": patch_grids,
        "patch_tokens": patch_tokens,
        "visual_tokens": visual_tokens,
    }


def print_summary(result):
    print("Dynamic-resolution visual-token budget")
    print(
        f"patch_size={result['patch_size']}, "
        f"spatial_merge={result['merge_size']}x{result['merge_size']}"
    )
    for label, grid, raw, merged in zip(
        result["labels"],
        result["patch_grids"],
        result["patch_tokens"],
        result["visual_tokens"],
    ):
        print(f"{label}: grid={tuple(grid)}, patches={raw}, visual_tokens={merged}")


def plot_results(result, output_path=OUTPUT_PATH):
    """Plot pixel count and visual-token count without quality claims."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Dynamic resolution: image detail increases the visual-token budget",
        fontsize=13,
        fontweight="bold",
    )

    pixel_counts = np.prod(result["image_shapes"], axis=1)
    x = np.arange(len(result["labels"]))

    axes[0].bar(x, pixel_counts, color="#93C5FD", edgecolor="#64748B")
    axes[0].set_xticks(x, result["labels"], rotation=20)
    axes[0].set_ylabel("Pixels")
    axes[0].set_title("Input pixel count")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        x,
        result["patch_tokens"],
        label="Patch tokens before merge",
        color="#C4B5FD",
        edgecolor="#64748B",
    )
    axes[1].bar(
        x,
        result["visual_tokens"],
        label="Visual tokens after merge",
        color="#86EFAC",
        edgecolor="#64748B",
    )
    axes[1].set_xticks(x, result["labels"], rotation=20)
    axes[1].set_ylabel("Token count")
    axes[1].set_title("Context and compute cost")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    fig.text(
        0.5,
        0.01,
        "Counts are derived from image shape, patch size and merge factor; "
        "they are not model-quality scores.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    plt.tight_layout(rect=(0, 0.05, 1, 0.93))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    result = run_experiment()
    print_summary(result)
    output_path = plot_results(result)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
