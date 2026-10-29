"""Geometry of resizing, tiling and multi-scale image processing."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

IMAGE_SIZE = 1024
PATCH_SIZE = 16
STANDARD_SIZE = 224
OUTPUT_PATH = Path("assets/ch07_high_resolution_processing.png")


def create_demo_image(image_size=IMAGE_SIZE, seed=42):
    """Create a deterministic synthetic image for geometry visualization."""
    rng = np.random.RandomState(seed)
    image = rng.normal(0, 0.12, (image_size, image_size))
    image[image_size // 5 : image_size // 3, image_size // 6 : image_size // 2] += 1
    image[image_size // 2 : image_size * 4 // 5, image_size * 3 // 5 :] += 0.6
    return image


def patch_division_method(image, patch_size, standard_size):
    """Split an image into non-overlapping patches."""
    del standard_size  # Kept in the public signature for tutorial compatibility.
    h, w = image.shape
    n_patches_h = h // patch_size
    n_patches_w = w // patch_size
    patches = []
    positions = []
    for row in range(n_patches_h):
        for col in range(n_patches_w):
            patches.append(
                image[
                    row * patch_size : (row + 1) * patch_size,
                    col * patch_size : (col + 1) * patch_size,
                ]
            )
            positions.append((row, col))
    return patches, positions, (n_patches_h, n_patches_w)


def dynamic_resolution_method(image, target_tokens=576, patch_size=PATCH_SIZE):
    """Choose a patch grid near a token budget while preserving aspect ratio."""
    h, w = image.shape
    aspect_ratio = w / h
    grid_h = max(1, int(round(np.sqrt(target_tokens / aspect_ratio))))
    grid_w = max(1, int(round(grid_h * aspect_ratio)))
    resolution = (grid_h * patch_size, grid_w * patch_size)

    row_indices = np.linspace(0, h - 1, resolution[0]).astype(int)
    col_indices = np.linspace(0, w - 1, resolution[1]).astype(int)
    resized = image[np.ix_(row_indices, col_indices)]
    return resized, resolution, (grid_h, grid_w)


def progressive_processing_method(image, scales=None, patch_size=PATCH_SIZE):
    """Count patch grids at several image scales."""
    if scales is None:
        scales = [1.0, 0.5, 0.25]
    features = []
    for scale in scales:
        step = max(1, int(round(1 / scale)))
        scaled = image[::step, ::step]
        grid = (
            int(np.ceil(scaled.shape[0] / patch_size)),
            int(np.ceil(scaled.shape[1] / patch_size)),
        )
        features.append({"scale": scale, "size": scaled.shape, "grid": grid})
    return features


def run_experiment(
    image_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    standard_size=STANDARD_SIZE,
    seed=42,
):
    image = create_demo_image(image_size, seed=seed)
    patches, _, patch_grid = patch_division_method(
        image, patch_size, standard_size
    )
    resized, resized_shape, dynamic_grid = dynamic_resolution_method(image)
    scales = progressive_processing_method(image, patch_size=patch_size)

    methods_info = {
        "Full patch grid": {
            "views": 1,
            "tokens": len(patches),
            "resolution": (image_size, image_size),
        },
        "Token-budget resize": {
            "views": 1,
            "tokens": int(np.prod(dynamic_grid)),
            "resolution": resized_shape,
        },
        "Three scales": {
            "views": len(scales),
            "tokens": sum(int(np.prod(item["grid"])) for item in scales),
            "resolution": (image_size, image_size),
        },
    }
    return {
        "image": image,
        "resized": resized,
        "patch_grid": patch_grid,
        "methods_info": methods_info,
        "image_size": image_size,
        "patch_size": patch_size,
    }


def print_summary(result):
    print("High-resolution processing geometry")
    for name, info in result["methods_info"].items():
        print(
            f"{name}: views={info['views']}, tokens={info['tokens']}, "
            f"resolution={info['resolution']}"
        )


def plot_high_resolution_processing(result, output_path=OUTPUT_PATH):
    """Plot image geometry and derived token counts."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "High-resolution processing: detail versus visual-token count",
        fontsize=14,
        fontweight="bold",
    )

    axes[0, 0].imshow(result["image"], cmap="gray")
    axes[0, 0].set_title("Original synthetic image")

    axes[0, 1].imshow(result["image"], cmap="gray", alpha=0.65)
    grid_h, grid_w = result["patch_grid"]
    step = max(1, grid_h // 16)
    for row in range(0, grid_h + 1, step):
        axes[0, 1].axhline(
            row * result["patch_size"], color="#DC2626", linewidth=0.45
        )
    for col in range(0, grid_w + 1, step):
        axes[0, 1].axvline(
            col * result["patch_size"], color="#DC2626", linewidth=0.45
        )
    axes[0, 1].set_title(f"Patch grid: {grid_h} x {grid_w}")

    axes[1, 0].imshow(result["resized"], cmap="gray")
    axes[1, 0].set_title(f"Token-budget resize: {result['resized'].shape}")

    names = list(result["methods_info"])
    tokens = [result["methods_info"][name]["tokens"] for name in names]
    axes[1, 1].bar(names, tokens, color=("#93C5FD", "#C4B5FD", "#86EFAC"))
    axes[1, 1].set_ylabel("Derived visual-token count")
    axes[1, 1].set_title("Geometry cost, not task accuracy")
    axes[1, 1].tick_params(axis="x", rotation=18)
    axes[1, 1].grid(axis="y", alpha=0.25)

    for ax in axes.flat[:3]:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    result = run_experiment()
    print_summary(result)
    output_path = plot_high_resolution_processing(result)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
