#!/usr/bin/env python3
"""
Chapter 8.1: model compression trade-offs.

This script uses small NumPy arrays to explain quantization, pruning, and
distillation without requiring GPU, model weights, or external APIs.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class QuantizationResult:
    bits: int
    scale: float
    memory_kb: float
    mean_abs_error: float
    max_abs_error: float


@dataclass
class PruningResult:
    sparsity: float
    remaining_ratio: float
    nonzero_value_storage_kb: float
    illustrative_speedup_upper_bound: float
    needs_sparse_kernel: bool


def symmetric_quantize(weights: np.ndarray, bits: int) -> QuantizationResult:
    """Quantize weights with a simple symmetric integer scheme."""
    if bits < 2:
        raise ValueError("bits must be >= 2")

    max_abs = float(np.max(np.abs(weights)))
    qmax = (2 ** (bits - 1)) - 1
    scale = max_abs / qmax if max_abs > 0 else 1.0
    q = np.clip(np.round(weights / scale), -qmax, qmax)
    dequantized = q * scale
    error = np.abs(weights - dequantized)

    return QuantizationResult(
        bits=bits,
        scale=scale,
        memory_kb=weights.size * bits / 8 / 1024,
        mean_abs_error=float(np.mean(error)),
        max_abs_error=float(np.max(error)),
    )


def magnitude_prune(
    weights: np.ndarray,
    sparsity: float,
    sparse_kernel_supported: bool = False,
) -> PruningResult:
    """Prune low-magnitude weights and estimate whether sparsity becomes speed."""
    if not 0 <= sparsity < 1:
        raise ValueError("sparsity must be in [0, 1)")

    threshold = np.quantile(np.abs(weights), sparsity)
    mask = np.abs(weights) > threshold
    remaining_ratio = float(np.mean(mask))

    # This is an idealized compute upper bound, not a latency measurement.
    speedup_upper_bound = 1 / remaining_ratio if sparse_kernel_supported else 1.0

    return PruningResult(
        sparsity=sparsity,
        remaining_ratio=remaining_ratio,
        # Excludes indices and sparse-format metadata; not an actual file size.
        nonzero_value_storage_kb=weights.size * remaining_ratio * weights.itemsize / 1024,
        illustrative_speedup_upper_bound=speedup_upper_bound,
        needs_sparse_kernel=not sparse_kernel_supported,
    )


def estimate_distillation_savings(
    teacher_params_b: float,
    student_params_b: float,
    teacher_quality: float,
    student_quality: float,
) -> Dict[str, float]:
    """Estimate model-size and quality trade-off from teacher to student."""
    if teacher_params_b <= 0 or student_params_b <= 0:
        raise ValueError("model parameter counts must be positive")
    if student_params_b > teacher_params_b:
        raise ValueError("student should be smaller than teacher")

    return {
        "size_reduction": 1 - student_params_b / teacher_params_b,
        "quality_drop": teacher_quality - student_quality,
        "relative_quality": student_quality / teacher_quality,
    }


def run_compression_comparison(seed: int = 7) -> Dict[str, object]:
    """Run a deterministic compression comparison for the chapter example."""
    rng = np.random.default_rng(seed)
    weights = rng.normal(loc=0.0, scale=0.35, size=(512, 512)).astype(np.float32)

    quantization = [
        symmetric_quantize(weights, bits=16),
        symmetric_quantize(weights, bits=8),
        symmetric_quantize(weights, bits=4),
    ]
    pruning = [
        magnitude_prune(weights, sparsity=0.30, sparse_kernel_supported=False),
        magnitude_prune(weights, sparsity=0.50, sparse_kernel_supported=True),
    ]
    distillation = estimate_distillation_savings(
        teacher_params_b=70,
        student_params_b=7,
        teacher_quality=0.86,
        student_quality=0.80,
    )

    return {
        "data_kind": "synthetic_weights_and_illustrative_quality",
        "baseline_memory_kb": weights.size * 4 / 1024,
        "quantization": quantization,
        "pruning": pruning,
        "distillation": distillation,
    }


def main() -> None:
    result = run_compression_comparison()
    print("Chapter 8.1: compression trade-offs")
    print(f"Baseline FP32 memory: {result['baseline_memory_kb']:.1f} KB")

    print("\nQuantization:")
    for item in result["quantization"]:
        print(
            f"  INT{item.bits:<2} memory={item.memory_kb:6.1f} KB "
            f"mean_error={item.mean_abs_error:.5f}"
        )

    print("\nPruning:")
    for item in result["pruning"]:
        support = "with sparse kernel" if not item.needs_sparse_kernel else "no sparse kernel"
        print(
            f"  sparsity={item.sparsity:.0%} remaining={item.remaining_ratio:.0%} "
            f"illustrative_upper_bound={item.illustrative_speedup_upper_bound:.2f}x "
            f"({support}; not measured latency)"
        )

    print("\nDistillation (illustrative quality scores, not a model evaluation):")
    for key, value in result["distillation"].items():
        print(f"  {key}: {value:.2%}")


if __name__ == "__main__":
    main()
