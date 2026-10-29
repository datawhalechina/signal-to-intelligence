"""Reusable numerical building blocks for Chapter 4 Transformer experiments.

This module intentionally contains no plotting, console output, or file-system
side effects. Demo scripts and future projects can use these functions while
keeping their own experiment and visualization logic separate.
"""

from __future__ import annotations

import numpy as np


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Apply softmax over the final axis with numerical stabilization."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return attention output and weights for two-dimensional Q, K and V."""
    if query.ndim != 2 or key.ndim != 2 or value.ndim != 2:
        raise ValueError("query, key, and value must be two-dimensional arrays")
    if query.shape[1] != key.shape[1]:
        raise ValueError("query and key must have the same feature dimension")
    if key.shape[0] != value.shape[0]:
        raise ValueError("key and value must have the same sequence length")

    scores = query @ key.T / np.sqrt(query.shape[-1])
    if mask is not None:
        if mask.shape != scores.shape:
            raise ValueError("mask shape must match the attention score matrix")
        scores = scores + mask

    weights = stable_softmax(scores)
    return weights @ value, weights


def causal_mask(sequence_length: int, masked_value: float = -1e9) -> np.ndarray:
    """Create an additive mask that blocks attention to future positions."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    return np.triu(np.full((sequence_length, sequence_length), masked_value), k=1)


def layer_norm(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Normalize each final-axis feature vector without learnable parameters."""
    mean = np.mean(values, axis=-1, keepdims=True)
    std = np.std(values, axis=-1, keepdims=True)
    return (values - mean) / (std + epsilon)


def feed_forward_network(
    values: np.ndarray,
    weight_1: np.ndarray,
    bias_1: np.ndarray,
    weight_2: np.ndarray,
    bias_2: np.ndarray,
) -> np.ndarray:
    """Apply the two-layer ReLU feed-forward block used by the demo."""
    hidden = np.maximum(0, values @ weight_1 + bias_1)
    return hidden @ weight_2 + bias_2
