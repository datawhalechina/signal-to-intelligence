"""
实验4.1：自注意力机制
对应章节：第4章 - Transformer详解
目标：用 NumPy 实现简单的多头自注意力，并可视化注意力权重
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# ============ 配置 ============
SEQ_LEN = 8
D_MODEL = 32
NUM_HEADS = 4
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "assets/ch04_self_attention.png"


def _load_attention_core():
    """Load the local core module for direct scripts and path-based imports."""
    module_path = Path(__file__).with_name("attention_core.py")
    spec = spec_from_file_location("ch04_attention_core", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_CORE = _load_attention_core()

# Backward-compatible public names used by the existing notebook and tests.
scaled_dot_product_attention = _CORE.scaled_dot_product_attention
layer_norm = _CORE.layer_norm
feed_forward_network = _CORE.feed_forward_network


def build_synthetic_sequence(seq_len, d_model, rng=None):
    if d_model < 4:
        raise ValueError("d_model must be at least 4")

    rng = rng or np.random.default_rng()
    positions = np.arange(seq_len)
    x = np.zeros((1, seq_len, d_model))

    x[0, :, 0] = positions / seq_len
    x[0, :, 1] = np.sin(positions)
    x[0, :, 2] = np.cos(positions)
    x[0, :, 3] = (positions % 2) * 2 - 1
    x[0, :, 4:] = 0.05 * rng.standard_normal((seq_len, d_model - 4))
    return x


def build_demo_projection_matrices(d_model=D_MODEL, num_heads=NUM_HEADS):
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    d_head = d_model // num_heads
    w_q = np.zeros((d_model, d_model))
    w_k = np.zeros((d_model, d_model))
    w_v = np.eye(d_model)
    w_o = np.eye(d_model)

    for h in range(num_heads):
        start = h * d_head
        w_q[start, start] = 2.0
        w_k[start, start] = 2.0
        w_q[start + 1, start + 1] = 1.5
        w_k[start + 1, start + 1] = 1.5

    return w_q, w_k, w_v, w_o


def multi_head_attention(X, W_q, W_k, W_v, W_o, num_heads=NUM_HEADS):
    batch_size, seq_len, _ = X.shape
    d_model = W_q.shape[1]
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    d_head = d_model // num_heads

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    Q = Q.reshape(batch_size, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)

    attn_outputs = []
    attn_weights_list = []
    for h in range(num_heads):
        output, attn_weights = scaled_dot_product_attention(Q[0, h], K[0, h], V[0, h])
        attn_outputs.append(output)
        attn_weights_list.append(attn_weights)

    attn_output = np.concatenate(attn_outputs, axis=-1)
    output = attn_output @ W_o
    return output, attn_weights_list


def run_experiment(seq_len=SEQ_LEN, d_model=D_MODEL, num_heads=NUM_HEADS, seed=42):
    rng = np.random.RandomState(seed)
    x = build_synthetic_sequence(seq_len, d_model, rng=rng)
    w_q, w_k, w_v, w_o = build_demo_projection_matrices(d_model, num_heads)
    output, attn_weights_list = multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads)
    avg_attn_weights = np.mean(np.array(attn_weights_list), axis=0)

    return {
        "X": x,
        "output": output,
        "attention_weights": attn_weights_list,
        "average_attention_weights": avg_attn_weights,
        "num_heads": num_heads,
        "d_model": d_model,
        "d_head": d_model // num_heads,
        "seq_len": seq_len,
    }


def print_summary(result):
    print("=" * 70)
    print("Self-Attention")
    print("=" * 70)
    print(f"Sequence length: {result['seq_len']}")
    print(f"Model dimension: {result['d_model']}")
    print(f"Number of heads: {result['num_heads']}")
    print(f"Head dimension: {result['d_head']}")
    print()

    print("Input shape:")
    print("-" * 70)
    print(f"X: {result['X'].shape}")
    print()

    print("Output shape:")
    print("-" * 70)
    print(f"output: {result['output'].shape}")
    print()

    print("Attention weight statistics:")
    print("-" * 70)
    for h, weights in enumerate(result["attention_weights"]):
        print(
            f"Head {h}: min={weights.min():.4f}, "
            f"max={weights.max():.4f}, mean={weights.mean():.4f}"
        )
    print()

    avg_attn_weights = result["average_attention_weights"]
    print("Average attention weights (across heads):")
    print("-" * 70)
    print(avg_attn_weights.round(4))
    print()

    print("Attention pattern analysis:")
    print("-" * 70)
    for i in range(result["seq_len"]):
        top_k = 3
        top_indices = np.argsort(avg_attn_weights[i])[-top_k:][::-1]
        top_weights = avg_attn_weights[i, top_indices]
        print(f"Position {i} attends most to: {top_indices} (weights: {top_weights.round(4)})")
    print()

    print("=" * 70)


def plot_attention(result, output_path=OUTPUT_PATH):
    seq_len = result["seq_len"]
    token_positions = np.arange(seq_len)
    avg_attn_weights = result["average_attention_weights"]
    attn_weights_list = result["attention_weights"]
    num_heads = result["num_heads"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    im = ax.imshow(avg_attn_weights, cmap="YlOrRd", aspect="auto")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_xticks(token_positions)
    ax.set_yticks(token_positions)
    ax.set_title("Average Attention Weights")
    plt.colorbar(im, ax=ax)

    for h in range(min(2, num_heads)):
        ax = axes[1, h]
        im = ax.imshow(attn_weights_list[h], cmap="YlOrRd", aspect="auto")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        ax.set_xticks(token_positions)
        ax.set_yticks(token_positions)
        ax.set_title(f"Head {h} Attention Weights")
        plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    for h in range(num_heads):
        weights_flat = attn_weights_list[h].flatten()
        ax.hist(weights_flat, bins=25, alpha=0.45, label=f"Head {h}")
    ax.set_xlabel("Attention Weight")
    ax.set_ylabel("Frequency")
    ax.set_title("Attention Weight Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    result = run_experiment()
    print_summary(result)
    output_path = plot_attention(result)
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
