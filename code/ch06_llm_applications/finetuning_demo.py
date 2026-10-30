"""
实验6.2：微调方法参数量对比
对应章节：第6章 6.2 微调：让 LLM 适配你的任务
目标：在明确结构假设下，可视化全量微调与 LoRA 的参数量差异
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_PATH = "assets/ch06_lora_parameters.png"

# ── 参数量计算 ────────────────────────────────────────────────────────────────

# 以 7B 模型为例（近似 LLaMA-2-7B 结构）
# 注意力层：q_proj, k_proj, v_proj, o_proj，每层 4 个矩阵
# 模型维度 d=4096，层数 32
D_MODEL = 4096
N_LAYERS = 32
N_ATTN_MATRICES = 4   # q, k, v, o

total_params = 7_000_000_000  # rounded reference size, not derived from the matrices above

# LoRA 只作用于注意力层的 4 个矩阵
# 每个矩阵 [d, d]，LoRA 参数 = 2 * d * r（A 矩阵 + B 矩阵）
lora_ranks = [4, 8, 16, 32, 64]

def lora_params(rank):
    """计算 LoRA 可训练参数量"""
    per_matrix = 2 * D_MODEL * rank
    return N_LAYERS * N_ATTN_MATRICES * per_matrix

methods = ["Full Fine-tuning"] + [f"LoRA r={r}" for r in lora_ranks]
params  = [total_params] + [lora_params(r) for r in lora_ranks]
ratios  = [p / total_params * 100 for p in params]

# ── 绘图 ──────────────────────────────────────────────────────────────────────

def plot_lora_parameters(output_path=OUTPUT_PATH):
    """保存 LoRA 参数量对比图。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Fine-tuning Parameter Comparison: Full vs LoRA (7B Model)",
        fontsize=13,
        fontweight="bold",
    )

    colors = ["#e74c3c"] + ["#3498db"] * len(lora_ranks)

    # 左图：绝对参数量（对数坐标）
    ax1 = axes[0]
    bars = ax1.bar(
        methods,
        [p / 1e9 for p in params],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    ax1.set_yscale("log")
    ax1.set_ylabel("Trainable Parameters (Billions)", fontsize=11)
    ax1.set_title("Absolute Parameter Count (log scale)", fontsize=11)
    ax1.tick_params(axis="x", rotation=30)

    for bar, p in zip(bars, params):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.3,
            f"{p/1e9:.2f}B" if p >= 1e9 else f"{p/1e6:.1f}M",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    # 右图：参数比例（线性坐标）
    ax2 = axes[1]
    bars2 = ax2.bar(methods, ratios, color=colors, edgecolor="white", linewidth=0.8)
    ax2.set_ylabel("Trainable Parameters (%)", fontsize=11)
    ax2.set_title("Percentage of Total Parameters", fontsize=11)
    ax2.tick_params(axis="x", rotation=30)

    for bar, r in zip(bars2, ratios):
        label = f"{r:.1f}%" if r >= 0.1 else f"{r:.3f}%"
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"图表已保存：{output_path}")

if __name__ == "__main__":
    plot_lora_parameters()
    print("第6章：微调参数量对比演示\n")
    print(f"{'方法':<20} {'可训练参数':>15} {'占比':>10}")
    print("-" * 48)
    for method, p, r in zip(methods, params, ratios):
        print(f"{method:<20} {p:>15,.0f} {r:>9.3f}%")
