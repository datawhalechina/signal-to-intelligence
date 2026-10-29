"""Run the optional lightweight CNN comparison for non-stationary detection."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from cnn_baseline import train_and_evaluate_cnn
from nonstationary import ChirpConfig, evaluate_nonstationary_detectors


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"


def main() -> None:
    config = ChirpConfig(snr_db=-8.0)
    cnn = train_and_evaluate_cnn(config)
    traditional = evaluate_nonstationary_detectors(
        config, n_trials=1000, target_false_alarm=0.05, seed=242
    )
    labels = ["Global FFT\npeak", "Time-frequency\ntrajectory", "Tiny 1D CNN"]
    values = [
        traditional["spectral_peak"].detection_probability,
        traditional["time_frequency_ridge"].detection_probability,
        cnn.detection_probability,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(labels, values, color=["#4C78A8", "#59A14F", "#F28E2B"])
    axes[0].set(ylabel="Detection probability", ylim=(0, 1.02))
    axes[0].set_title("Held-out chirp detection at 5% false-alarm target")
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].plot(range(1, len(cnn.losses) + 1), cnn.losses, "o-")
    axes[1].set(xlabel="Epoch", ylabel="Binary cross-entropy loss")
    axes[1].set_title(f"Tiny CNN training ({cnn.parameter_count} parameters)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = OUTPUT_DIR / "cnn_comparison.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    result = {
        "config": vars(config),
        "spectral_peak": vars(traditional["spectral_peak"]),
        "time_frequency_ridge": vars(traditional["time_frequency_ridge"]),
        "tiny_cnn": vars(cnn),
    }
    metrics_path = OUTPUT_DIR / "cnn_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Figure: {figure_path}")
    print(f"Metrics: {metrics_path}")
    print(f"CNN parameters: {cnn.parameter_count}")
    print(f"CNN Pd/Pfa: {cnn.detection_probability:.3f}/{cnn.false_alarm_rate:.3f}")


if __name__ == "__main__":
    main()
