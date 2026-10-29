"""Run the reproducible signal-detection comparison project.

Example:
    python projects/signal_detection/main.py --trials 2000
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluate import evaluate_detectors, sweep_snr
from experiment import SignalConfig
from nonstationary import ChirpConfig, evaluate_nonstationary_detectors, sweep_chirp_snr


DEFAULT_SNR_VALUES = np.array([-26, -23, -20, -17, -14, -11, -8], dtype=float)
DEFAULT_CHIRP_SNR_VALUES = np.array([-18, -15, -12, -9, -6, -3], dtype=float)
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"


def run_project(n_trials: int = 2000, target_false_alarm: float = 0.05, seed: int = 42):
    """Run the main SNR sweep and one template-mismatch failure case."""
    sweep = sweep_snr(
        DEFAULT_SNR_VALUES,
        n_trials=n_trials,
        target_false_alarm=target_false_alarm,
        seed=seed,
    )
    failure_config = SignalConfig(snr_db=-12.0)
    matched = evaluate_detectors(
        failure_config,
        n_trials=n_trials,
        target_false_alarm=target_false_alarm,
        seed=seed + 1000,
    )
    mismatched = evaluate_detectors(
        failure_config,
        n_trials=n_trials,
        target_false_alarm=target_false_alarm,
        template_frequency=0.145,
        seed=seed + 1000,
    )
    chirp_sweep = sweep_chirp_snr(
        DEFAULT_CHIRP_SNR_VALUES,
        n_trials=n_trials,
        target_false_alarm=target_false_alarm,
        seed=seed + 2000,
    )
    chirp_config = ChirpConfig(snr_db=-8.0)
    chirp_matched = evaluate_nonstationary_detectors(
        chirp_config,
        n_trials=n_trials,
        target_false_alarm=target_false_alarm,
        seed=seed + 3000,
    )
    chirp_mismatched = evaluate_nonstationary_detectors(
        chirp_config,
        n_trials=n_trials,
        target_false_alarm=target_false_alarm,
        ridge_end_frequency=0.16,
        seed=seed + 3000,
    )
    return {
        "sweep": sweep,
        "matched": matched,
        "mismatched": mismatched,
        "chirp_sweep": chirp_sweep,
        "chirp_matched": chirp_matched,
        "chirp_mismatched": chirp_mismatched,
    }


def plot_results(result: dict, output_path: Path) -> Path:
    """Create the project main figure: SNR comparison plus failure-case contrast."""
    sweep = result["sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].plot(sweep["snr_db"], sweep["energy_pd"], "o-", label="Energy detector")
    axes[0].plot(
        sweep["snr_db"],
        sweep["matched_filter_pd"],
        "s-",
        label="Matched filter (known template)",
    )
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="SNR (dB)",
        ylabel="Detection probability",
        ylim=(-0.02, 1.02),
    )
    axes[0].set_title("Detection probability at fixed false-alarm target")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    failure_labels = ["Energy", "Matched\ntemplate", "Mismatched\ntemplate"]
    failure_values = [
        result["matched"]["energy"].detection_probability,
        result["matched"]["matched_filter"].detection_probability,
        result["mismatched"]["matched_filter"].detection_probability,
    ]
    axes[1].bar(failure_labels, failure_values, color=["#4C78A8", "#59A14F", "#E15759"])
    axes[1].set(ylabel="Detection probability", ylim=(0, 1.02))
    axes[1].set_title("Failure case: template frequency mismatch at -12 dB")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def serialize_result(result: dict) -> dict:
    """Convert NumPy values and metric objects into JSON-compatible data."""
    sweep = {name: values.tolist() for name, values in result["sweep"].items()}
    serialized = {
        "sweep": sweep,
        "matched": {name: vars(metrics) for name, metrics in result["matched"].items()},
        "mismatched": {name: vars(metrics) for name, metrics in result["mismatched"].items()},
    }
    serialized["chirp_sweep"] = {
        name: values.tolist() for name, values in result["chirp_sweep"].items()
    }
    for key in ("chirp_matched", "chirp_mismatched"):
        serialized[key] = {name: vars(metrics) for name, metrics in result[key].items()}
    return serialized


def plot_nonstationary_results(result: dict, output_path: Path) -> Path:
    """Plot the SNR sweep and trajectory-mismatch failure case for chirp detection."""
    sweep = result["chirp_sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(sweep["snr_db"], sweep["spectral_peak_pd"], "o-", label="Global FFT peak")
    axes[0].plot(
        sweep["snr_db"],
        sweep["time_frequency_ridge_pd"],
        "s-",
        label="Time-frequency trajectory",
    )
    axes[0].set(xlabel="SNR (dB)", ylabel="Detection probability", ylim=(-0.02, 1.02))
    axes[0].set_title("Non-stationary chirp detection at fixed false-alarm target")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    labels = ["Global FFT\npeak", "Matched\ntrajectory", "Mismatched\ntrajectory"]
    values = [
        result["chirp_matched"]["spectral_peak"].detection_probability,
        result["chirp_matched"]["time_frequency_ridge"].detection_probability,
        result["chirp_mismatched"]["time_frequency_ridge"].detection_probability,
    ]
    axes[1].bar(labels, values, color=["#4C78A8", "#59A14F", "#E15759"])
    axes[1].set(ylabel="Detection probability", ylim=(0, 1.02))
    axes[1].set_title("Failure case: incorrect time-frequency trajectory at -8 dB")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare synthetic signal detectors.")
    parser.add_argument(
        "--trials", type=int, default=2000, help="Monte Carlo trials per hypothesis"
    )
    parser.add_argument("--target-pfa", type=float, default=0.05, help="Target false-alarm rate")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = run_project(args.trials, args.target_pfa)
    figure_path = plot_results(result, args.output_dir / "detection_comparison.png")
    nonstationary_figure_path = plot_nonstationary_results(
        result, args.output_dir / "nonstationary_detection.png"
    )
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(serialize_result(result), indent=2), encoding="utf-8")

    print(f"Main figure: {figure_path}")
    print(f"Non-stationary figure: {nonstationary_figure_path}")
    print(f"Metrics: {result_path}")
    print(
        "At -12 dB, known-template Pd: "
        f"{result['matched']['matched_filter'].detection_probability:.3f}"
    )
    print(
        "At -12 dB, mismatched-template Pd: "
        f"{result['mismatched']['matched_filter'].detection_probability:.3f}"
    )
    print(
        "At -8 dB chirp, global-spectrum Pd: "
        f"{result['chirp_matched']['spectral_peak'].detection_probability:.3f}"
    )
    print(
        "At -8 dB chirp, matched-trajectory Pd: "
        f"{result['chirp_matched']['time_frequency_ridge'].detection_probability:.3f}"
    )
    print(
        "At -8 dB chirp, mismatched-trajectory Pd: "
        f"{result['chirp_mismatched']['time_frequency_ridge'].detection_probability:.3f}"
    )


if __name__ == "__main__":
    main()
