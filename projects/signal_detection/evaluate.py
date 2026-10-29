"""Evaluation helpers for comparing signal detectors at a fixed false-alarm rate."""

from dataclasses import dataclass

import numpy as np

from experiment import (
    SignalConfig,
    energy_statistic,
    generate_observations,
    matched_filter_statistic,
    sinusoid_template,
)


@dataclass(frozen=True)
class DetectorMetrics:
    """Detection probability and empirical false-alarm rate for one detector."""

    threshold: float
    detection_probability: float
    false_alarm_rate: float


def calibrate_threshold(null_statistics: np.ndarray, target_false_alarm: float) -> float:
    """Choose a threshold from H0 samples for a target upper-tail false-alarm rate."""
    if not 0 < target_false_alarm < 1:
        raise ValueError("target_false_alarm must be between 0 and 1")
    return float(np.quantile(null_statistics, 1 - target_false_alarm))


def evaluate_statistic(
    null_statistics: np.ndarray,
    signal_statistics: np.ndarray,
    target_false_alarm: float,
) -> DetectorMetrics:
    """Calibrate a statistic on H0 and report its empirical H0/H1 outcomes."""
    threshold = calibrate_threshold(null_statistics, target_false_alarm)
    return DetectorMetrics(
        threshold=threshold,
        detection_probability=float(np.mean(signal_statistics > threshold)),
        false_alarm_rate=float(np.mean(null_statistics > threshold)),
    )


def evaluate_detectors(
    config: SignalConfig,
    n_trials: int = 2000,
    target_false_alarm: float = 0.05,
    template_frequency: float | None = None,
    seed: int = 42,
) -> dict[str, DetectorMetrics]:
    """Compare energy detection and a (possibly mismatched) matched filter."""
    null_observations = generate_observations(
        config, n_trials, signal_present=False, seed=seed
    )
    signal_observations = generate_observations(
        config, n_trials, signal_present=True, seed=seed + 1
    )
    template = sinusoid_template(config, frequency=template_frequency)

    return {
        "energy": evaluate_statistic(
            energy_statistic(null_observations),
            energy_statistic(signal_observations),
            target_false_alarm,
        ),
        "matched_filter": evaluate_statistic(
            matched_filter_statistic(null_observations, template),
            matched_filter_statistic(signal_observations, template),
            target_false_alarm,
        ),
    }


def sweep_snr(
    snr_values: list[float] | np.ndarray,
    n_trials: int = 2000,
    target_false_alarm: float = 0.05,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Evaluate both detectors over an SNR sweep using independent, reproducible trials."""
    snr_array = np.asarray(snr_values, dtype=float)
    results = {
        "snr_db": snr_array,
        "energy_pd": np.empty(len(snr_array)),
        "energy_pfa": np.empty(len(snr_array)),
        "matched_filter_pd": np.empty(len(snr_array)),
        "matched_filter_pfa": np.empty(len(snr_array)),
    }
    for index, snr_db in enumerate(snr_array):
        metrics = evaluate_detectors(
            SignalConfig(snr_db=float(snr_db)),
            n_trials=n_trials,
            target_false_alarm=target_false_alarm,
            seed=seed + index * 10,
        )
        results["energy_pd"][index] = metrics["energy"].detection_probability
        results["energy_pfa"][index] = metrics["energy"].false_alarm_rate
        results["matched_filter_pd"][index] = metrics["matched_filter"].detection_probability
        results["matched_filter_pfa"][index] = metrics["matched_filter"].false_alarm_rate
    return results
