"""Synthetic data and detector statistics for the signal-detection project.

This module deliberately contains no plotting or command-line code, so the
experiment can be reproduced from tests and reused by later projects.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignalConfig:
    """Configuration for a sinusoid-in-white-noise detection experiment."""

    sample_length: int = 128
    frequency: float = 0.125
    amplitude: float = 1.0
    snr_db: float = -18.0


def sinusoid_template(config: SignalConfig, frequency: float | None = None) -> np.ndarray:
    """Return the known signal template used by the matched filter."""
    if config.sample_length <= 0:
        raise ValueError("sample_length must be positive")
    signal_frequency = config.frequency if frequency is None else frequency
    samples = np.arange(config.sample_length)
    return config.amplitude * np.sin(2 * np.pi * signal_frequency * samples)


def noise_standard_deviation(config: SignalConfig) -> float:
    """Convert the per-sample SNR definition to the AWGN standard deviation."""
    if not np.isfinite(config.snr_db):
        raise ValueError("snr_db must be finite")
    signal_power = np.mean(sinusoid_template(config) ** 2)
    snr_linear = 10 ** (config.snr_db / 10)
    return float(np.sqrt(signal_power / snr_linear))


def generate_observations(
    config: SignalConfig,
    n_trials: int,
    signal_present: bool,
    seed: int,
) -> np.ndarray:
    """Generate H0 or H1 observations with a deterministic local random seed."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    rng = np.random.default_rng(seed)
    observations = rng.normal(
        loc=0.0,
        scale=noise_standard_deviation(config),
        size=(n_trials, config.sample_length),
    )
    if signal_present:
        observations += sinusoid_template(config)
    return observations


def energy_statistic(observations: np.ndarray) -> np.ndarray:
    """Return one energy statistic per observation."""
    return np.sum(np.square(observations), axis=1)


def matched_filter_statistic(observations: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Return the correlation with a known template for each observation."""
    if observations.ndim != 2:
        raise ValueError("observations must be a two-dimensional array")
    if observations.shape[1] != template.shape[0]:
        raise ValueError("template length must match observation length")
    return observations @ template
