"""Non-stationary chirp experiment and time-frequency detection statistics."""

from dataclasses import dataclass

import numpy as np

from evaluate import DetectorMetrics, evaluate_statistic


@dataclass(frozen=True)
class ChirpConfig:
    """Configuration for a linear chirp in additive white Gaussian noise."""

    sample_length: int = 256
    start_frequency: float = 0.04
    end_frequency: float = 0.28
    amplitude: float = 1.0
    snr_db: float = -10.0
    frame_length: int = 32
    hop_length: int = 16


def chirp_template(config: ChirpConfig, end_frequency: float | None = None) -> np.ndarray:
    """Generate a linear-frequency chirp using normalized cycles per sample."""
    if config.sample_length <= 1:
        raise ValueError("sample_length must be greater than one")
    target_end = config.end_frequency if end_frequency is None else end_frequency
    samples = np.arange(config.sample_length)
    chirp_rate = (target_end - config.start_frequency) / (config.sample_length - 1)
    phase = 2 * np.pi * (
        config.start_frequency * samples + 0.5 * chirp_rate * np.square(samples)
    )
    return config.amplitude * np.sin(phase)


def generate_chirp_observations(
    config: ChirpConfig,
    n_trials: int,
    signal_present: bool,
    seed: int,
) -> np.ndarray:
    """Generate reproducible H0 or H1 observations for the chirp task."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    template = chirp_template(config)
    signal_power = np.mean(np.square(template))
    noise_std = np.sqrt(signal_power / (10 ** (config.snr_db / 10)))
    rng = np.random.default_rng(seed)
    observations = rng.normal(0.0, noise_std, (n_trials, config.sample_length))
    if signal_present:
        observations += template
    return observations


def global_spectral_peak_statistic(observations: np.ndarray) -> np.ndarray:
    """Detect a signal by its largest global FFT bin, ignoring temporal evolution."""
    spectrum = np.fft.rfft(observations, axis=1)
    return np.max(np.square(np.abs(spectrum[:, 1:])), axis=1)


def time_frequency_ridge_statistic(
    observations: np.ndarray,
    config: ChirpConfig,
    end_frequency: float | None = None,
) -> np.ndarray:
    """Accumulate STFT power along an expected linear time-frequency trajectory."""
    if config.frame_length > config.sample_length:
        raise ValueError("frame_length cannot exceed sample_length")
    starts = np.arange(0, config.sample_length - config.frame_length + 1, config.hop_length)
    window = np.hanning(config.frame_length)
    frame_centers = starts + (config.frame_length - 1) / 2
    target_end = config.end_frequency if end_frequency is None else end_frequency
    expected_frequencies = config.start_frequency + (
        (target_end - config.start_frequency) * frame_centers / (config.sample_length - 1)
    )
    frequency_bins = np.clip(
        np.rint(expected_frequencies * config.frame_length).astype(int),
        0,
        config.frame_length // 2,
    )

    ridge_power = np.zeros(observations.shape[0])
    for start, frequency_bin in zip(starts, frequency_bins):
        frames = observations[:, start:start + config.frame_length] * window
        spectrum = np.fft.rfft(frames, axis=1)
        ridge_power += np.square(np.abs(spectrum[:, frequency_bin]))
    return ridge_power


def evaluate_nonstationary_detectors(
    config: ChirpConfig,
    n_trials: int = 2000,
    target_false_alarm: float = 0.05,
    ridge_end_frequency: float | None = None,
    seed: int = 42,
) -> dict[str, DetectorMetrics]:
    """Compare a global spectrum detector with a time-frequency trajectory detector."""
    null_observations = generate_chirp_observations(config, n_trials, False, seed)
    signal_observations = generate_chirp_observations(config, n_trials, True, seed + 1)
    return {
        "spectral_peak": evaluate_statistic(
            global_spectral_peak_statistic(null_observations),
            global_spectral_peak_statistic(signal_observations),
            target_false_alarm,
        ),
        "time_frequency_ridge": evaluate_statistic(
            time_frequency_ridge_statistic(null_observations, config, ridge_end_frequency),
            time_frequency_ridge_statistic(signal_observations, config, ridge_end_frequency),
            target_false_alarm,
        ),
    }


def sweep_chirp_snr(
    snr_values: list[float] | np.ndarray,
    n_trials: int = 2000,
    target_false_alarm: float = 0.05,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Evaluate global and time-frequency detectors over a chirp SNR sweep."""
    snr_array = np.asarray(snr_values, dtype=float)
    results = {
        "snr_db": snr_array,
        "spectral_peak_pd": np.empty(len(snr_array)),
        "time_frequency_ridge_pd": np.empty(len(snr_array)),
    }
    for index, snr_db in enumerate(snr_array):
        metrics = evaluate_nonstationary_detectors(
            ChirpConfig(snr_db=float(snr_db)),
            n_trials=n_trials,
            target_false_alarm=target_false_alarm,
            seed=seed + index * 10,
        )
        results["spectral_peak_pd"][index] = metrics["spectral_peak"].detection_probability
        results["time_frequency_ridge_pd"][index] = metrics[
            "time_frequency_ridge"
        ].detection_probability
    return results
