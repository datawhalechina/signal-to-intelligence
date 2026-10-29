"""Behavior tests for the first chapter project."""

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT / "projects" / "signal_detection"


def load_project_module(name: str):
    module_path = PROJECT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PROJECT_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_matched_filter_outperforms_energy_when_template_is_known():
    evaluate = load_project_module("evaluate")
    experiment = load_project_module("experiment")

    metrics = evaluate.evaluate_detectors(
        experiment.SignalConfig(snr_db=-20.0), n_trials=1000, target_false_alarm=0.05
    )

    assert metrics["matched_filter"].detection_probability > metrics["energy"].detection_probability
    assert abs(metrics["energy"].false_alarm_rate - 0.05) < 0.02
    assert abs(metrics["matched_filter"].false_alarm_rate - 0.05) < 0.02


def test_template_mismatch_exposes_a_matched_filter_limit():
    evaluate = load_project_module("evaluate")
    experiment = load_project_module("experiment")
    config = experiment.SignalConfig(snr_db=-12.0)

    matched = evaluate.evaluate_detectors(config, n_trials=1000, template_frequency=0.125)
    mismatched = evaluate.evaluate_detectors(config, n_trials=1000, template_frequency=0.145)

    assert matched["matched_filter"].detection_probability > 0.8
    assert mismatched["matched_filter"].detection_probability < 0.3


def test_snr_sweep_returns_aligned_finite_results():
    evaluate = load_project_module("evaluate")
    result = evaluate.sweep_snr([-24.0, -18.0, -12.0], n_trials=300)

    assert np.allclose(result["snr_db"], [-24.0, -18.0, -12.0])
    for values in result.values():
        assert np.all(np.isfinite(values))


def test_stricter_false_alarm_budget_raises_the_detection_threshold():
    evaluate = load_project_module("evaluate")
    experiment = load_project_module("experiment")
    config = experiment.SignalConfig(snr_db=-16.0)

    strict = evaluate.evaluate_detectors(
        config, n_trials=1000, target_false_alarm=0.01
    )
    relaxed = evaluate.evaluate_detectors(
        config, n_trials=1000, target_false_alarm=0.10
    )

    assert strict["matched_filter"].threshold > relaxed["matched_filter"].threshold
    assert (
        strict["matched_filter"].detection_probability
        < relaxed["matched_filter"].detection_probability
    )


def test_time_frequency_trajectory_outperforms_global_spectrum_for_a_chirp():
    nonstationary = load_project_module("nonstationary")
    metrics = nonstationary.evaluate_nonstationary_detectors(
        nonstationary.ChirpConfig(snr_db=-12.0), n_trials=1000
    )

    assert (
        metrics["time_frequency_ridge"].detection_probability
        > metrics["spectral_peak"].detection_probability + 0.3
    )
    assert abs(metrics["time_frequency_ridge"].false_alarm_rate - 0.05) < 0.02


def test_wrong_time_frequency_trajectory_exposes_the_prior_mismatch():
    nonstationary = load_project_module("nonstationary")
    config = nonstationary.ChirpConfig(snr_db=-8.0)

    matched = nonstationary.evaluate_nonstationary_detectors(config, n_trials=1000)
    mismatched = nonstationary.evaluate_nonstationary_detectors(
        config, n_trials=1000, ridge_end_frequency=0.16
    )

    assert matched["time_frequency_ridge"].detection_probability > 0.85
    assert (
        matched["time_frequency_ridge"].detection_probability
        > mismatched["time_frequency_ridge"].detection_probability + 0.4
    )
