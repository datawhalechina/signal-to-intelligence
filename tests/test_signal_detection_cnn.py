"""Optional tests for the lightweight CNN signal-detection baseline."""

import importlib.util
from pathlib import Path
import sys

import pytest


pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT / "projects" / "signal_detection"


def load_cnn_module():
    path = PROJECT_DIR / "cnn_baseline.py"
    spec = importlib.util.spec_from_file_location("cnn_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PROJECT_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_tiny_cnn_has_a_small_output_contract():
    cnn = load_cnn_module()
    torch = pytest.importorskip("torch")
    model = cnn.build_tiny_cnn()

    output = model(torch.zeros(4, 1, 256))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert output.shape == (4, 1)
    assert parameter_count < 5000


def test_cnn_training_uses_separate_calibration_and_test_splits():
    cnn = load_cnn_module()
    nonstationary = importlib.import_module("nonstationary")
    result = cnn.train_and_evaluate_cnn(
        nonstationary.ChirpConfig(snr_db=-6.0),
        train_per_class=400,
        validation_per_class=200,
        test_per_class=300,
        epochs=8,
    )

    assert result.losses[-1] < result.losses[0]
    assert result.parameter_count < 5000
    assert 0 <= result.false_alarm_rate <= 0.12
