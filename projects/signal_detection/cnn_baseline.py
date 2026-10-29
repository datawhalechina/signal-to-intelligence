"""Optional lightweight 1D CNN baseline for the non-stationary chirp task."""

from dataclasses import dataclass

import numpy as np

from nonstationary import ChirpConfig, generate_chirp_observations


@dataclass(frozen=True)
class CnnDetectionResult:
    """Held-out CNN metrics after threshold calibration on validation noise."""

    threshold: float
    detection_probability: float
    false_alarm_rate: float
    parameter_count: int
    losses: tuple[float, ...]


def _load_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "The CNN baseline requires `pip install -e \".[ml]\"`."
        ) from exc
    return torch, nn


def build_tiny_cnn():
    """Build a small 1D CNN with fewer than 5,000 trainable parameters."""
    _, nn = _load_torch()
    return nn.Sequential(
        nn.Conv1d(1, 8, kernel_size=9, padding=4),
        nn.ReLU(),
        nn.MaxPool1d(4),
        nn.Conv1d(8, 16, kernel_size=7, padding=3),
        nn.ReLU(),
        nn.AdaptiveMaxPool1d(16),
        nn.Flatten(),
        nn.Linear(16 * 16, 1),
    )


def _balanced_dataset(config: ChirpConfig, samples_per_class: int, seed: int):
    null = generate_chirp_observations(config, samples_per_class, False, seed)
    signal = generate_chirp_observations(config, samples_per_class, True, seed + 1)
    observations = np.concatenate([null, signal])
    labels = np.concatenate([np.zeros(samples_per_class), np.ones(samples_per_class)])
    rng = np.random.default_rng(seed + 2)
    order = rng.permutation(len(labels))
    return observations[order], labels[order]


def _predict_scores(model, observations: np.ndarray, scale: float) -> np.ndarray:
    torch, _ = _load_torch()
    inputs = torch.tensor(observations / scale, dtype=torch.float32).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(inputs).squeeze(1)).cpu().numpy()


def train_and_evaluate_cnn(
    config: ChirpConfig,
    train_per_class: int = 1200,
    validation_per_class: int = 500,
    test_per_class: int = 1000,
    epochs: int = 30,
    learning_rate: float = 0.003,
    target_false_alarm: float = 0.05,
    seed: int = 42,
) -> CnnDetectionResult:
    """Train, calibrate, and evaluate the CNN on independent deterministic splits."""
    if min(train_per_class, validation_per_class, test_per_class, epochs) <= 0:
        raise ValueError("dataset sizes and epochs must be positive")
    if not 0 < target_false_alarm < 1:
        raise ValueError("target_false_alarm must be between 0 and 1")

    torch, nn = _load_torch()
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    train_x, train_y = _balanced_dataset(config, train_per_class, seed)
    scale = float(np.std(train_x))
    inputs = torch.tensor(train_x / scale, dtype=torch.float32).unsqueeze(1)
    labels = torch.tensor(train_y, dtype=torch.float32)
    model = build_tiny_cnn()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    losses = []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(inputs).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    validation_null = generate_chirp_observations(
        config, validation_per_class, False, seed + 100
    )
    validation_scores = _predict_scores(model, validation_null, scale)
    threshold = float(np.quantile(validation_scores, 1 - target_false_alarm))
    test_null = generate_chirp_observations(config, test_per_class, False, seed + 200)
    test_signal = generate_chirp_observations(config, test_per_class, True, seed + 201)
    null_scores = _predict_scores(model, test_null, scale)
    signal_scores = _predict_scores(model, test_signal, scale)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return CnnDetectionResult(
        threshold=threshold,
        detection_probability=float(np.mean(signal_scores > threshold)),
        false_alarm_rate=float(np.mean(null_scores > threshold)),
        parameter_count=parameter_count,
        losses=tuple(losses),
    )
