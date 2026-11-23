"""Untrained neural sequence architectures; PyTorch optional (deep extra)."""

from __future__ import annotations

import torch
from torch import nn

from ensemble_learning import HORIZONS

NEURAL_ARCHITECTURES = ("mlp", "cnn1d", "lstm", "gru")


class SequenceEncoder(nn.Module):
    """One representation per historical feature window [batch, time, features]."""

    def __init__(
        self, architecture: str, input_size: int, hidden_size: int = 32,
        sequence_length: int | None = None,
    ):
        super().__init__()
        if architecture not in NEURAL_ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture}")
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input and hidden sizes must be positive")
        if architecture == "mlp" and (sequence_length is None or sequence_length < 1):
            raise ValueError("MLP requires positive sequence_length")
        self.architecture = architecture
        self.input_size = input_size
        self.sequence_length = sequence_length
        if architecture == "mlp":
            self.encoder = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(input_size * sequence_length, hidden_size),
                nn.ReLU(),
            )
        elif architecture == "cnn1d":
            self.encoder = nn.Sequential(
                nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
        elif architecture == "lstm":
            self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] == 0 or x.shape[2] != self.input_size:
            raise ValueError("Expected [batch, history>0, input_size] tensor")
        if self.architecture == "mlp":
            if x.shape[1] != self.sequence_length:
                raise ValueError("MLP history differs from configured sequence_length")
            return self.encoder(x)
        if self.architecture == "cnn1d":
            return self.encoder(x.transpose(1, 2))
        sequence, _ = self.encoder(x)
        return sequence[:, -1, :]


class IndependentSequenceClassifier(nn.Module):
    """Single-horizon classifier returning two logits."""

    def __init__(
        self, architecture: str, input_size: int,
        hidden_size: int = 32, sequence_length: int | None = None,
    ):
        super().__init__()
        self.encoder = SequenceEncoder(architecture, input_size, hidden_size, sequence_length)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class MultiTaskSequenceClassifier(nn.Module):
    """Shared encoder with independent 5, 10, 15 minute heads."""

    def __init__(
        self, architecture: str, input_size: int,
        hidden_size: int = 32, sequence_length: int | None = None,
        horizons: tuple[int, ...] = HORIZONS,
    ):
        super().__init__()
        if not horizons or len(horizons) != len(set(horizons)):
            raise ValueError("horizons must be non-empty and unique")
        self.encoder = SequenceEncoder(architecture, input_size, hidden_size, sequence_length)
        self.heads = nn.ModuleDict({str(h): nn.Linear(hidden_size, 2) for h in horizons})

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        features = self.encoder(x)
        return {int(h): head(features) for h, head in self.heads.items()}
