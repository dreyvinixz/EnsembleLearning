"""Optional PyTorch encoders, not yet integrated into the experiment runner.

Import this module only after installing the [deep] extra. The forward methods
accept strictly causal sequences assembled by a future audited window builder.
No claim of a trained or validated GRU is made here.
"""

from __future__ import annotations

import torch
from torch import nn

from ensemble_learning import HORIZONS


class IndependentGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.encoder = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError("GRU expects (batch, sequence_length>0, features)")
        sequence, _ = self.encoder(x)
        return self.head(sequence[:, -1, :])


class MultiTaskGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        horizons: tuple[int, ...] = HORIZONS,
    ):
        super().__init__()
        self.encoder = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.heads = nn.ModuleDict({str(h): nn.Linear(hidden_size, 2) for h in horizons})

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError("GRU expects (batch, sequence_length>0, features)")
        sequence, _ = self.encoder(x)
        encoded = sequence[:, -1, :]
        return {int(h): layer(encoded) for h, layer in self.heads.items()}
