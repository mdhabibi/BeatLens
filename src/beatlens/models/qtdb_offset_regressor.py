"""Compact temporal CNN that regresses sparse QTDB event offsets."""

from __future__ import annotations

import torch
from torch import nn

from beatlens.infrastructure.qtdb import TARGET_NAMES


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = 2 * dilation
        self.layers = nn.Sequential(
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layers(inputs)


class QTDBOffsetRegressor(nn.Module):
    """Regress ordered marker offsets; output is normalized to ``[-1, 1]``."""

    def __init__(self, input_leads: int = 2, max_slots: int = 8) -> None:
        super().__init__()
        self.target_names = TARGET_NAMES
        self.max_slots = max_slots
        self.encoder = nn.Sequential(
            nn.Conv1d(input_leads, 16, kernel_size=9, stride=2, padding=4),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv1d(16, 24, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            nn.Conv1d(24, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            _ResidualBlock(32, dilation=1),
            _ResidualBlock(32, dilation=2),
            nn.AdaptiveAvgPool1d(79),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 79, 192),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(192, len(TARGET_NAMES) * max_slots),
            nn.Tanh(),
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        batch = signals.shape[0]
        return self.regressor(self.encoder(signals)).view(
            batch, len(TARGET_NAMES), self.max_slots
        )
