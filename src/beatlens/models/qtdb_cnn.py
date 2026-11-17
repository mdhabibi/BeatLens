"""Small fully convolutional baseline for QTDB fiducial localization."""

from __future__ import annotations

import torch
from torch import nn

from beatlens.infrastructure.qtdb import TARGET_NAMES


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = 2 * dilation
        self.layers = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=padding, dilation=dilation),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layers(inputs)


class QTDBLocalizationCNN(nn.Module):
    """Map a two-lead beat window to five sparse fiducial heatmaps."""

    target_names = TARGET_NAMES

    def __init__(self, input_leads: int = 2, channels: int = 48) -> None:
        super().__init__()
        self.stem = nn.Conv1d(input_leads, channels, kernel_size=9, padding=4)
        self.blocks = nn.Sequential(
            ResidualTemporalBlock(channels, dilation=1),
            ResidualTemporalBlock(channels, dilation=2),
            ResidualTemporalBlock(channels, dilation=4),
            ResidualTemporalBlock(channels, dilation=8),
        )
        self.head = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, len(TARGET_NAMES), kernel_size=1),
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Return logits shaped ``(batch, targets, samples)``."""
        return self.head(self.blocks(self.stem(signals)))
