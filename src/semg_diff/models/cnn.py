from __future__ import annotations

import torch
from torch import nn


class SimpleEMGCNN(nn.Module):
    """Compact 1D-CNN for windows shaped as batch x channels x time."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: list[int] | tuple[int, ...] = (64, 128, 128),
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        for channels in hidden_channels:
            layers.extend(
                [
                    nn.Conv1d(current, int(channels), kernel_size=7, padding=3, bias=False),
                    nn.BatchNorm1d(int(channels)),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Dropout(dropout),
                ]
            )
            current = int(channels)
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(current, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))
