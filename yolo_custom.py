"""Custom YOLO modules used by this project."""

from __future__ import annotations

import torch
import torch.nn as nn


class SingleChannelSpatialAttention(nn.Module):
    """Single-channel residual spatial attention for YOLO feature maps."""

    def __init__(self, channels: int, kernel_size: int = 7, residual: bool = True) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.attn = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * self.attn(x)
        return x + y if self.residual else y


def register_custom_modules() -> None:
    """Register custom modules with Ultralytics' YAML parser."""
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    modules.SingleChannelSpatialAttention = SingleChannelSpatialAttention
    tasks.SingleChannelSpatialAttention = SingleChannelSpatialAttention
