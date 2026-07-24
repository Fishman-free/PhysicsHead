"""Compact sibling head for support and image-plane motion proxies."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PhysicsHead(nn.Module):
    """Decode flat shared features into support and motion proxy fields.

    Args:
        in_channels: Number of shared decoder feature channels.
        motion_scale: Multiplicative scale applied after the rotated gradient.

    Inputs have shape ``(B*T, C, H, W)``. Outputs remain flat, with shapes
    ``(B*T, 1, H, W)`` and ``(B*T, 2, H, W)`` respectively.
    """

    def __init__(self, in_channels: int = 64, motion_scale: float = 5.0) -> None:
        super().__init__()
        self.motion_scale = float(motion_scale)
        self.support_head = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.stream_head = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Tanh(),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _rotated_gradient(stream: torch.Tensor) -> torch.Tensor:
        padded = F.pad(stream, (1, 1, 1, 1), mode="replicate")
        d_dy = (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]) / 2.0
        d_dx = (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]) / 2.0
        return torch.cat((d_dy, -d_dx), dim=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("features must have shape (B*T, C, H, W)")
        support = self.support_head(features)
        stream = self.stream_head(features)
        motion = self._rotated_gradient(stream) * self.motion_scale
        return support, motion


def reshape_sequence_outputs(
    support_flat: torch.Tensor,
    motion_flat: torch.Tensor,
    batch_size: int,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape flat head outputs to ``(B, T, C, H, W)`` tensors."""
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    expected = batch_size * sequence_length
    if support_flat.shape[0] != expected or motion_flat.shape[0] != expected:
        raise ValueError("flat output length does not equal batch_size * sequence_length")
    support = support_flat.reshape(batch_size, sequence_length, *support_flat.shape[1:])
    motion = motion_flat.reshape(batch_size, sequence_length, *motion_flat.shape[1:])
    return support, motion
