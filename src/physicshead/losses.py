"""Observable supervision losses for PhysicsHead proxy outputs."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MaskSupervisionLoss(nn.Module):
    """Binary cross-entropy plus half-weight per-sample Dice by default."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, support: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        clamped = support.clamp(1e-6, 1.0 - 1e-6)
        bce = F.binary_cross_entropy(clamped, target_mask, reduction="mean")
        batch_size = support.shape[0]
        support_flat = support.reshape(batch_size, -1)
        target_flat = target_mask.reshape(batch_size, -1)
        intersection = (support_flat * target_flat).sum(dim=1)
        denominator = support_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        return self.bce_weight * bce + self.dice_weight * (1.0 - dice).mean()


def _base_grid(
    batch_size: int,
    height: int,
    width: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    y = torch.linspace(-1, 1, height, device=reference.device, dtype=reference.dtype)
    x = torch.linspace(-1, 1, width, device=reference.device, dtype=reference.dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)


class WarpingLoss(nn.Module):
    """Adjacent-mask backward-warp L1 loss over the union of visible support.

    Target masks are detached, so gradients flow only through the motion proxy.
    Motion is expected in pixel-per-frame units.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, target_mask: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        batch_size, steps, _, height, width = motion.shape
        if steps < 2:
            return motion.new_zeros(1).squeeze()

        base = _base_grid(batch_size, height, width, motion)
        total = motion.new_zeros(1).squeeze()
        for step in range(steps - 1):
            current = target_mask[:, step].detach()
            following = target_mask[:, step + 1].detach()
            displacement = motion[:, step] * (2.0 / max(height, width))
            grid = base - displacement.permute(0, 2, 3, 1)
            warped = F.grid_sample(
                current,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            ink = ((current > 0.05) | (following > 0.05)).float()
            ink_area = ink.mean() + self.eps
            total = total + ((warped - following).abs() * ink).mean() / ink_area
        return total / (steps - 1)


class FlowSupervisionLoss(nn.Module):
    """Masked MSE or Smooth L1 against an offline apparent-flow target."""

    def __init__(
        self,
        loss_type: str = "mse",
        beta: float = 1.0,
        use_ink_mask: bool = True,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.beta = beta
        self.use_ink_mask = use_ink_mask

    def forward(
        self,
        motion: torch.Tensor,
        flow_target: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.loss_type == "mse":
            loss = (motion - flow_target).pow(2)
        else:
            loss = F.smooth_l1_loss(
                motion,
                flow_target,
                beta=self.beta,
                reduction="none",
            )
        if self.use_ink_mask and target_mask is not None:
            ink = (target_mask > 0.05).float().expand_as(loss)
            return (loss * ink).sum() / (ink.sum() + 1e-8)
        return loss.mean()
