"""ChangeScorer — detects where something changed between a 2022 and 2024 tile.

The scorer computes the element-wise feature delta between the two years and
passes it through a small convolutional head to produce a per-pixel change
probability map.

This signal is used **only for new construction detection** (areas that were
NOT already construction in 2022).  For preexisting sites the change signal is
intentionally bypassed — see ConstructionChangeDetector.forward().
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ChangeScorer(nn.Module):
    """Temporal change scorer based on SAM2 feature deltas.

    Args:
        feature_dim: Dimension of SAM2 FPN features (256 for Hiera-L).
    """

    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()

        # The input is the concatenation of [feat_2024 - feat_2022] and
        # [|feat_2024 - feat_2022|] — both delta and magnitude — giving
        # the head richer change signal without adding many parameters.
        in_channels = feature_dim * 2

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, feats_2022: Tensor, feats_2024: Tensor) -> Tensor:
        """
        Args:
            feats_2022: [B, C, H_f, W_f] — SAM2 FPN features of the 2022 tile.
            feats_2024: [B, C, H_f, W_f] — SAM2 FPN features of the 2024 tile.

        Returns:
            Change probability map [B, 1, H_f, W_f] in (0, 1).
        """
        delta = feats_2024 - feats_2022          # signed change
        mag   = torch.abs(delta)                 # change magnitude
        x     = torch.cat([delta, mag], dim=1)   # [B, 2C, H_f, W_f]
        return torch.sigmoid(self.head(x))
