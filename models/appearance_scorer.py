"""AppearanceScorer — scores how much a query feature map looks like any of the
K corpus prototypes.

Forward pass:
  1. L2-normalise query features and prototypes.
  2. Compute cosine similarity between every query spatial position and every
     prototype → correlation volume [B, K, H_f, W_f].
  3. Fuse K channels into a single probability map via a tiny ConvNet.
  4. Sigmoid → [B, 1, H_f, W_f] in (0, 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AppearanceScorer(nn.Module):
    """Dense prototype matching scorer.

    Args:
        feature_dim:    Dimension of SAM2 FPN features (256 for Hiera-L).
        num_prototypes: Number of corpus prototypes (K).
    """

    def __init__(self, feature_dim: int = 256, num_prototypes: int = 20) -> None:
        super().__init__()
        self.feature_dim    = feature_dim
        self.num_prototypes = num_prototypes

        # Lightweight fusion: [B, K, H, W] → [B, 1, H, W]
        self.fusion = nn.Sequential(
            nn.Conv2d(num_prototypes, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, query_features: Tensor, prototypes: Tensor) -> Tensor:
        """
        Args:
            query_features: [B, C, H_f, W_f] — SAM2 FPN features of the 2024 tile.
            prototypes:     [K, C]            — corpus prototypes (prebuilt offline).

        Returns:
            Appearance probability map [B, 1, H_f, W_f] in (0, 1).
        """
        B, C, Hf, Wf = query_features.shape
        K = prototypes.shape[0]

        # L2-normalise
        q = F.normalize(query_features, dim=1)          # [B, C, H_f, W_f]
        p = F.normalize(prototypes, dim=1)               # [K, C]

        # Reshape for batched matmul: [B, H_f*W_f, C] × [C, K] → [B, H_f*W_f, K]
        q_flat = q.permute(0, 2, 3, 1).reshape(B, Hf * Wf, C)
        sim    = torch.matmul(q_flat, p.t())             # [B, H_f*W_f, K]
        sim    = sim.reshape(B, Hf, Wf, K).permute(0, 3, 1, 2)  # [B, K, H_f, W_f]

        # Fuse and sigmoid
        logits = self.fusion(sim)                        # [B, 1, H_f, W_f]
        return torch.sigmoid(logits)
