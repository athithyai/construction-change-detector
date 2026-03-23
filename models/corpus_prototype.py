"""Build a global prototype corpus from all 2022 construction site tiles.

Algorithm:
1. For each (tile, mask) pair in the 2022 dataset, pass the tile through the
   frozen SAM2 image encoder → feature map [256, H_f, W_f].
2. Downsample the mask to feature resolution and extract foreground feature
   vectors → shape [N_fg, 256].
3. Accumulate all foreground vectors across all sites.
4. Run scikit-learn KMeans(k) on the accumulated vectors.
5. Save the k cluster centroids as a torch tensor → ``prototypes.pt``.

The resulting k=20 prototypes represent the full visual diversity of
construction sites in the 2022 dataset (bare soil, scaffolding, concrete,
cranes, etc.) without needing any 2024 labels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm

logger = logging.getLogger(__name__)


def build_corpus_prototypes(
    image_encoder: torch.nn.Module,
    dataset,
    k: int = 20,
    max_fg_per_site: int = 2000,
    device: torch.device = torch.device("cpu"),
    output_path: Optional[str] = None,
) -> torch.Tensor:
    """Extract per-site foreground features, cluster into k prototypes.

    Args:
        image_encoder:  Frozen SAM2 image encoder (Hiera + FPN neck).
        dataset:        ConstructionChangeDataset (or any iterable that yields
                        dicts with "tile_2022" and "mask_2022" tensors).
        k:              Number of prototype clusters.
        max_fg_per_site: Maximum foreground feature vectors sampled per site
                        (random sub-sample) to keep memory bounded.
        device:         Torch device for encoder inference.
        output_path:    If provided, save prototypes to this path.

    Returns:
        Prototype tensor of shape [k, feature_dim].
    """
    image_encoder.eval()
    image_encoder.to(device)

    all_features: list[np.ndarray] = []

    with torch.no_grad():
        for sample in tqdm(dataset, desc="Extracting prototype features"):
            tile  = sample["tile_2022"]                    # [3, H, W]
            mask  = sample["mask_2022"]                    # [1, H, W] in {0,1}

            tile = tile.unsqueeze(0).to(device)            # [1, 3, H, W]

            enc_out   = image_encoder(tile)
            feat_map  = enc_out["vision_features"]         # [1, C, H_f, W_f]
            C, Hf, Wf = feat_map.shape[1:]

            # Resize mask to feature resolution
            mask_f = F.interpolate(
                mask.unsqueeze(0).float().to(device),
                size=(Hf, Wf),
                mode="nearest",
            ).squeeze(0).squeeze(0)                        # [H_f, W_f]

            fg_mask = mask_f > 0.5                         # bool
            if fg_mask.sum() < 1:
                continue

            # [C, H_f, W_f] → [N_fg, C]
            feats = feat_map.squeeze(0).permute(1, 2, 0)  # [H_f, W_f, C]
            fg_feats = feats[fg_mask].cpu().numpy()        # [N_fg, C]

            # Sub-sample to cap memory usage
            if len(fg_feats) > max_fg_per_site:
                idx = np.random.choice(len(fg_feats), max_fg_per_site, replace=False)
                fg_feats = fg_feats[idx]

            all_features.append(fg_feats)

    if not all_features:
        raise RuntimeError("No foreground features found — check mask paths.")

    X = np.concatenate(all_features, axis=0)
    logger.info("Running KMeans(k=%d) on %d feature vectors …", k, len(X))

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X)

    centroids = torch.from_numpy(km.cluster_centers_.astype(np.float32))  # [k, C]

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"prototypes": centroids, "k": k, "inertia": float(km.inertia_)},
            output_path,
        )
        logger.info("Prototypes saved → %s", output_path)

    return centroids
