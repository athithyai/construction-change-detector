"""Evaluate a trained checkpoint on the validation or test split.

Usage::

    python scripts/evaluate.py \\
        --checkpoint checkpoints/best.pt \\
        --prototypes checkpoints/prototypes.pt \\
        --split val
"""

from __future__ import annotations

import argparse
import logging

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.dataset import ConstructionChangeDataset
from evaluation.metrics import SegmentationEvaluator
from models.detector import ConstructionChangeDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate construction change detector.")
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--prototypes",  required=True)
    p.add_argument("--config",      default="configs/base.yaml")
    p.add_argument("--split",       default="val", choices=["train", "val"])
    p.add_argument("--threshold",   type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ConstructionChangeDetector(k_prototypes=cfg.model.k_prototypes)
    model.load_sam2(cfg.model.sam2_hf_id)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_trainable_state_dict(state)
    model.to(device).eval()

    # Load prototypes
    proto_ckpt = torch.load(args.prototypes, map_location=device)
    prototypes = proto_ckpt["prototypes"].to(device)

    # Dataset
    dataset = ConstructionChangeDataset(
        tile_dir=cfg.data.tile_dir,
        image_size=cfg.data.image_size,
        augment=False,
        min_fg_pixels=cfg.data.min_fg_pixels,
        split=args.split,
        val_split=cfg.training.val_split,
        seed=cfg.training.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    evaluator = SegmentationEvaluator(threshold=args.threshold)

    with torch.no_grad():
        for batch in loader:
            tile_2022 = batch["tile_2022"].to(device)
            tile_2024 = batch["tile_2024"].to(device)
            mask_2022 = batch["mask_2022"].to(device)
            mask_2024 = batch["mask_2024"].to(device)

            out = model(tile_2022, tile_2024, mask_2022, prototypes)
            evaluator.update(out["prob_map"].cpu(), mask_2024.cpu())

    metrics = evaluator.compute()
    logger.info("=== Evaluation results (%s split) ===", args.split)
    for k, v in metrics.items():
        logger.info("  %-12s %.4f", k, v)


if __name__ == "__main__":
    main()
