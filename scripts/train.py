"""Train the ConstructionChangeDetector.

Run build_prototypes.py first — training requires checkpoints/prototypes.pt.

Usage::

    python scripts/train.py --config configs/train.yaml

Override any config key on the CLI::

    python scripts/train.py training.batch_size=4 training.epochs=100
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from omegaconf import OmegaConf

from training.trainer import EpisodicTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("overrides", nargs="*", help="key=value config overrides")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    for override in args.overrides:
        key, _, value = override.partition("=")
        OmegaConf.update(cfg, key, value, merge=True)

    # Reproducibility
    seed = cfg.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    trainer = EpisodicTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
