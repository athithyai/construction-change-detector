"""EpisodicTrainer — trains AppearanceScorer and ChangeScorer jointly.

The training loop:
  • Loads prebuilt corpus prototypes from ``checkpoints/prototypes.pt``.
  • Feeds (tile_2022, tile_2024, mask_2022, mask_2024) batches to the detector.
  • Computes combined BCE + Dice loss on ``prob_map`` vs ``mask_2024``.
  • Validates every epoch; saves best checkpoint by validation IoU.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import ConstructionChangeDataset
from evaluation.metrics import SegmentationEvaluator
from losses.segmentation_losses import CombinedSegmentationLoss
from models.detector import ConstructionChangeDetector

logger = logging.getLogger(__name__)


class EpisodicTrainer:

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s", self.device)

        # --- Model ---
        self.model = ConstructionChangeDetector(
            k_prototypes=cfg.model.k_prototypes
        ).to(self.device)
        self.model.load_sam2(cfg.model.sam2_hf_id)

        # --- Prototypes ---
        proto_path = Path(cfg.training.checkpoint_dir) / "prototypes.pt"
        if not proto_path.exists():
            raise FileNotFoundError(
                f"Prototypes not found at {proto_path}. "
                "Run scripts/build_prototypes.py first."
            )
        ckpt = torch.load(proto_path, map_location="cpu")
        self.prototypes: torch.Tensor = ckpt["prototypes"].to(self.device)
        logger.info("Loaded %d prototypes.", self.prototypes.shape[0])

        # --- Loss ---
        self.criterion = CombinedSegmentationLoss(
            bce_weight=cfg.training.loss_bce_weight,
            dice_weight=cfg.training.loss_dice_weight,
        )

        # --- Optimizer (only trainable heads) ---
        trainable = list(self.model.appearance_scorer.parameters()) + \
                    list(self.model.change_scorer.parameters())
        self.optimizer = AdamW(
            trainable,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

        # --- LR schedule: linear warmup → cosine annealing ---
        warmup_steps = cfg.training.warmup_epochs
        total_steps  = cfg.training.epochs
        warmup_sched = LinearLR(
            self.optimizer, start_factor=0.1, total_iters=warmup_steps
        )
        cosine_sched = CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_steps],
        )

        # --- AMP ---
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp)

        # --- Data ---
        self.train_loader, self.val_loader = self._build_dataloaders()

        # --- Logging ---
        log_dir = Path(cfg.training.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer    = SummaryWriter(str(log_dir))
        self.ckpt_dir  = Path(cfg.training.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_iou  = 0.0
        self.log_every = cfg.training.log_every

    # ------------------------------------------------------------------

    def _build_dataloaders(self):
        common = dict(
            tile_dir=self.cfg.data.tile_dir,
            image_size=self.cfg.data.image_size,
            min_fg_pixels=self.cfg.data.min_fg_pixels,
            val_split=self.cfg.training.val_split,
            seed=self.cfg.training.seed,
        )
        train_ds = ConstructionChangeDataset(augment=True,  split="train", **common)
        val_ds   = ConstructionChangeDataset(augment=False, split="val",   **common)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader

    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        # Keep encoder in eval mode regardless
        if self.model.image_encoder is not None:
            self.model.image_encoder.eval()

        total_loss = 0.0
        global_step = epoch * len(self.train_loader)

        for step, batch in enumerate(self.train_loader):
            tile_2022 = batch["tile_2022"].to(self.device)
            tile_2024 = batch["tile_2024"].to(self.device)
            mask_2022 = batch["mask_2022"].to(self.device)
            mask_2024 = batch["mask_2024"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.cfg.training.amp):
                out       = self.model(tile_2022, tile_2024, mask_2022, self.prototypes)
                loss_dict = self.criterion(out["prob_map"], mask_2024)

            self.scaler.scale(loss_dict["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                list(self.model.appearance_scorer.parameters()) +
                list(self.model.change_scorer.parameters()),
                max_norm=1.0,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss_dict["loss"].item()

            if step % self.log_every == 0:
                gstep = global_step + step
                self.writer.add_scalar("train/loss",      loss_dict["loss"].item(),      gstep)
                self.writer.add_scalar("train/bce_loss",  loss_dict["bce_loss"].item(),  gstep)
                self.writer.add_scalar("train/dice_loss", loss_dict["dice_loss"].item(), gstep)

        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        self.model.eval()
        evaluator = SegmentationEvaluator(threshold=0.5)

        for batch in self.val_loader:
            tile_2022 = batch["tile_2022"].to(self.device)
            tile_2024 = batch["tile_2024"].to(self.device)
            mask_2022 = batch["mask_2022"].to(self.device)
            mask_2024 = batch["mask_2024"].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.cfg.training.amp):
                out = self.model(tile_2022, tile_2024, mask_2022, self.prototypes)

            evaluator.update(out["prob_map"].cpu(), mask_2024.cpu())

        metrics = evaluator.compute()
        for k, v in metrics.items():
            self.writer.add_scalar(f"val/{k}", v, epoch)
        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        state = {
            "epoch":   epoch,
            "best_iou": self.best_iou,
            **self.model.state_dict_trainable(),
            "optimizer": self.optimizer.state_dict(),
        }
        name = "best.pt" if is_best else f"epoch_{epoch:03d}.pt"
        torch.save(state, self.ckpt_dir / name)

    def load_checkpoint(self, path: str) -> int:
        state = torch.load(path, map_location=self.device)
        self.model.load_trainable_state_dict(state)
        self.best_iou = state.get("best_iou", 0.0)
        return state.get("epoch", 0)

    # ------------------------------------------------------------------

    def train(self) -> None:
        logger.info("Starting training for %d epochs …", self.cfg.training.epochs)
        for epoch in range(self.cfg.training.epochs):
            train_loss = self.train_epoch(epoch)
            metrics    = self.validate(epoch)
            self.scheduler.step()

            logger.info(
                "Epoch %03d | loss=%.4f | iou=%.4f | f1=%.4f | "
                "precision=%.4f | recall=%.4f",
                epoch, train_loss,
                metrics["iou"], metrics["f1"],
                metrics["precision"], metrics["recall"],
            )

            if metrics["iou"] > self.best_iou:
                self.best_iou = metrics["iou"]
                self.save_checkpoint(epoch, is_best=True)
                logger.info("  ↑ New best IoU=%.4f — saved best.pt", self.best_iou)

            if (epoch + 1) % self.cfg.training.checkpoint_every == 0:
                self.save_checkpoint(epoch, is_best=False)

        self.writer.close()
        logger.info("Training complete. Best IoU=%.4f", self.best_iou)
