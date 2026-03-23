"""Albumentations pipelines for aerial imagery.

Both tile_2022 and tile_2024 share the same geometric transforms (so spatial
alignment is preserved), but each can receive independent colour jitter via
the additional_targets mechanism.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# additional_targets: tile_2022 is treated as a second "image",
# mask_2022 is treated as a second "mask".
_EXTRA = {"image2": "image", "mask2": "mask"}


def get_train_transforms(image_size: int = 1024) -> A.Compose:
    """Augmentation pipeline used during training."""
    return A.Compose(
        [
            A.RandomResizedCrop(
                height=image_size, width=image_size,
                scale=(0.5, 1.0), ratio=(0.9, 1.1),
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.2, hue=0.05, p=0.5,
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ],
        additional_targets=_EXTRA,
    )


def get_inference_transforms(image_size: int = 1024) -> A.Compose:
    """Deterministic pipeline used during inference / validation."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ],
        additional_targets=_EXTRA,
    )
