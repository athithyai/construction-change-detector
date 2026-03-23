"""Utilities for tiling large aerial images and stitching probability maps.

Also contains the GeoTIFF writer that preserves CRS/transform metadata so
interpreters can load output rasters directly in QGIS / ArcGIS.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_bounds


# ---------------------------------------------------------------------------
# Image tiling
# ---------------------------------------------------------------------------

def tile_image_for_inference(
    image: np.ndarray,
    tile_size: int = 1024,
    overlap: int = 128,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """Split a large HxWxC image into overlapping tiles.

    Returns:
        tiles:  list of (tile_size x tile_size x C) uint8 arrays.
        coords: list of (x0, y0, x1, y1) pixel coordinates in the source image.
    """
    h, w = image.shape[:2]
    stride = tile_size - overlap

    tiles: List[np.ndarray]              = []
    coords: List[Tuple[int, int, int, int]] = []

    y = 0
    while y < h:
        x = 0
        while x < w:
            x1 = min(x + tile_size, w)
            y1 = min(y + tile_size, h)
            x0 = max(0, x1 - tile_size)
            y0 = max(0, y1 - tile_size)

            tile = image[y0:y1, x0:x1]

            # Pad to tile_size if the image edge is smaller
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                pad_h = tile_size - tile.shape[0]
                pad_w = tile_size - tile.shape[1]
                if image.ndim == 3:
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)))
                else:
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w)))

            tiles.append(tile)
            coords.append((x0, y0, x1, y1))
            x += stride
        y += stride

    return tiles, coords


def stitch_tiles(
    tile_probs: List[np.ndarray],
    coords: List[Tuple[int, int, int, int]],
    original_shape: Tuple[int, int],
) -> np.ndarray:
    """Reconstruct a full probability map from tiled predictions.

    Overlap regions are resolved by Gaussian-weighted blending, which avoids
    hard seam artefacts.

    Args:
        tile_probs:     List of HxW float32 probability arrays (values in [0,1]).
        coords:         Matching list of (x0, y0, x1, y1) pixel coords.
        original_shape: (H, W) of the full image.

    Returns:
        HxW float32 probability map.
    """
    H, W = original_shape
    accum  = np.zeros((H, W), dtype=np.float64)
    weight = np.zeros((H, W), dtype=np.float64)

    for prob, (x0, y0, x1, y1) in zip(tile_probs, coords):
        h, w = y1 - y0, x1 - x0
        prob_crop = prob[:h, :w]

        # 2-D Gaussian window centred on the tile
        gw = _gaussian_window(h, w)

        accum [y0:y1, x0:x1] += prob_crop * gw
        weight[y0:y1, x0:x1] += gw

    weight = np.where(weight > 0, weight, 1.0)
    return (accum / weight).astype(np.float32)


# ---------------------------------------------------------------------------
# GeoTIFF I/O
# ---------------------------------------------------------------------------

def load_geotiff_rgb(path: str) -> Tuple[np.ndarray, rasterio.DatasetReader]:
    """Load a GeoTIFF as HxWx3 uint8 array and return it with its open dataset
    (for CRS / transform metadata).  Caller should close the dataset when done."""
    src = rasterio.open(path)
    n = min(src.count, 3)
    bands = src.read(list(range(1, n + 1)))   # [C, H, W]
    img   = np.transpose(bands, (1, 2, 0))     # [H, W, C]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img, src


def save_probability_geotiff(
    prob_bands: List[np.ndarray],
    band_names: List[str],
    reference_path: str,
    output_path: str,
) -> None:
    """Save one or more probability bands as a multi-band GeoTIFF.

    CRS and affine transform are copied from *reference_path* so the output is
    immediately loadable in QGIS / ArcGIS at the correct location.

    Args:
        prob_bands:     List of HxW float32 arrays, each in [0, 1].
        band_names:     Human-readable label for each band (written as metadata).
        reference_path: Source GeoTIFF from which CRS/transform are copied.
        output_path:    Destination file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_path) as ref:
        crs       = ref.crs
        transform = ref.transform
        height, width = prob_bands[0].shape

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=len(prob_bands),
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-1.0,
        compress="lzw",
    ) as dst:
        for i, (band, name) in enumerate(zip(prob_bands, band_names), start=1):
            dst.write(band.astype(np.float32), i)
            dst.update_tags(i, name=name)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gaussian_window(h: int, w: int, sigma_ratio: float = 0.3) -> np.ndarray:
    """Return a 2-D Gaussian weight matrix of shape (h, w)."""
    sigma_y = h * sigma_ratio
    sigma_x = w * sigma_ratio
    cy, cx  = h / 2.0, w / 2.0
    ys = np.arange(h)
    xs = np.arange(w)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.exp(
        -0.5 * ((yy - cy) ** 2 / sigma_y ** 2 + (xx - cx) ** 2 / sigma_x ** 2)
    )
