"""SamSpade Dashboard — FastAPI backend.

Workflow
--------
Draw bbox → fetch 2022 + 2024 RGB + CIR imagery → compute NDVI + depth for
both years → dense SAM2 auto-segment on 2024 RGB → per-segment CLIP + NDVI +
depth roughness → terrain-label + construction-score.

Single endpoint: POST /api/analyze-bbox
"""
from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import io
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from pyproj import Transformer

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))

from data.pdok_downloader import PDOKDownloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Global model state ────────────────────────────────────────────────────────
SAM2_MODEL      = None
SAM2_MASK_GEN   = None
CLIP_MODEL      = None
CLIP_PROCESSOR  = None
CLIP_TEXT_FEATS: Optional[torch.Tensor] = None
DEPTH_MODEL     = None
DEPTH_PROCESSOR = None
DEVICE: Optional[torch.device] = None
MODEL_READY     = False
LAST_GPKG: Optional[Path] = None

CIR_WMS_URL = "https://service.pdok.nl/hwh/luchtfotocir/wms/v1_0"

# ── CLIP labels ───────────────────────────────────────────────────────────────
CLIP_LABELS = [
    "construction site with cranes and machinery",       #  0
    "building under construction concrete frame",        #  1
    "excavation pit earthwork foundation",               #  2
    "scaffolding on building facade",                    #  3
    "demolition site rubble and debris",                 #  4
    "concrete pour steel structure being built",         #  5
    "road asphalt pavement",                             #  6
    "parking lot with vehicles",                         #  7
    "residential house rooftops",                        #  8
    "office or commercial building roof",                #  9
    "industrial warehouse factory roof",                 # 10
    "green vegetation trees park grass",                 # 11
    "water canal river pond",                            # 12
    "bare soil gravel sand field",                       # 13
]

# Label-index groups for score fusion
_CONSTR_IDXS  = [0, 1, 2, 3, 4, 5, 13]   # construction + bare soil
_VEGE_IDXS    = [11]
_WATER_IDXS   = [12]
_BUILD_IDXS   = [8, 9, 10]
_PAVED_IDXS   = [6, 7]

# ── Terrain output classes ────────────────────────────────────────────────────
TERRAIN_COLORS = {
    "likely construction terrain": "#FF4500",   # vivid orange-red
    "exposed soil / bare ground":  "#CD853F",   # peru brown
    "vegetation":                  "#22BB44",   # medium green
    "water":                       "#1E90FF",   # dodger blue
    "roof / building":             "#9370DB",   # medium purple
    "paved surface":               "#778899",   # slate gray
    "shadow / unknown":            "#505060",   # dark blue-gray
}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="SamSpade — Construction Terrain Analysis")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

STATIC = ROOT / "static"
STATIC.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# ── Model loading ─────────────────────────────────────────────────────────────
def _load_model() -> None:
    global SAM2_MODEL, SAM2_MASK_GEN, CLIP_MODEL, CLIP_PROCESSOR, CLIP_TEXT_FEATS
    global DEPTH_MODEL, DEPTH_PROCESSOR, DEVICE, MODEL_READY
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        DEVICE = device

        logger.info("Loading SAM2 Hiera-L on %s …", device)
        from sam2.build_sam import build_sam2_hf
        SAM2_MODEL = build_sam2_hf("facebook/sam2.1-hiera-large", device=device)
        SAM2_MODEL.eval()

        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        SAM2_MASK_GEN = SAM2AutomaticMaskGenerator(
            model=SAM2_MODEL,
            points_per_side=32,           # dense grid → 1024 prompt points
            pred_iou_thresh=0.65,         # lowered for better coverage
            stability_score_thresh=0.75,  # lowered for fewer gaps
            min_mask_region_area=80,
            crop_n_layers=1,              # also segment at 2× crop scale
            crop_n_points_downscale_factor=2,
        )
        logger.info("SAM2AutomaticMaskGenerator ready (32 pts/side).")

        logger.info("Loading CLIP ViT-L/14 …")
        from transformers import CLIPProcessor, CLIPModel
        CLIP_MODEL     = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        CLIP_MODEL.eval()
        with torch.no_grad():
            txt_in  = CLIP_PROCESSOR(text=CLIP_LABELS, return_tensors="pt", padding=True).to(device)
            txt_out = CLIP_MODEL.text_model(
                input_ids=txt_in["input_ids"], attention_mask=txt_in["attention_mask"]
            )
            CLIP_TEXT_FEATS = F.normalize(
                CLIP_MODEL.text_projection(txt_out.pooler_output), dim=-1
            )
        logger.info("CLIP ready — %d label embeddings.", len(CLIP_LABELS))

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            logger.info("Loading Depth Anything V2 Small …")
            DEPTH_PROCESSOR = AutoImageProcessor.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf"
            )
            DEPTH_MODEL = AutoModelForDepthEstimation.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf"
            ).to(device)
            DEPTH_MODEL.eval()
            logger.info("Depth Anything V2 ready.")
        except Exception as exc:
            logger.warning("Depth Anything V2 not loaded: %s", exc)

        MODEL_READY = True
        logger.info("All models ready ✓  device=%s", device)
    except Exception as exc:
        logger.error("Model load failed: %s", exc)
        raise


@app.on_event("startup")
async def startup() -> None:
    loop = asyncio.get_event_loop()
    loop.run_in_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1), _load_model)


# ── Request schema ────────────────────────────────────────────────────────────
class AnalyzeBboxRequest(BaseModel):
    bbox_wgs84: list[float]   # [west, south, east, north]


# ── Geo helpers ───────────────────────────────────────────────────────────────
def _bbox_rd_from_wgs84(bbox_wgs84: list) -> tuple:
    t = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    w, s = t.transform(bbox_wgs84[0], bbox_wgs84[1])
    e, n = t.transform(bbox_wgs84[2], bbox_wgs84[3])
    return (w, s, e, n)


def _to_wgs84_bounds(bbox_rd: tuple) -> tuple:
    t2 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    west,  south = t2.transform(bbox_rd[0], bbox_rd[1])
    east,  north = t2.transform(bbox_rd[2], bbox_rd[3])
    return west, south, east, north


def _enforce_min_extent(bbox_rd: tuple, min_half: float = 300.0) -> tuple:
    """Expand to at least 600 × 600 m so PDOK doesn't return blank tiles."""
    xmin, ymin, xmax, ymax = bbox_rd
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half   = max((xmax - xmin) / 2, (ymax - ymin) / 2, min_half)
    return (cx - half, cy - half, cx + half, cy + half)


def _fetch_tile(dl_8cm, dl_25cm, bbox_rd: tuple, year: int, path: str):
    r = dl_8cm.download_tile(bbox_rd, year, path)
    if r is not None:
        return r
    logger.warning("8 cm blank for year=%d — trying 25 cm…", year)
    path25 = path.replace(".tif", "_25cm.tif")
    r = dl_25cm.download_tile(bbox_rd, year, path25)
    if r is not None:
        import shutil
        shutil.move(path25, path)
        return r
    return None


def _load_geotiff(path: str):
    """Return (H×W×3 uint8, crs, transform)."""
    import rasterio
    with rasterio.open(path) as src:
        img = src.read([1, 2, 3]).transpose(1, 2, 0)
        return img.copy(), src.crs, src.transform


# ── Image utilities ───────────────────────────────────────────────────────────
def _img_to_b64(img_np: np.ndarray, max_side: int = 768) -> str:
    """Resize (keep aspect) and encode H×W×3 uint8 → data:image/png;base64,…"""
    import cv2
    H, W  = img_np.shape[:2]
    scale = min(1.0, max_side / max(H, W, 1))
    if scale < 1.0:
        img_np = cv2.resize(img_np, (int(W * scale), int(H * scale)),
                            interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── CIR / NDVI ────────────────────────────────────────────────────────────────
def _fetch_cir_tile(bbox_rd: tuple, year: int) -> Optional[np.ndarray]:
    import requests as _req
    params = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS": f"{year}_ortho25", "STYLES": "",
        "CRS": "EPSG:28992",
        "BBOX": f"{bbox_rd[0]},{bbox_rd[1]},{bbox_rd[2]},{bbox_rd[3]}",
        "WIDTH": "1024", "HEIGHT": "1024",
        "FORMAT": "image/png",
    }
    try:
        r  = _req.get(CIR_WMS_URL, params=params, timeout=15)
        ct = r.headers.get("Content-Type", "")
        if "xml" in ct or not r.content:
            return None
        arr = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
        if (arr == 255).all(axis=-1).mean() > 0.95:
            return None
        return arr
    except Exception as exc:
        logger.warning("CIR fetch year=%d failed: %s", year, exc)
        return None


def _compute_ndvi(cir_np: np.ndarray, target_hw: tuple) -> np.ndarray:
    """CIR (R=NIR, G=Red) → NDVI ∈ [−1, 1] at target_hw resolution."""
    import cv2
    nir  = cir_np[:, :, 0].astype(np.float32)
    red  = cir_np[:, :, 1].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)
    H, W = target_hw
    return cv2.resize(ndvi, (W, H), interpolation=cv2.INTER_LINEAR)


def _ndvi_to_overlay(ndvi: np.ndarray) -> str:
    """NDVI → RGBA PNG with RdYlGn colormap: red = bare/construction, green = vegetation."""
    from matplotlib import cm
    norm = np.clip((ndvi + 1.0) / 2.0, 0.0, 1.0)     # [−1,1] → [0,1]
    rgba = (cm.get_cmap("RdYlGn")(norm) * 255).astype(np.uint8)
    rgba[:, :, 3] = 210
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _nir_to_overlay(cir_np: np.ndarray, target_hw: tuple) -> str:
    """Extract NIR band (channel 0 of PDOK CIR) → grayscale RGBA PNG at target_hw.

    High NIR = bright (dense vegetation).  Low NIR = dark (bare soil / construction).
    """
    import cv2
    from matplotlib import cm
    nir  = cir_np[:, :, 0].astype(np.float32)
    H, W = target_hw
    if nir.shape != (H, W):
        nir = cv2.resize(nir, (W, H), interpolation=cv2.INTER_LINEAR)
    mn, mx = nir.min(), nir.max()
    norm   = (nir - mn) / (mx - mn + 1e-6)
    rgba   = (cm.get_cmap("YlGn")(norm) * 255).astype(np.uint8)
    rgba[:, :, 3] = 210
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── NDBI ──────────────────────────────────────────────────────────────────────
def _compute_ndbi(cir_np: np.ndarray, target_hw: tuple) -> np.ndarray:
    """Proxy NDBI = (Red − NIR) / (Red + NIR + ε) from PDOK CIR channels.

    High NDBI (near +1) → bare soil / built-up surface.
    Low NDBI  (near −1) → dense vegetation (inverse of NDVI).
    CIR layout: channel 0 = NIR, channel 1 = Red.
    """
    import cv2
    nir  = cir_np[:, :, 0].astype(np.float32)
    red  = cir_np[:, :, 1].astype(np.float32)
    ndbi = (red - nir) / (red + nir + 1e-6)
    H, W = target_hw
    return cv2.resize(ndbi, (W, H), interpolation=cv2.INTER_LINEAR)


def _ndbi_to_overlay(ndbi: np.ndarray) -> str:
    """NDBI [−1,1] → RGBA PNG with RdBu_r colormap (red = built-up, blue = veg)."""
    from matplotlib import cm
    norm = np.clip((ndbi + 1.0) / 2.0, 0.0, 1.0)
    rgba = (cm.get_cmap("RdBu_r")(norm) * 255).astype(np.uint8)
    rgba[:, :, 3] = 210
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── Depth ─────────────────────────────────────────────────────────────────────
def _compute_depth(img_np: np.ndarray) -> Optional[np.ndarray]:
    """Depth Anything V2 Small → H×W float32 depth map normalised to [0, 1]."""
    if DEPTH_MODEL is None:
        return None
    import cv2
    H, W = img_np.shape[:2]
    try:
        with torch.no_grad():
            inp   = DEPTH_PROCESSOR(images=Image.fromarray(img_np),
                                    return_tensors="pt").to(DEVICE)
            depth = DEPTH_MODEL(**inp).predicted_depth.squeeze().cpu().numpy().astype(np.float32)
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        mn, mx = depth.min(), depth.max()
        return (depth - mn) / (mx - mn + 1e-6)
    except Exception as exc:
        logger.warning("Depth inference failed: %s", exc)
        return None


def _depth_to_overlay(depth: np.ndarray) -> str:
    """Depth [0,1] → RGBA PNG with inferno colormap (dark=near, bright=far)."""
    from matplotlib import cm
    rgba = (cm.get_cmap("inferno")(depth) * 255).astype(np.uint8)
    rgba[:, :, 3] = 210
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── CLIP per-segment classification ──────────────────────────────────────────
def _clip_classify_masks(img_np: np.ndarray, masks_data: list) -> list:
    """Batch-CLIP-classify each SAM2 mask crop. Returns list of dicts."""
    n_labels = len(CLIP_LABELS)
    empty    = {"top_idx": [0], "top_scores": [0.0], "all_sims": [0.0] * n_labels}
    if CLIP_MODEL is None or not masks_data:
        return [empty] * len(masks_data)

    pad = 8
    H, W = img_np.shape[:2]
    crops = []
    for m in masks_data:
        x, y, bw, bh = [int(v) for v in m["bbox"]]
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(W, x + bw + pad), min(H, y + bh + pad)
        crop = img_np[y1:y2, x1:x2].copy()
        crop[~m["segmentation"][y1:y2, x1:x2]] = 128   # gray outside mask
        crops.append(Image.fromarray(crop))

    results, bs = [], 32
    with torch.no_grad():
        for i in range(0, len(crops), bs):
            batch = crops[i: i + bs]
            inp   = CLIP_PROCESSOR(images=batch, return_tensors="pt",
                                   padding=True).to(DEVICE)
            v_out = CLIP_MODEL.vision_model(pixel_values=inp["pixel_values"])
            feats = F.normalize(CLIP_MODEL.visual_projection(v_out.pooler_output), dim=-1)
            sims  = (feats @ CLIP_TEXT_FEATS.T).cpu()
            for j in range(len(batch)):
                top = sims[j].topk(3)
                results.append({
                    "top_idx":    top.indices.tolist(),
                    "top_scores": top.values.tolist(),
                    "all_sims":   sims[j].tolist(),     # full similarity vector
                })
    return results


# ── Construction score ────────────────────────────────────────────────────────
def _construction_score(
    all_sims: list,
    ndvi_24: Optional[float],
    ndvi_22: Optional[float],
    depth_var_24: Optional[float],
    depth_var_22: Optional[float] = None,
) -> float:
    """Fuse multi-signal → construction probability [0, 1].

    Signals
    -------
    * ndvi_abs:   Low NDVI in 2024 → bare/exposed (construction-like)
    * ndvi_delta: NDVI decreased 2022→2024 → vegetation loss (new construction)
    * roughness:  High depth variance → disturbed terrain.
                  Suppressed when NDVI is high — tree canopy is also rough!
    * clip_s:     CLIP semantic similarity to construction labels
    """
    # 1. Absolute NDVI — bare soil in 2024
    if ndvi_24 is not None:
        ndvi_abs = float(np.clip((0.20 - ndvi_24) / 0.50, 0.0, 1.0))
    else:
        ndvi_abs = 0.25

    # 2. Temporal NDVI delta — vegetation LOSS from 2022 → 2024
    if ndvi_22 is not None and ndvi_24 is not None:
        delta = ndvi_22 - ndvi_24           # positive = greening lost
        ndvi_delta = float(np.clip(delta / 0.30, 0.0, 1.0))
    else:
        ndvi_delta = 0.0

    # 3. Depth roughness — suppressed for vegetated areas (tree canopy is rough)
    if depth_var_24 is not None:
        roughness_raw = float(np.clip((depth_var_24 / 1000.0) / 0.025, 0.0, 1.0))
        # Penalise roughness score when NDVI is clearly high (vegetation > 0.25)
        veg_supp = max(0.0, (ndvi_24 if ndvi_24 is not None else 0.0) - 0.25) * 3.0
        roughness = roughness_raw * max(0.0, 1.0 - veg_supp)
    else:
        roughness = 0.2

    # 4. CLIP semantic similarity
    if all_sims:
        raw    = max(all_sims[i] for i in _CONSTR_IDXS if i < len(all_sims))
        clip_s = float(np.clip((raw - 0.08) / 0.22, 0.0, 1.0))
    else:
        clip_s = 0.0

    return round(0.25 * ndvi_abs + 0.30 * ndvi_delta + 0.20 * roughness + 0.25 * clip_s, 3)


# ── Terrain label assignment ──────────────────────────────────────────────────
def _assign_terrain_label(
    all_sims: list,
    construction_score: float,
    ndvi_24: Optional[float],
    ndvi_22: Optional[float],
) -> str:
    """Map fused scores → one of the 7 terrain classes.

    Hard vegetation gate: if area is clearly green (high NDVI in 2024, or
    green in BOTH years), it is labelled vegetation regardless of construction
    score — prevents forests/parks from being mis-classified as construction.
    """
    # ── Hard vegetation gate ──────────────────────────────────────────────────
    if ndvi_24 is not None and ndvi_24 > 0.35:
        return "vegetation"
    if (ndvi_22 is not None and ndvi_22 > 0.30) and (ndvi_24 is not None and ndvi_24 > 0.22):
        return "vegetation"

    # ── Strong construction signal ────────────────────────────────────────────
    if construction_score >= 0.60:
        return "likely construction terrain"

    if all_sims and len(all_sims) >= 14:
        water_s = all_sims[12]
        veg_s   = all_sims[11]
        build_s = max(all_sims[i] for i in _BUILD_IDXS)
        paved_s = max(all_sims[i] for i in _PAVED_IDXS)
        bare_s  = all_sims[13]

        if water_s > 0.22:
            return "water"
        if build_s > 0.22:
            return "roof / building"
        if paved_s > 0.22:
            return "paved surface"
        if veg_s > 0.22 or (ndvi_24 is not None and ndvi_24 > 0.22):
            return "vegetation"
        if bare_s > 0.20 or construction_score >= 0.35:
            return "exposed soil / bare ground"

    return "shadow / unknown"


# ── Vectorization ─────────────────────────────────────────────────────────────
def _masks_to_gdf(masks_sorted: list, img_shape: tuple, tile_crs, tile_transform):
    """SAM2 masks (sorted descending by area) → GeoDataFrame in WGS84.

    Large masks are painted first; small masks overwrite them → every pixel
    ends up with its smallest enclosing segment label.
    """
    import geopandas as gpd
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union

    H, W    = img_shape[:2]
    labeled = np.zeros((H, W), dtype=np.int32)
    for idx, m in enumerate(masks_sorted, start=1):
        labeled[m["segmentation"]] = idx

    # Fill pixels not covered by any SAM mask via nearest-labelled-pixel
    uncovered = labeled == 0
    if uncovered.any():
        try:
            from scipy.ndimage import distance_transform_edt
            _, (row_idx, col_idx) = distance_transform_edt(
                uncovered, return_distances=True, return_indices=True
            )
            labeled[uncovered] = labeled[row_idx[uncovered], col_idx[uncovered]]
            logger.info("  filled %d uncovered pixels via nearest-segment", int(uncovered.sum()))
        except Exception as _fill_exc:
            logger.warning("pixel fill skipped: %s", _fill_exc)

    polys_by_id: dict = {}
    for geom_j, val in rio_shapes(
        labeled, mask=(labeled > 0).astype(np.uint8), transform=tile_transform
    ):
        v = int(val)
        p = shapely_shape(geom_j)
        if not p.is_empty and p.area > 0.5:
            polys_by_id.setdefault(v, []).append(p)

    rows = [{"mask_id": v, "geometry": unary_union(polys)}
            for v, polys in polys_by_id.items()]
    if not rows:
        return gpd.GeoDataFrame(columns=["mask_id", "geometry"], crs=tile_crs)
    return gpd.GeoDataFrame(rows, crs=tile_crs).to_crs("EPSG:4326")


# ── Per-segment signal helpers ────────────────────────────────────────────────
def _mask_mean(signal: np.ndarray, seg: np.ndarray, target_hw: tuple) -> float:
    import cv2
    H, W = target_hw
    if signal.shape[:2] != (H, W):
        signal = cv2.resize(signal, (W, H), interpolation=cv2.INTER_LINEAR)
    seg_r = (seg if seg.shape == (H, W)
             else cv2.resize(seg.astype(np.uint8), (W, H),
                             interpolation=cv2.INTER_NEAREST).astype(bool))
    vals = signal[seg_r]
    return float(vals.mean()) if len(vals) > 0 else 0.0


def _depth_variance(depth: np.ndarray, seg: np.ndarray, target_hw: tuple) -> float:
    import cv2
    H, W = target_hw
    if depth.shape[:2] != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
    seg_r = (seg if seg.shape == (H, W)
             else cv2.resize(seg.astype(np.uint8), (W, H),
                             interpolation=cv2.INTER_NEAREST).astype(bool))
    vals = depth[seg_r]
    return float(np.var(vals)) if len(vals) > 1 else 0.0


# ── Main analysis pipeline ────────────────────────────────────────────────────
def _run_analyze_bbox(bbox_rd: tuple) -> dict:
    """
    Full dual-year analysis:
    1.  Fetch 2022 + 2024 RGB tiles (PDOK orthophoto)
    2.  Fetch 2022 + 2024 CIR tiles (PDOK CIR)
    3.  Compute NDVI 2022 + 2024  (from CIR)
    4.  Compute depth 2022 + 2024 (Depth Anything V2)
    5.  Dense SAM2 auto-segment on 2024 RGB
    6.  Clip segments to original drawn bbox
    7.  CLIP classify every segment
    8.  Per segment: NDVI 2022 / 2024 mean, depth mean / variance
    9.  Compute construction_score, assign terrain label
    10. Return all imagery overlays + GeoJSON segments + stats
    """
    import geopandas as gpd
    from shapely.geometry import box as shapely_box

    global LAST_GPKG

    original_bbox_rd = bbox_rd
    bbox_rd_exp      = _enforce_min_extent(bbox_rd)

    dl_8  = PDOKDownloader(resolution="8cm",  crs="EPSG:28992", request_delay=0)
    dl_25 = PDOKDownloader(resolution="25cm", crs="EPSG:28992", request_delay=0)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)

        # ── 1. Fetch RGB tiles ────────────────────────────────────────────────
        rgb: dict[int, Optional[np.ndarray]] = {}
        tile_crs: dict   = {}
        tile_tfm: dict   = {}
        for year in (2022, 2024):
            p = str(tmp_p / f"rgb_{year}.tif")
            if _fetch_tile(dl_8, dl_25, bbox_rd_exp, year, p) is not None:
                img, crs, tfm    = _load_geotiff(p)
                rgb[year]        = img
                tile_crs[year]   = crs
                tile_tfm[year]   = tfm
                logger.info("RGB %d: %dx%d", year, *img.shape[:2])
            else:
                rgb[year] = None
                logger.warning("RGB %d: no PDOK coverage.", year)

        if rgb[2024] is None:
            raise HTTPException(404, "No 2024 PDOK coverage at this location.")

        img_2024  = rgb[2024]
        H, W      = img_2024.shape[:2]
        target_hw = (H, W)

        # ── 2. Fetch CIR tiles ────────────────────────────────────────────────
        cir: dict[int, Optional[np.ndarray]] = {}
        for year in (2022, 2024):
            cir[year] = _fetch_cir_tile(bbox_rd_exp, year)
            logger.info("CIR %d: %s", year, "ok" if cir[year] is not None else "unavailable")

        # ── 3. NDVI ───────────────────────────────────────────────────────────
        ndvi: dict[int, Optional[np.ndarray]] = {}
        for year in (2022, 2024):
            ndvi[year] = _compute_ndvi(cir[year], target_hw) if cir[year] is not None else None

        # ── 4. Depth ──────────────────────────────────────────────────────────
        depth: dict[int, Optional[np.ndarray]] = {}
        for year, img_np in [(2022, rgb[2022]), (2024, img_2024)]:
            depth[year] = _compute_depth(img_np) if img_np is not None else None
        logger.info("Depth: 2022=%s 2024=%s",
                    depth[2022] is not None, depth[2024] is not None)

        # ── 5. Compute NDBI + build all imagery overlays ─────────────────────
        ndbi: dict[int, Optional[np.ndarray]] = {}
        for year in (2022, 2024):
            ndbi[year] = _compute_ndbi(cir[year], target_hw) if cir[year] is not None else None

        # Only send RGB overlays to frontend — derived overlays removed from UI
        overlays: dict[str, Optional[str]] = {}
        for year in (2022, 2024):
            overlays[f"rgb_{year}"] = _img_to_b64(rgb[year]) if rgb[year] is not None else None

        # ── 6. SAM2 dense auto-segment on 2024 ───────────────────────────────
        logger.info("SAM2 auto-segment 2024 (%dx%d) …", H, W)
        masks_data = SAM2_MASK_GEN.generate(img_2024)
        logger.info("  %d raw masks", len(masks_data))
        # Filter out any mask that covers > 90 % of the tile (background catch-all)
        total_px     = H * W
        masks_data   = [m for m in masks_data if m["area"] < 0.90 * total_px]
        # Sort DESCENDING: large first, small masks overwrite → every pixel gets
        # its finest (smallest) enclosing segment
        masks_sorted = sorted(masks_data, key=lambda m: m["area"], reverse=True)
        logger.info("  %d masks after foreground filter", len(masks_sorted))

        # ── 7. Vectorize ──────────────────────────────────────────────────────
        gdf_wgs = _masks_to_gdf(masks_sorted, img_2024.shape,
                                 tile_crs[2024], tile_tfm[2024])
        logger.info("  %d vector polygons before clip", len(gdf_wgs))

        # Clip to the exact original drawn bbox
        t_rd2w = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
        ox1, oy1 = t_rd2w.transform(original_bbox_rd[0], original_bbox_rd[1])
        ox2, oy2 = t_rd2w.transform(original_bbox_rd[2], original_bbox_rd[3])
        clip_box = shapely_box(ox1, oy1, ox2, oy2)
        gdf_wgs  = gdf_wgs[gdf_wgs.geometry.intersects(clip_box)].copy()
        gdf_wgs["geometry"] = gdf_wgs.geometry.intersection(clip_box)
        gdf_wgs  = gdf_wgs[~gdf_wgs.geometry.is_empty].reset_index(drop=True)
        logger.info("  %d polygons after clip", len(gdf_wgs))

        # ── 8. CLIP classify all segments ─────────────────────────────────────
        logger.info("  CLIP classifying %d segments …", len(masks_sorted))
        clip_res = _clip_classify_masks(img_2024, masks_sorted)

        # ── 9. Per-segment metrics ────────────────────────────────────────────
        features_json = []
        records       = []
        label_counts: dict[str, int] = {}

        for _, row in gdf_wgs.iterrows():
            mask_idx = int(row["mask_id"]) - 1
            if not (0 <= mask_idx < len(masks_sorted)):
                continue

            mdata = masks_sorted[mask_idx]
            cr    = (clip_res[mask_idx] if mask_idx < len(clip_res)
                     else {"top_idx": [0], "top_scores": [0.0],
                           "all_sims": [0.0] * len(CLIP_LABELS)})
            seg   = mdata["segmentation"]   # H×W bool at tile resolution

            # NDVI 2022 / 2024
            ndvi_22 = (round(_mask_mean(ndvi[2022], seg, target_hw), 3)
                       if ndvi[2022] is not None else None)
            ndvi_24 = (round(_mask_mean(ndvi[2024], seg, target_hw), 3)
                       if ndvi[2024] is not None else None)

            # Depth mean + variance for both years
            depth_mean   = (round(_mask_mean(depth[2024], seg, target_hw), 3)
                            if depth[2024] is not None else None)
            depth_var    = (round(_depth_variance(depth[2024], seg, target_hw) * 1000, 3)
                            if depth[2024] is not None else None)
            depth_var_22 = (round(_depth_variance(depth[2022], seg, target_hw) * 1000, 3)
                            if depth[2022] is not None else None)

            # NDBI 2022 / 2024
            ndbi_22 = (round(_mask_mean(ndbi[2022], seg, target_hw), 3)
                       if ndbi[2022] is not None else None)
            ndbi_24 = (round(_mask_mean(ndbi[2024], seg, target_hw), 3)
                       if ndbi[2024] is not None else None)

            # Scores — temporal-aware
            cs      = _construction_score(cr.get("all_sims", []), ndvi_24, ndvi_22,
                                          depth_var, depth_var_22)
            terrain = _assign_terrain_label(cr.get("all_sims", []), cs, ndvi_24, ndvi_22)
            color   = TERRAIN_COLORS.get(terrain, "#666666")
            top3    = [{"label": CLIP_LABELS[i], "score": round(float(s), 3)}
                       for i, s in zip(cr["top_idx"], cr["top_scores"])]

            prop = {
                "terrain_label":      terrain,
                "construction_score": cs,
                "color":              color,
                "mean_ndvi_2022":     ndvi_22,
                "mean_ndvi_2024":     ndvi_24,
                "ndvi_delta":         round((ndvi_22 or 0.0) - (ndvi_24 or 0.0), 3),
                "mean_ndbi_2022":     ndbi_22,
                "mean_ndbi_2024":     ndbi_24,
                "depth_mean":         depth_mean,
                "depth_roughness":    depth_var,
                "area_px":            int(mdata["area"]),
                "top3_clip":          top3,
                "predicted_iou":      round(float(mdata.get("predicted_iou", 0)), 3),
                "stability_score":    round(float(mdata.get("stability_score", 0)), 3),
            }
            features_json.append({
                "type": "Feature",
                "geometry": row.geometry.__geo_interface__,
                "properties": prop,
            })
            records.append({"geometry": row.geometry, **{k: v for k, v in prop.items()
                                                          if k not in ("color", "top3_clip")}})
            label_counts[terrain] = label_counts.get(terrain, 0) + 1

        # Construction terrain drawn on top (z-order)
        features_json.sort(key=lambda f: (
            0 if f["properties"]["terrain_label"] == "likely construction terrain" else 1,
            -f["properties"]["area_px"],
        ))

        # ── 10. Save GeoPackage ───────────────────────────────────────────────
        if records:
            gdf_out  = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
            gpkg_out = ROOT.parent / "outputs" / "segments_analysis.gpkg"
            gpkg_out.parent.mkdir(exist_ok=True)
            gdf_out.to_file(str(gpkg_out), driver="GPKG", layer="segments")
            LAST_GPKG = gpkg_out
            logger.info("GeoPackage → %s", gpkg_out)

        # Original drawn bbox (WGS84) — used for fitBounds + segment clip
        west,  south,  east,  north  = _to_wgs84_bounds(original_bbox_rd)
        # Expanded tile bbox (WGS84) — used to place image overlays correctly
        tw, ts, te, tn               = _to_wgs84_bounds(bbox_rd_exp)
        n_constr = sum(1 for f in features_json
                       if f["properties"]["terrain_label"] == "likely construction terrain")

        return {
            "bounds":      [[south, west], [north, east]],  # drawn area, for fitBounds
            "tile_bounds": [[ts,    tw],   [tn,    te]],    # expanded tile, for image overlays
            "overlays":    overlays,
            "segments":    {"type": "FeatureCollection", "features": features_json},
            "stats": {
                "total":        len(features_json),
                "construction": n_constr,
                "labels":       label_counts,
            },
        }


# ── API endpoints ─────────────────────────────────────────────────────────────
@app.post("/api/analyze-bbox")
async def analyze_bbox(req: AnalyzeBboxRequest) -> dict:
    if not MODEL_READY:
        raise HTTPException(503, "Models loading — check /api/health and retry.")
    bbox_rd = _bbox_rd_from_wgs84(req.bbox_wgs84)
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=1),
            _run_analyze_bbox, bbox_rd,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("analyze-bbox error")
        raise HTTPException(500, str(exc))


@app.get("/api/bt2022")
async def bt2022_geojson():
    """Return BT2022 construction polygons as GeoJSON (WGS84, simplified)."""
    from fastapi.responses import Response
    gpkg = ROOT.parent / "data/raw/BT2022.gpkg"
    if not gpkg.exists():
        logger.error("BT2022.gpkg not found at %s", gpkg)
        return Response('{"type":"FeatureCollection","features":[]}',
                        media_type="application/geo+json")
    try:
        import geopandas as gpd
        gdf = gpd.read_file(str(gpkg))
        logger.info("BT2022: %d rows, CRS=%s", len(gdf), gdf.crs)
        if str(gdf.crs) != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        gdf["geometry"] = gdf.geometry.simplify(0.00005, preserve_topology=True)
        gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
        keep = ["geometry"]
        for col in ["identificatie", "naam", "typegebouw", "oppervlakte"]:
            if col in gdf.columns:
                gdf[col] = gdf[col].fillna("").astype(str)
                keep.append(col)
        geojson_str = gdf[keep].to_json()
        logger.info("BT2022 GeoJSON: %d features", len(gdf))
        return Response(geojson_str, media_type="application/geo+json")
    except Exception as exc:
        logger.exception("BT2022 load failed")
        return Response('{"type":"FeatureCollection","features":[]}',
                        media_type="application/geo+json")


@app.get("/api/health")
async def health() -> dict:
    return {"ready": MODEL_READY, "device": str(DEVICE) if DEVICE else "loading"}


@app.get("/api/download/results.gpkg")
async def download_gpkg() -> FileResponse:
    if LAST_GPKG is None or not LAST_GPKG.exists():
        raise HTTPException(404, "No results yet — run analysis first.")
    return FileResponse(str(LAST_GPKG), media_type="application/geopackage+sqlite3",
                        filename="segments_analysis.gpkg")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")
