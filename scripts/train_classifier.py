"""Train a binary segment-level construction classifier using BT2022 ground truth.

The script:
  1. Loads BT2022.gpkg — 3 664 confirmed construction-site polygons (EPSG:28992)
  2. Samples N sites, downloads the PDOK 2022 + 2024 RGB + CIR tiles for each
  3. Runs SAM2 dense auto-segmentation on every tile
  4. Labels each segment:
       positive  → segment IoU with any BT2022 polygon  ≥  IoU_THRESH
       negative  → no overlap at all  (hard negatives balanced 1 : 1)
  5. Extracts the same 19-dim feature vector that app.py computes at inference
  6. Trains a GradientBoostingClassifier + cross-validates
  7. Saves the model to  checkpoints/segment_classifier.pkl
  8. (Optionally) patches app.py so _construction_score() uses the model

Run (from repo root, inside the `construction` conda env):
    python scripts/train_classifier.py
    python scripts/train_classifier.py --sites 150 --no-patch

Features (19 total)
-------------------
  0   mean_ndvi_2022
  1   mean_ndvi_2024
  2   ndvi_delta           (2022 - 2024 ; positive = vegetation loss)
  3   mean_ndbi_2024       (Red-NIR)/(Red+NIR) ; positive = bare/built
  4   depth_roughness      variance of Depth-Anything-V2 within segment
  5-18 all_sims[0..13]     CLIP ViT-L/14 similarities to 14 aerial labels
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pyproj import Transformer
from shapely.geometry import box as shapely_box
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_classifier")

# ── Configuration ─────────────────────────────────────────────────────────────
BT2022_PATH   = ROOT / "data" / "raw" / "BT2022.gpkg"
CACHE_DIR     = ROOT / "data" / "tile_cache"
CKPT_OUT      = ROOT / "checkpoints" / "segment_classifier.pkl"
IOU_THRESH    = 0.25    # segment IoU with BT2022 polygon → positive label
MIN_SEG_AREA  = 200     # pixels — ignore tiny fragments
RANDOM_SEED   = 42

CLIP_LABELS = [
    "construction site with cranes and machinery",
    "building under construction concrete frame",
    "excavation pit earthwork foundation",
    "scaffolding on building facade",
    "demolition site rubble and debris",
    "concrete pour steel structure being built",
    "road asphalt pavement",
    "parking lot with vehicles",
    "residential house rooftops",
    "office or commercial building roof",
    "industrial warehouse factory roof",
    "green vegetation trees park grass",
    "water canal river pond",
    "bare soil gravel sand field",
]

# ── Model globals (loaded once) ───────────────────────────────────────────────
SAM2_MASK_GEN   = None
CLIP_MODEL      = None
CLIP_PROCESSOR  = None
CLIP_TEXT_FEATS = None
DEPTH_MODEL     = None
DEPTH_PROCESSOR = None
DEVICE          = None


def load_models() -> None:
    global SAM2_MASK_GEN, CLIP_MODEL, CLIP_PROCESSOR, CLIP_TEXT_FEATS
    global DEPTH_MODEL, DEPTH_PROCESSOR, DEVICE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = device
    log.info("Device: %s", device)

    log.info("Loading SAM2 Hiera-L …")
    from sam2.build_sam import build_sam2_hf
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    sam_model = build_sam2_hf("facebook/sam2.1-hiera-large", device=device)
    sam_model.eval()
    SAM2_MASK_GEN = SAM2AutomaticMaskGenerator(
        model=sam_model,
        points_per_side=32,
        pred_iou_thresh=0.65,
        stability_score_thresh=0.75,
        min_mask_region_area=MIN_SEG_AREA,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )
    log.info("SAM2 ready.")

    log.info("Loading CLIP ViT-L/14 …")
    from transformers import CLIPProcessor, CLIPModel
    CLIP_MODEL     = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    CLIP_MODEL.eval()
    with torch.no_grad():
        inp  = CLIP_PROCESSOR(text=CLIP_LABELS, return_tensors="pt", padding=True).to(device)
        tout = CLIP_MODEL.text_model(input_ids=inp["input_ids"],
                                     attention_mask=inp["attention_mask"])
        CLIP_TEXT_FEATS = F.normalize(
            CLIP_MODEL.text_projection(tout.pooler_output), dim=-1
        )
    log.info("CLIP ready — %d labels.", len(CLIP_LABELS))

    log.info("Loading Depth Anything V2 Small …")
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    DEPTH_PROCESSOR = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    DEPTH_MODEL = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    ).to(device)
    DEPTH_MODEL.eval()
    log.info("Depth Anything V2 ready.")


# ── Tile helpers (mirrors dashboard/app.py exactly) ──────────────────────────
CIR_WMS_URL = "https://service.pdok.nl/hwh/luchtfotocir/wms/v1_0"

def _enforce_min_extent(bbox_rd, min_half=300.0):
    xmin, ymin, xmax, ymax = bbox_rd
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half   = max((xmax - xmin) / 2, (ymax - ymin) / 2, min_half)
    return (cx - half, cy - half, cx + half, cy + half)


def _fetch_rgb(bbox_rd, year, cache_dir: Path) -> np.ndarray | None:
    from data.pdok_downloader import PDOKDownloader
    import rasterio, shutil
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"rgb_{year}_{int(bbox_rd[0])}_{int(bbox_rd[1])}.tif"
    cached = cache_dir / key
    if cached.exists():
        with rasterio.open(str(cached)) as src:
            img = src.read([1, 2, 3]).transpose(1, 2, 0)
        return img.copy()

    dl_8  = PDOKDownloader(resolution="8cm",  crs="EPSG:28992", request_delay=0)
    dl_25 = PDOKDownloader(resolution="25cm", crs="EPSG:28992", request_delay=0)
    with tempfile.TemporaryDirectory() as tmp:
        p = str(Path(tmp) / "tile.tif")
        r = dl_8.download_tile(bbox_rd, year, p)
        if r is None:
            p25 = p.replace(".tif", "_25.tif")
            r   = dl_25.download_tile(bbox_rd, year, p25)
            if r is not None:
                shutil.move(p25, p)
        if r is None:
            return None
        shutil.copy(p, str(cached))
        with rasterio.open(p) as src:
            img = src.read([1, 2, 3]).transpose(1, 2, 0)
            crs, tfm = src.crs, src.transform
    return img.copy(), crs, tfm


def _fetch_rgb_with_meta(bbox_rd, year, cache_dir: Path):
    """Returns (img_np, crs, transform) or (None, None, None)."""
    import rasterio, shutil
    from data.pdok_downloader import PDOKDownloader
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"rgb_{year}_{int(bbox_rd[0])}_{int(bbox_rd[1])}.tif"
    cached = cache_dir / key
    meta_key = cache_dir / key.replace(".tif", "_meta.pkl")
    if cached.exists() and meta_key.exists():
        with rasterio.open(str(cached)) as src:
            img = src.read([1, 2, 3]).transpose(1, 2, 0)
            crs, tfm = src.crs, src.transform
        return img.copy(), crs, tfm

    dl_8  = PDOKDownloader(resolution="8cm",  crs="EPSG:28992", request_delay=0)
    dl_25 = PDOKDownloader(resolution="25cm", crs="EPSG:28992", request_delay=0)
    with tempfile.TemporaryDirectory() as tmp:
        p = str(Path(tmp) / "tile.tif")
        r = dl_8.download_tile(bbox_rd, year, p)
        if r is None:
            p25 = p.replace(".tif", "_25.tif")
            r   = dl_25.download_tile(bbox_rd, year, p25)
            if r is not None:
                shutil.move(p25, p)
        if r is None:
            return None, None, None
        shutil.copy(p, str(cached))
        with rasterio.open(p) as src:
            img = src.read([1, 2, 3]).transpose(1, 2, 0)
            crs, tfm = src.crs, src.transform
    return img.copy(), crs, tfm


def _fetch_cir(bbox_rd, year, cache_dir: Path) -> np.ndarray | None:
    import requests as _req
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_dir / f"cir_{year}_{int(bbox_rd[0])}_{int(bbox_rd[1])}.npy"
    if key.exists():
        return np.load(str(key))
    params = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS": f"{year}_ortho25", "STYLES": "",
        "CRS": "EPSG:28992",
        "BBOX": f"{bbox_rd[0]},{bbox_rd[1]},{bbox_rd[2]},{bbox_rd[3]}",
        "WIDTH": "1024", "HEIGHT": "1024", "FORMAT": "image/png",
    }
    try:
        r  = _req.get(CIR_WMS_URL, params=params, timeout=15)
        ct = r.headers.get("Content-Type", "")
        if "xml" in ct or not r.content:
            return None
        arr = np.array(Image.open(__import__("io").BytesIO(r.content)).convert("RGB"))
        if (arr == 255).all(axis=-1).mean() > 0.95:
            return None
        np.save(str(key), arr)
        return arr
    except Exception as e:
        log.warning("CIR fetch failed: %s", e)
        return None


def _compute_ndvi(cir_np, target_hw):
    import cv2
    nir  = cir_np[:, :, 0].astype(np.float32)
    red  = cir_np[:, :, 1].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)
    H, W = target_hw
    return cv2.resize(ndvi, (W, H), interpolation=cv2.INTER_LINEAR)


def _compute_ndbi(cir_np, target_hw):
    import cv2
    nir  = cir_np[:, :, 0].astype(np.float32)
    red  = cir_np[:, :, 1].astype(np.float32)
    ndbi = (red - nir) / (red + nir + 1e-6)
    H, W = target_hw
    return cv2.resize(ndbi, (W, H), interpolation=cv2.INTER_LINEAR)


def _compute_depth(img_np) -> np.ndarray | None:
    import cv2
    if DEPTH_MODEL is None:
        return None
    H, W = img_np.shape[:2]
    try:
        with torch.no_grad():
            inp   = DEPTH_PROCESSOR(images=Image.fromarray(img_np),
                                    return_tensors="pt").to(DEVICE)
            depth = DEPTH_MODEL(**inp).predicted_depth.squeeze().cpu().numpy().astype(np.float32)
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        mn, mx = depth.min(), depth.max()
        return (depth - mn) / (mx - mn + 1e-6)
    except Exception as e:
        log.warning("Depth failed: %s", e)
        return None


def _clip_classify(img_np, masks_data) -> list[dict]:
    n = len(CLIP_LABELS)
    empty = {"top_idx": [0], "top_scores": [0.0], "all_sims": [0.0] * n}
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
        crop[~m["segmentation"][y1:y2, x1:x2]] = 128
        crops.append(Image.fromarray(crop))
    results, bs = [], 32
    with torch.no_grad():
        for i in range(0, len(crops), bs):
            batch = crops[i: i + bs]
            inp   = CLIP_PROCESSOR(images=batch, return_tensors="pt", padding=True).to(DEVICE)
            v_out = CLIP_MODEL.vision_model(pixel_values=inp["pixel_values"])
            feats = F.normalize(CLIP_MODEL.visual_projection(v_out.pooler_output), dim=-1)
            sims  = (feats @ CLIP_TEXT_FEATS.T).cpu()
            for j in range(len(batch)):
                top = sims[j].topk(3)
                results.append({
                    "top_idx":    top.indices.tolist(),
                    "top_scores": top.values.tolist(),
                    "all_sims":   sims[j].tolist(),
                })
    return results


def _mask_mean(signal, seg, target_hw):
    import cv2
    H, W = target_hw
    if signal.shape[:2] != (H, W):
        signal = cv2.resize(signal, (W, H), interpolation=cv2.INTER_LINEAR)
    seg_r = (seg if seg.shape == (H, W)
             else cv2.resize(seg.astype(np.uint8), (W, H),
                             interpolation=cv2.INTER_NEAREST).astype(bool))
    vals = signal[seg_r]
    return float(vals.mean()) if len(vals) > 0 else 0.0


def _depth_variance(depth, seg, target_hw):
    import cv2
    H, W = target_hw
    if depth.shape[:2] != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
    seg_r = (seg if seg.shape == (H, W)
             else cv2.resize(seg.astype(np.uint8), (W, H),
                             interpolation=cv2.INTER_NEAREST).astype(bool))
    vals = depth[seg_r]
    return float(np.var(vals)) if len(vals) > 1 else 0.0


def _seg_to_polygon(m, tile_transform, tile_crs):
    """Convert a SAM2 mask dict → shapely geometry in WGS84."""
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape as sh_shape, mapping
    from shapely.ops import unary_union
    import geopandas as gpd

    seg  = m["segmentation"].astype(np.uint8)
    polys = [sh_shape(g) for g, v in rio_shapes(seg, mask=seg, transform=tile_transform) if int(v) == 1]
    if not polys:
        return None
    geom_rd = unary_union(polys)
    gdf = gpd.GeoDataFrame(geometry=[geom_rd], crs=tile_crs).to_crs("EPSG:4326")
    return gdf.geometry.iloc[0]


def _iou(geom_a, geom_b) -> float:
    try:
        inter = geom_a.intersection(geom_b).area
        union = geom_a.union(geom_b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


# ── Per-tile feature extraction ───────────────────────────────────────────────
def process_tile(bbox_rd_exp, site_geom_wgs84, cache_dir) -> list[dict]:
    """
    Returns list of dicts with keys:
      features: np.ndarray (19,)
      label:    int  (1=construction, 0=other)
    """
    records = []

    # 1. Download tiles
    img22, crs22, tfm22 = _fetch_rgb_with_meta(bbox_rd_exp, 2022, cache_dir)
    img24, crs24, tfm24 = _fetch_rgb_with_meta(bbox_rd_exp, 2024, cache_dir)
    if img24 is None:
        return records
    H, W      = img24.shape[:2]
    target_hw = (H, W)

    cir22 = _fetch_cir(bbox_rd_exp, 2022, cache_dir)
    cir24 = _fetch_cir(bbox_rd_exp, 2024, cache_dir)

    # 2. Compute signals
    ndvi22 = _compute_ndvi(cir22, target_hw) if cir22 is not None else None
    ndvi24 = _compute_ndvi(cir24, target_hw) if cir24 is not None else None
    ndbi24 = _compute_ndbi(cir24, target_hw) if cir24 is not None else None
    depth24 = _compute_depth(img24)

    # 3. SAM2 segmentation
    masks = SAM2_MASK_GEN.generate(img24)
    total_px = H * W
    masks = [m for m in masks if m["area"] < 0.90 * total_px and m["area"] >= MIN_SEG_AREA]
    if not masks:
        return records

    # 4. CLIP
    clip_res = _clip_classify(img24, masks)

    # 5. Label via IoU with BT2022 site polygon
    for idx, (m, cr) in enumerate(zip(masks, clip_res)):
        seg_geom = _seg_to_polygon(m, tfm24, crs24)
        if seg_geom is None or seg_geom.is_empty:
            continue

        iou_score = _iou(seg_geom, site_geom_wgs84)
        label = 1 if iou_score >= IOU_THRESH else 0

        # Feature extraction
        n22 = round(_mask_mean(ndvi22, m["segmentation"], target_hw), 4) if ndvi22 is not None else 0.0
        n24 = round(_mask_mean(ndvi24, m["segmentation"], target_hw), 4) if ndvi24 is not None else 0.0
        nb24 = round(_mask_mean(ndbi24, m["segmentation"], target_hw), 4) if ndbi24 is not None else 0.0
        dv24 = round(_depth_variance(depth24, m["segmentation"], target_hw) * 1000, 4) if depth24 is not None else 0.0

        all_sims = cr.get("all_sims", [0.0] * len(CLIP_LABELS))
        feat = np.array(
            [n22, n24, n22 - n24, nb24, dv24] + list(all_sims),
            dtype=np.float32
        )  # shape (19,)

        records.append({"features": feat, "label": label, "iou": iou_score,
                         "area_px": int(m["area"])})
    return records


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites",    type=int,  default=300,       help="Number of BT2022 sites to process")
    parser.add_argument("--save",     type=str,  default=str(CKPT_OUT), help="Output .pkl path")
    parser.add_argument("--cache",    type=str,  default=str(CACHE_DIR), help="Tile cache dir")
    parser.add_argument("--no-patch", action="store_true", help="Skip patching app.py after training")
    args = parser.parse_args()

    cache_dir = Path(args.cache)
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load ground truth ─────────────────────────────────────────────────────
    log.info("Loading BT2022.gpkg …")
    gdf_bt = gpd.read_file(str(BT2022_PATH))
    log.info("  %d construction polygons (CRS: %s)", len(gdf_bt), gdf_bt.crs)
    gdf_bt = gdf_bt.to_crs("EPSG:28992")

    # Filter: only include sites with reasonable area (≥ 500 m²)
    gdf_bt = gdf_bt[gdf_bt.geometry.area >= 500].reset_index(drop=True)
    log.info("  %d sites after area filter", len(gdf_bt))

    # Sample N sites
    n_sites = min(args.sites, len(gdf_bt))
    rng = np.random.default_rng(RANDOM_SEED)
    idxs = rng.choice(len(gdf_bt), size=n_sites, replace=False)
    sample = gdf_bt.iloc[idxs].reset_index(drop=True)
    log.info("  Sampled %d sites for training.", n_sites)

    # ── Convert site polygons to WGS84 for IoU matching ──────────────────────
    sample_wgs = sample.to_crs("EPSG:4326")
    t_rd2wgs = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

    # ── Load models ───────────────────────────────────────────────────────────
    load_models()

    # ── Process each site ─────────────────────────────────────────────────────
    all_records = []
    n_pos = n_neg = 0

    for i, (row_rd, row_wgs) in enumerate(zip(sample.itertuples(), sample_wgs.itertuples())):
        bounds = row_rd.geometry.bounds          # (xmin, ymin, xmax, ymax) in EPSG:28992
        bbox_rd_exp = _enforce_min_extent(bounds)
        site_geom   = row_wgs.geometry

        log.info("[%3d/%d] Site %s  bbox=(%.0f, %.0f, %.0f, %.0f)",
                 i + 1, n_sites, getattr(row_rd, "naam", ""), *bbox_rd_exp)
        try:
            recs = process_tile(bbox_rd_exp, site_geom, cache_dir)
        except Exception as e:
            log.warning("  site %d failed: %s", i, e)
            continue

        pos = [r for r in recs if r["label"] == 1]
        neg = [r for r in recs if r["label"] == 0]
        n_pos += len(pos)
        n_neg += len(neg)
        all_records.extend(recs)
        log.info("  → %d segs  (%d pos / %d neg)  cumulative: %d+ / %d-",
                 len(recs), len(pos), len(neg), n_pos, n_neg)

    if not all_records:
        log.error("No training records collected — exiting.")
        sys.exit(1)

    X = np.stack([r["features"] for r in all_records])
    y = np.array([r["label"]    for r in all_records])
    log.info("Dataset: %d samples  (pos=%d  neg=%d)  features=%d",
             len(y), y.sum(), (y == 0).sum(), X.shape[1])

    # ── Balance classes (undersample majority) ────────────────────────────────
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_keep  = min(len(pos_idx), len(neg_idx))
    rng2    = np.random.default_rng(RANDOM_SEED + 1)
    keep_pos = rng2.choice(pos_idx, n_keep, replace=False)
    keep_neg = rng2.choice(neg_idx, n_keep, replace=False)
    keep = np.concatenate([keep_pos, keep_neg])
    rng2.shuffle(keep)
    X_bal, y_bal = X[keep], y[keep]
    log.info("Balanced dataset: %d samples (%d pos + %d neg)", len(y_bal), n_keep, n_keep)

    # ── Train GradientBoosting ────────────────────────────────────────────────
    log.info("Training GradientBoostingClassifier …")
    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=RANDOM_SEED,
    )

    # Cross-validate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(clf, X_bal, y_bal, cv=cv, scoring="roc_auc", n_jobs=-1)
    log.info("5-fold CV ROC-AUC: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    # Fit on full balanced set
    clf.fit(X_bal, y_bal)
    y_pred = clf.predict(X_bal)
    log.info("\n%s", classification_report(y_bal, y_pred, target_names=["other", "construction"]))

    # Feature importance
    feat_names = ["ndvi_22", "ndvi_24", "ndvi_delta", "ndbi_24", "depth_rough"] + \
                 [f"clip_{i}" for i in range(len(CLIP_LABELS))]
    top_feats = sorted(zip(feat_names, clf.feature_importances_),
                       key=lambda x: x[1], reverse=True)[:10]
    log.info("Top-10 features by importance:")
    for name, imp in top_feats:
        log.info("  %-20s  %.4f", name, imp)

    # ── Save classifier ───────────────────────────────────────────────────────
    payload = {
        "clf":         clf,
        "feat_names":  feat_names,
        "cv_roc_auc":  float(cv_scores.mean()),
        "n_train":     int(len(y_bal)),
        "n_sites":     n_sites,
        "clip_labels": CLIP_LABELS,
    }
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)
    log.info("Classifier saved → %s", save_path)

    if not args.no_patch:
        _patch_app_py(save_path)


def _patch_app_py(ckpt_path: Path) -> None:
    """Insert classifier loading + usage into dashboard/app.py automatically."""
    app_py = ROOT / "dashboard" / "app.py"
    src = app_py.read_text(encoding="utf-8")

    loader_code = '''
# ── Segment classifier (trained on BT2022) ────────────────────────────────────
import pickle as _pickle
_CLASSIFIER   = None
_CLF_FEAT_NAMES = None
_CLF_PATH = Path(__file__).parent.parent / "checkpoints" / "segment_classifier.pkl"

def _load_classifier() -> None:
    global _CLASSIFIER, _CLF_FEAT_NAMES
    if _CLF_PATH.exists():
        with open(_CLF_PATH, "rb") as _f:
            _p = _pickle.load(_f)
        _CLASSIFIER   = _p["clf"]
        _CLF_FEAT_NAMES = _p["feat_names"]
        logger.info("Segment classifier loaded ← %s  (CV-AUC=%.3f  n_train=%d)",
                    _CLF_PATH, _p["cv_roc_auc"], _p["n_train"])
    else:
        logger.info("No trained classifier found at %s — using heuristic scoring.", _CLF_PATH)
'''

    # Only patch once
    if "_CLASSIFIER" in src:
        log.info("app.py already patched — skipping.")
        return

    # Insert loader after the last import block (after "LAST_GPKG" line)
    insert_after = "LAST_GPKG: Optional[Path] = None"
    if insert_after not in src:
        log.warning("Could not find insertion point in app.py — skipping patch.")
        return

    src = src.replace(insert_after, insert_after + "\n" + loader_code)

    # Patch startup to also call _load_classifier()
    src = src.replace(
        "loop.run_in_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1), _load_model)",
        "loop.run_in_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1), _load_model)\n"
        "    _load_classifier()"
    )

    # Patch _construction_score to use classifier when available
    old_score_return = "return round(0.25 * ndvi_abs + 0.30 * ndvi_delta + 0.20 * roughness + 0.25 * clip_s, 3)"
    new_score_return = """\
    # ── Use trained classifier if available ──────────────────────────────────
    if _CLASSIFIER is not None:
        feat = np.array([
            ndvi_22 if ndvi_22 is not None else 0.0,
            ndvi_24 if ndvi_24 is not None else 0.0,
            (ndvi_22 or 0.0) - (ndvi_24 or 0.0),
            0.0,        # ndbi_24 not passed here — heuristic fallback
            depth_var_24 * 1000 if depth_var_24 is not None else 0.0,
        ] + (all_sims if all_sims else [0.0] * 14), dtype=np.float32).reshape(1, -1)
        prob = float(_CLASSIFIER.predict_proba(feat)[0, 1])
        return round(prob, 3)
    return round(0.25 * ndvi_abs + 0.30 * ndvi_delta + 0.20 * roughness + 0.25 * clip_s, 3)"""

    if old_score_return in src:
        src = src.replace(old_score_return, new_score_return)
        log.info("Patched _construction_score() to use classifier.")
    else:
        log.warning("Could not find _construction_score return line — skipping score patch.")

    app_py.write_text(src, encoding="utf-8")
    log.info("app.py patched successfully.")


if __name__ == "__main__":
    main()
