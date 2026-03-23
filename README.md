# Labwerk — Construction Site Change Detector

> **One-shot construction site detection from aerial imagery using a frozen SAM2 encoder, corpus prototypes, and spatially-routed change detection.**

The Netherlands registers ~3 700 active construction sites in its 2022 land-use database. This project detects which of those sites are **still under construction in 2024** and which **new sites have appeared** — all from freely available PDOK aerial orthophotos, without any 2024 labels.

---

## The Core Insight

Construction sites are not objects. A car looks like a car. A construction site looks like bare soil, or scaffolding, or a concrete slab, or a tower crane — whatever stage of a multi-year process is visible that day. A single prototype vector cannot capture this diversity.

This project addresses that with two ideas working together:

**1 — Corpus prototypes instead of a single support embedding**
All 2022 construction site tiles are passed through the frozen SAM2 encoder. Foreground features are extracted and clustered (K-means, k=20). The 20 centroids represent the full visual vocabulary of Dutch construction sites — bare ground, machinery, formwork, partial structures — without any 2024 supervision.

**2 — Spatially-routed dual-path detection**
The 2022 land-use mask acts as a hard spatial router:

```
Preexisting sites  (was construction in 2022)
  → score = appearance_2024                           ← change signal would suppress these
                                                         (they look similar across years)
New sites          (was NOT construction in 2022)
  → score = appearance_2024 × change_2022→2024        ← needs both appearance AND change

Combined = preexisting_score + new_score              ← spatially disjoint, no double-counting
```

This means a site that completed construction between 2022 and 2024 (appearance score low in 2024) is correctly suppressed, while a stable field that was never construction (change score low) is not flagged.

---

## Architecture

```
                          ┌──────────────────────────────────────────┐
                          │          SAM2 Image Encoder               │
                          │    (Hiera-L backbone + FPN neck)          │
                          │             FROZEN                        │
                          └────────────┬────────────┬────────────────┘
                                       │            │
                              tile_2022│            │tile_2024
                                       ▼            ▼
                              feats_2022          feats_2024  [B, 256, H/16, W/16]
                                       │            │
                          ┌────────────┘            └────────────────┐
                          │                                          │
                          ▼                                          ▼
               ┌─────────────────────┐                 ┌─────────────────────┐
               │    ChangeScorer     │                 │   AppearanceScorer  │
               │  (feature delta)    │                 │  (cosine vs K=20    │
               │                     │                 │   corpus prototypes)│
               └────────┬────────────┘                 └──────────┬──────────┘
                        │  change_map                              │  appear_map
                        │  [B,1,H_f,W_f]                          │  [B,1,H_f,W_f]
                        └──────────────────┬───────────────────────┘
                                           │
                              mask_2022 (rasterised polygons)
                                           │
                              ┌────────────▼────────────┐
                              │     Spatial Router       │
                              │                          │
                              │  preexisting =           │
                              │    appear × mask_2022    │
                              │                          │
                              │  new_sites =             │
                              │    appear × change       │
                              │    × (1 − mask_2022)     │
                              │                          │
                              │  output = preexisting    │
                              │         + new_sites      │
                              └────────────┬─────────────┘
                                           │
                                    prob_map [0,1]
                                  (GeoTIFF, EPSG:28992)
```

**Trainable parameters:** `AppearanceScorer` + `ChangeScorer` (~2 M params)
**Frozen:** SAM2 Hiera-L encoder + FPN neck (~300 M params)

---

## Data

| Source | Description |
|---|---|
| [PDOK Luchtfoto WMS](https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0) | 8 cm orthoHR aerial photography, year-specific layers (`2022_orthoHR`, `2024_orthoHR`) |
| `BT2022.gpkg` | 3 664 construction site polygons from the 2022 Dutch land-use survey (EPSG:28992) |

Tiles are downloaded automatically by `scripts/download_pdok.py`. No local imagery is required to get started.

> **WMS quirks discovered during development**
> - PDOK does **not** support the `TIME` dimension — use year-specific layer names
> - Bounding boxes smaller than ~600 m return blank white images — the downloader enforces a 600 m minimum extent centred on each polygon
> - `2021_ortho25` is absent from the service entirely

---

## Pipeline

```
BT2022.gpkg
    │
    ▼
scripts/download_pdok.py          # Download 2022 + 2024 orthoHR tiles for all 3 664 sites
    │                             # → data/tiles/<site_id>/{tile_2022.tif, tile_2024.tif, mask_2022.png}
    ▼
scripts/build_prototypes.py       # SAM2-encode all 2022 tiles → KMeans(k=20) → prototypes.pt
    │                             # → checkpoints/prototypes.pt
    ▼
scripts/train.py                  # Train AppearanceScorer + ChangeScorer
    │                             # → checkpoints/best.pt  |  logs/ (TensorBoard)
    ▼
scripts/evaluate.py               # IoU / F1 / AUC on validation split
    │
    ▼
scripts/score_area.py             # Produce probability GeoTIFF for any bbox
                                  # → outputs/construction_prob_2024.tif
```

---

## Quick Start

### 1 — Environment

```bash
conda create -n construction python=3.11
conda activate construction
pip install -r requirements.txt
```

### 2 — Download tiles

```bash
python scripts/download_pdok.py \
    --polygons data/raw/BT2022.gpkg \
    --years 2022 2024 \
    --resolution 8cm \
    --out data/tiles/
```

~3 700 sites × 2 years ≈ 2 hours on a standard connection. The script is resumable — already-downloaded tiles are skipped.

### 3 — Build prototypes

```bash
python scripts/build_prototypes.py \
    --tiles data/tiles/ \
    --k 20 \
    --out checkpoints/prototypes.pt
```

### 4 — Train

```bash
python scripts/train.py --config configs/train.yaml
```

Monitor with TensorBoard:

```bash
tensorboard --logdir logs/
```

### 5 — Score an area

```bash
python scripts/score_area.py \
    --bbox 100000,450000,150000,500000 \
    --prototypes checkpoints/prototypes.pt \
    --checkpoint checkpoints/best.pt \
    --mask2022 data/raw/BT2022.gpkg \
    --out outputs/construction_prob_2024.tif
```

The output is a single-band float32 GeoTIFF (EPSG:28992) with values in [0, 1] — 1 = high probability of active construction in 2024.

---

## Dev Server Launcher

All pipeline stages are pre-configured in `.claude/launch.json` for one-click launch from Claude Code or any compatible IDE.

| Server | Command |
|---|---|
| `download-pdok` | Downloads all WMS tiles |
| `build-prototypes` | Builds the K=20 corpus |
| `train` | Trains the model |
| `evaluate` | Runs validation metrics |
| `score-area` | Produces output GeoTIFF |
| `tensorboard` | Training dashboard (port 6006) |

---

## Project Structure

```
Labwerk/
├── configs/
│   ├── base.yaml            # Shared hyperparameters
│   ├── train.yaml           # Training overrides
│   └── inference.yaml       # Inference overrides
├── data/
│   ├── pdok_downloader.py   # PDOK WMS client (requests-based)
│   ├── dataset.py           # ConstructionChangeDataset
│   └── transforms.py        # Augmentation pipeline
├── models/
│   ├── detector.py          # ConstructionChangeDetector (top-level)
│   ├── appearance_scorer.py # Cosine similarity vs corpus prototypes
│   ├── change_scorer.py     # Feature delta encoder
│   ├── corpus_prototype.py  # KMeans prototype builder
│   └── feature_utils.py     # Shared feature utilities
├── losses/
│   └── segmentation_losses.py  # BCE + Dice loss
├── training/
│   └── trainer.py           # Training loop with AMP + checkpointing
├── evaluation/
│   └── metrics.py           # IoU, F1, AUC
├── scripts/
│   ├── download_pdok.py
│   ├── build_prototypes.py
│   ├── train.py
│   ├── evaluate.py
│   └── score_area.py
└── data/
    ├── raw/
    │   └── BT2022.gpkg      # 3 664 construction polygons
    └── tiles/               # Downloaded tile pairs (gitignored)
```

---

## Requirements

- Python 3.11+
- PyTorch ≥ 2.3
- CUDA GPU recommended (SAM2 Hiera-L encoder is large)
- ~50 GB disk for all 3 664 × 2 tiles at 8 cm orthoHR

---

## License

MIT
