---
title: SamSpade
emoji: 🏗
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

<div align="center">

# 🕵️ SamSpade

**Construction-terrain detection from aerial imagery — Netherlands**

Draw a bounding box anywhere in the Netherlands. Get dense SAM2 segments classified as construction, vegetation, paved, roof, water, or bare soil — overlaid on live PDOK aerial imagery.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![SAM2](https://img.shields.io/badge/SAM2-Hiera--L-0064FF?style=flat-square)](https://github.com/facebookresearch/sam2)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--L/14-412991?style=flat-square)](https://huggingface.co/openai/clip-vit-large-patch14)
[![PDOK](https://img.shields.io/badge/Imagery-PDOK%208cm-00A550?style=flat-square)](https://service.pdok.nl)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## What it does

1. **Draw a bounding box** anywhere in the Netherlands
2. SamSpade fetches **2022 and 2024 RGB + CIR** orthophotos from PDOK (8 cm/px, free)
3. Runs **dense SAM2 auto-segmentation** — every pixel gets a segment
4. For each segment, computes:
   - **NDVI 2022 & 2024** from CIR (NIR + Red)
   - **ΔNDVI** — vegetation loss between years
   - **NDBI** — normalised difference built-up index
   - **Depth roughness** via Depth Anything V2
   - **RGB pixel statistics** — green fraction, grey fraction, uniformity
   - **CLIP similarity** across 14 aerial land-cover labels
5. A **trained GradientBoosting classifier** (trained on 3 664 BT2022 ground-truth sites) assigns a construction probability to each segment
6. Hard exclusion gates prevent false positives:
   - Green pixels > 40 % → **vegetation**
   - NDVI > 0.35 → **vegetation**
   - Grey pixels > 55 % + temporally stable → **paved / railway**
   - Uniform colour > 65 % → **roof / building**
7. Results are **GeoJSON polygons** drawn over the live PDOK WMS base map

---

## Terrain classes

| Class | Colour | Description |
|---|---|---|
| likely construction terrain | 🟥 `#FF4500` | Bare/disturbed ground, active construction zone |
| exposed soil / bare ground | 🟫 `#CD853F` | Cleared land, agricultural bare soil |
| vegetation | 🟩 `#22BB44` | Grass, trees, crops, parks |
| water | 🟦 `#1E90FF` | Canals, rivers, ponds |
| roof / building | 🟣 `#9370DB` | Building rooftops |
| paved surface | ⬜ `#778899` | Roads, car parks, railway tracks |
| shadow / unknown | ⬛ `#505060` | Deep shadow, unclassified |

---

## Dashboard

The dashboard is a FastAPI + Leaflet.js app with a QGIS-style sidebar.

**Base maps** (always live, no download):
- PDOK RGB 2024 (default)
- PDOK RGB 2022 (radio switch)

**Result layers** (polygons on top of the basemap):
- All segments — coloured by terrain class
- Construction only — orange-red highlight
- BT2022 reference sites — dashed orange outline

Each polygon is clickable and shows a popup with NDVI, ΔNDVI, NDBI, depth roughness, green/grey/uniform pixel fractions, and top CLIP similarities.

---

## Quickstart

### Requirements

```bash
# Create and activate the environment
conda create -n construction python=3.11
conda activate construction

# PyTorch with CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Core dependencies
pip install sam2 transformers fastapi uvicorn geopandas shapely pyproj \
            rasterio pillow numpy scikit-learn scipy requests
```

### Run the dashboard

```bash
conda activate construction
cd /path/to/SAMSpade
python -m uvicorn dashboard.app:app --host 0.0.0.0 --port 8001
```

Open **http://localhost:8001** in your browser.

Models load on startup (~60–90 sec). The status bar turns green when ready.

### Usage

1. Click **✏ Draw bbox on map**
2. Draw a rectangle over an area of interest in the Netherlands
3. Click **▶ Analyze Area**
4. Wait ~30–60 sec (SAM2 + CLIP + depth inference)
5. Segments appear as coloured polygons on the live PDOK imagery
6. Switch base map between 2024 and 2022 with the radio buttons
7. Click any polygon for detailed metrics
8. Download results as GeoPackage

---

## Train the segment classifier

The construction classifier is trained on **BT2022.gpkg** — 3 664 confirmed construction-site polygons from the Dutch Building and Land-use Register.

```bash
conda activate construction
cd /path/to/SAMSpade

# Quick run — 50 sites (~20 min including tile download)
python scripts/train_classifier.py --sites 50

# Full training — 300 sites (~90 min, recommended)
python scripts/train_classifier.py --sites 300
```

Tiles are cached in `data/tile_cache/` so re-runs only redo inference.

**Feature vector (19-dim):**

| # | Feature | Description |
|---|---|---|
| 0 | `ndvi_22` | Mean NDVI in segment, 2022 |
| 1 | `ndvi_24` | Mean NDVI in segment, 2024 |
| 2 | `ndvi_delta` | NDVI change (2022 − 2024); positive = vegetation loss |
| 3 | `ndbi_24` | Normalised difference built-up index, 2024 |
| 4 | `depth_rough` | Depth variance (Depth Anything V2) |
| 5–18 | `clip_0..13` | CLIP ViT-L/14 similarity to 14 aerial land-cover prompts |

**Output:** `checkpoints/segment_classifier.pkl`
The script automatically patches `dashboard/app.py` to use the trained model at inference.

---

## API

### `POST /api/analyze-bbox`

```json
{ "bbox_wgs84": [west, south, east, north] }
```

**Response:**
```json
{
  "bounds": [[south, west], [north, east]],
  "segments": {
    "type": "FeatureCollection",
    "features": [{
      "type": "Feature",
      "geometry": { "type": "Polygon", "coordinates": [...] },
      "properties": {
        "terrain_label": "likely construction terrain",
        "construction_score": 0.82,
        "color": "#FF4500",
        "mean_ndvi_2022": 0.31,
        "mean_ndvi_2024": 0.04,
        "ndvi_delta": 0.27,
        "mean_ndbi_2024": 0.18,
        "depth_roughness": 14.2,
        "green_fraction": 0.03,
        "grey_fraction": 0.12,
        "uniform_fraction": 0.21,
        "barren_fraction": 0.61,
        "area_px": 4820,
        "top3_clip": [
          {"label": "excavation pit earthwork foundation", "score": 0.31},
          {"label": "bare soil gravel sand field", "score": 0.28},
          {"label": "construction site with cranes and machinery", "score": 0.22}
        ]
      }
    }]
  },
  "stats": {
    "total": 147,
    "construction": 12,
    "labels": { "vegetation": 61, "paved surface": 38, ... }
  }
}
```

### `GET /api/bt2022`
Returns BT2022 construction reference polygons as GeoJSON for the whole Netherlands.

### `GET /api/health`
```json
{ "ready": true, "device": "cuda" }
```

### `GET /api/download/results.gpkg`
Downloads the last analysis result as a GeoPackage file.

---

## Project structure

```
SAMSpade/
├── dashboard/
│   ├── app.py                  # FastAPI backend — full analysis pipeline
│   └── static/
│       └── index.html          # Leaflet.js frontend
├── scripts/
│   ├── train_classifier.py     # Train BT2022-based segment classifier
│   └── download_pdok.py        # Standalone PDOK tile downloader
├── data/
│   ├── raw/
│   │   └── BT2022.gpkg         # 3 664 construction-site ground truth polygons
│   └── tile_cache/             # Cached PDOK tiles (auto-created by training)
├── checkpoints/
│   └── segment_classifier.pkl  # Trained classifier (created by train_classifier.py)
└── outputs/
    └── segments_analysis.gpkg  # Last analysis result (auto-created)
```

---

## Models used

| Model | Purpose | Size |
|---|---|---|
| `facebook/sam2.1-hiera-large` | Dense auto-segmentation | ~900 MB |
| `openai/clip-vit-large-patch14` | Semantic classification per segment | ~900 MB |
| `depth-anything/Depth-Anything-V2-Small-hf` | Terrain roughness estimation | ~100 MB |
| `sklearn GradientBoostingClassifier` | Construction probability from features | < 1 MB |

All models are downloaded automatically from Hugging Face on first run.

---

## Data sources

- **PDOK Luchtfoto RGB** — `https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0` — 8 cm/px orthophotos 2022 & 2024
- **PDOK Luchtfoto CIR** — `https://service.pdok.nl/hwh/luchtfotocir/wms/v1_0` — colour-infrared for NDVI/NDBI
- **BT2022.gpkg** — Dutch Building Register construction-site polygons (EPSG:28992)
