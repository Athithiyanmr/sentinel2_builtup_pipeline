# 🛰️ Sentinel-2 Built-Up Area Detection Pipeline

> **An automated, modular ML pipeline for mapping built-up areas from Sentinel-2 satellite imagery — using spectral indices, Random Forest classification, and Microsoft Planetary Computer.**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Sentinel-2](https://img.shields.io/badge/Sentinel--2-003DA5?style=flat-square)](https://sentinel.esa.int)
[![Planetary Computer](https://img.shields.io/badge/Microsoft%20Planetary%20Computer-0078D4?style=flat-square&logo=microsoft&logoColor=white)](https://planetarycomputer.microsoft.com)
[![MIT License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 📌 What Is This?

Mapping built-up (urban) areas at scale is essential for urban planning, climate risk assessment, and land-use monitoring. Traditional methods require manual digitization or expensive commercial data.

This pipeline automates the full workflow — from satellite data acquisition to a classified built-up map — using free Sentinel-2 imagery and open-source ML tools. It is designed for **city-scale and regional-scale automation**, supporting both research and production use.

---

## 🔄 Pipeline Overview

```
1. Download Sentinel-2 tiles (Planetary Computer STAC API)
       ↓
2. Compute per-tile spectral index composites
       ↓
3. Build ML training dataset from training polygons / points
       ↓
4. Train Random Forest classifier
       ↓
5. Predict built-up probability & binary maps per tile
       ↓
6. Evaluate accuracy (F1, Precision, Recall, Confusion Matrix)
```

---

## ✨ Key Features

**📥 Automated Sentinel-2 Downloader**
- STAC API via Microsoft Planetary Computer
- Cloud filtering using scene metadata
- Downloads selected bands at 10m & 20m resolution
- Tile-aware search for large AOIs
- Resume-safe (skips already downloaded files)

**🛰️ Spectral Index Processing**
- Per-tile mean & median composites
- SCL cloud masking applied before aggregation
- Indices computed: NDVI, NDBI, BSI, NDWI, MNDWI

**🤖 Random Forest Classification**
- Supports both point and polygon training data
- Balanced sampling from polygon regions
- Outputs built-up probability raster + binary mask
- 3-fold cross-validation, F1, precision, recall reporting

**🗺️ Optional OSM Building Integration**
- Augment training with OpenStreetMap building footprints
- Strengthens urban class separation

---

## 🛰️ Spectral Indices Used

| Index | Measures | Formula |
|---|---|---|
| NDVI | Vegetation density | (NIR - Red) / (NIR + Red) |
| NDBI | Built-up surfaces | (SWIR - NIR) / (SWIR + NIR) |
| BSI | Bare soil | (SWIR + Red) - (NIR + Blue) / ... |
| NDWI | Water bodies | (Green - NIR) / (Green + NIR) |
| MNDWI | Modified water | (Green - SWIR) / (Green + SWIR) |

---

## 🗂️ Project Structure

```
sentinel2_builtup_pipeline/
│
├── scripts/
│   ├── download_s2_pc_by_tile.py      # Sentinel-2 downloader via STAC
│   ├── mean_indices.py                # Spectral index compositing
│   └── train_and_predict_builtup.py  # ML training & prediction
│
├── data/
│   ├── aoi/           # Area of interest shapefile
│   ├── training/      # Training polygons or points
│   ├── osm/           # Optional OSM building footprints
│   └── sentinel/      # Downloaded satellite tiles (auto-filled)
│
├── output/
│   ├── models/        # Saved Random Forest model (.joblib)
│   ├── prediction_tiles/  # Per-tile output rasters
│   └── logs/
│
├── run.ipynb          # Interactive notebook workflow
├── environment.yml
└── requirements.txt
```

---

## ⚙️ Setup

```bash
# Create environment
conda create --name s2builtup python=3.10
conda activate s2builtup
conda install jupyter nbconvert
conda install --file requirements.txt -c conda-forge
```

---

## ▶️ Usage

**Step 1 — Download Sentinel-2 tiles**
```bash
python scripts/download_s2_pc_by_tile.py \
  --outdir data/sentinel \
  --aoi data/aoi/your_aoi.shp \
  --year 2024 \
  --cloud 5 \
  --max-workers 6
```

**Step 2 — Compute spectral index composites**
```bash
python scripts/mean_indices.py
```

**Step 3 — Train classifier and generate predictions**
```bash
python scripts/train_and_predict_builtup.py
```

---

## 📤 Outputs

| File | Description |
|---|---|
| `*_MEAN.tif` | Per-pixel mean spectral index composite |
| `*_MEDIAN.tif` | Per-pixel median spectral index composite |
| `*_BUILTUP_PROB.tif` | Built-up probability (0–1) |
| `*_BUILTUP_MASK.tif` | Binary classification (built-up / non built-up) |
| `builtup_rf.joblib` | Saved Random Forest model |
| `training_summary.csv` | Accuracy metrics per run |

---

## 📈 Accuracy Metrics

The pipeline reports per-run:
- F1 Score
- Precision & Recall
- Confusion Matrix
- 3-fold Cross-Validation scores

---

## 🗺️ Roadmap

- [ ] UNet / DeepLab deep learning segmentation module
- [ ] Time-series built-up change detection
- [ ] Zonal statistics aggregation to admin boundaries
- [ ] Web map visualization with Leaflet or Kepler.gl

---

## 👤 Author

**Athithiyan M R** — Geospatial Data Scientist | Remote Sensing | Climate Analytics

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/athithiyan-m-r-/)
[![GitHub](https://img.shields.io/badge/GitHub-Athithiyanmr-181717?style=flat-square&logo=github)](https://github.com/Athithiyanmr)

---

## 🙏 Acknowledgements

- ESA Sentinel-2 Mission
- Microsoft Planetary Computer & STAC API
- OpenStreetMap contributors

---

## 📜 License

MIT License © 2026 Athithiyan M R
