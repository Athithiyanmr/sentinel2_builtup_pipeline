🌍 Sentinel-2 Built-up Area Detection Pipeline

A modular, reproducible workflow for mapping built-up areas using Sentinel-2 L2A imagery, spectral indices, Random Forest ML models, and optional OSM building footprints.

⸻

⭐ Overview

This repository provides a fully automated geospatial pipeline that:
	1.	Downloads Sentinel-2 tiles intersecting an AOI (via Microsoft Planetary Computer STAC API)
	2.	Computes per-tile mean & median spectral indices
	3.	Builds a machine-learning training dataset using polygons or points
	4.	Trains a Random Forest classifier
	5.	Predicts built-up probability & binary built-up maps
	6.	Organizes outputs cleanly by tile

It is designed for city-scale and regional-scale automation, supporting both research and production use.

⸻

🚀 Key Features

📥 1. Automated Sentinel-2 Downloader
	•	Uses Planetary Computer STAC API
	•	Downloads selected bands at 10m & 20m resolution
	•	Cloud filtering using metadata
	•	Handles tile grouping & file naming
	•	Resume-safe (skips existing files)

🛰️ 2. Spectral Index Processing

Generates per-tile mean & median composites for:
	•	NDVI (veg)
	•	NDBI (built-up)
	•	BSI (bare soil)
	•	NDWI / MNDWI (water)

Uses SCL cloud masking.

🤖 3. Machine Learning Classification
	•	Random Forest classifier
	•	Point and polygon training supported
	•	Automatic extraction of raster features
	•	Balanced sampling from polygons
	•	Built-up probability + binary masks

🗺️ 4. Optional OSM Building Training

You can add OSM building footprints to strengthen built-up training classes.

🧱 5. Modular Scripts

scripts/
  ├── download_s2_pc_by_tile.py
  ├── mean_indices.py
  └── train_and_predict_builtup.py

  📁 Recommended Project Structure
  sentinel2_builtup_pipeline/
│
├── scripts/
│   ├── download_s2_pc_by_tile.py
│   ├── mean_indices.py
│   └── train_and_predict_builtup.py
│
├── data/
│   ├── aoi/
│   │   ├── CMDA.shp  (Your AOI)
│   │   └── ...
│   ├── training/
│   │   ├── CMDA_overall.shp  (Your training polygons/points)
│   │   └── ...
│   ├── osm/ (optional)
│   │   ├── osm_buildings.shp
│   │   └── ...
│   └── sentinel/      (will be filled after download)
│       └── .gitkeep
│
├── output/
│   ├── models/
│   ├── prediction_tiles/
│   └── logs/
│
├── README.md
├── requirements.txt
└── .gitignore

⚙️ Installation
1. Install Dependencies
   
	conda create --name xyz python==3.10
	
	conda activate xyz
	
	conda install jupyter nbconvert
	
	conda install --file requirements.txt -c conda-forge

   
3. Prepare Input Data
   
✔ Place AOI under:

data/aoi/CMDA.shp

✔ Place training dataset under:
data/training/CMDA_overall.shp

▶️ Usage
Step 1 — Download Sentinel-2 tiles
python scripts/download_s2_pc_by_tile.py \
    --outdir data/sentinel \
    --aoi data/aoi/CMDA.shp \
    --year 2025 \
    --cloud 5 \
    --max-workers 6
 python scripts/mean_indices.py
 python scripts/train_and_predict_builtup.py

📤 Outputs

For each tile:
File                                           Description
*_MEAN_*.tif                        Per-pixel multi-acquisition mean indices
*_MEDIAN_*.tif                      Per-pixel median indices
*_BUILTUP_PROB.tif                  Built-up probability
*_BUILTUP_MASK.tif                  Binary classification

Model file:
output/models/builtup_rf.joblib
Training summary:
training_summary.csv

📈 Accuracy & Evaluation

The Random Forest classifier provides:
	•	F1 score
	•	Precision, recall
	•	Confusion matrix
	•	Cross-validation (3-fold)

You can expand the training dataset at any time to improve results.

🏗️ Roadmap:
	•	Add UNet / DeepLab deep-learning segmentation
	•	Time-series built-up change detection
	•	Zonal statistics for admin boundaries
	•	Web-map visualization with Leaflet or Kepler.gl

 🤝 Credits

This project was developed with assistance from ChatGPT 5.1 Flagship (OpenAI).
Satellite data provided via Microsoft Planetary Computer.

📜 License

MIT License



