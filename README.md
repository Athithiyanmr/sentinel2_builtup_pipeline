 # Sentinel-2 Built-up Area Detection Pipeline  
*A complete automated workflow for downloading, preprocessing, and classifying built-up areas using Sentinel-2 L2A imagery, spectral indices, Random Forest, and optional OSM building footprints.*

---

## 📌 Overview

This repository contains a fully automated pipeline for generating built-up area maps using **Sentinel-2 L2A satellite data**.  
The workflow includes:

1. **Download** all Sentinel-2 tiles intersecting an AOI from **Microsoft Planetary Computer**  
2. **Compute spectral indices** (NDVI, NDBI, BSI, NDWI) as **per-tile mean & median composites**  
3. **Train a Random Forest model** using:
   - Sentinel-2 spectral index stacks  
   - Optional **OSM building footprints** as built-up training data  
4. **Predict built-up probability & classify** for each tile  
5. Save outputs as per-tile rasters (`*_BUILTUP_PROB.tif`, `*_BUILTUP_MASK.tif`)

The pipeline is modular, reproducible, and suitable for large AOIs such as cities or districts.

---

## 🚀 Features

- Fully automated Sentinel-2 tile downloader  
- Cloud masking using SCL layer  
- Per-pixel **mean & median** composite generation  
- Computation of key spectral indices:
  - NDVI  
  - NDBI  
  - BSI  
  - NDWI (MNDWI variant)  
- Model training using Random Forest  
- Optional integration of **OSM building polygons** as training inputs  
- Per-tile built-up probability & binary classification rasters  
- GitHub-ready, modular folder structure  

---

## 📂 Project Structure
