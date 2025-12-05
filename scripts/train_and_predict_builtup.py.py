#!/usr/bin/env python3
"""
Simplified train_and_predict_builtup.py

- Reads per-tile index rasters (MEAN_* / MEDIAN_* files under each tile's 'indices' folder)
- Samples training data (points & polygons)
- Trains a RandomForest classifier
- Writes model to models/ and per-tile probability & mask rasters to tile indices folder

Edit the CONFIG section below to point to your directories and change parameters.
"""

import os
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from joblib import dump
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# CONFIG (edit these only)
# -------------------------
ROOT = Path("data/sentinel")                       # parent folder that contains tile subfolders
TRAIN_VECTOR = Path("data/training/CMDA_overall.shp")   # training shapefile (must have CLASS_COL)
CLASS_COL = "class"                               # column in training shapefile (1=built-up, 0=non-built-up)
OUT_MODEL = Path("models/builtup_rf_mean.joblib") # where to save trained model
TRAIN_SUMMARY = Path("models/training_summary.csv")
MAX_SAMPLES_PER_POLY = 100    # maximum pixels to sample per polygon
RANDOM_STATE = 42
N_TREES = 200
PROB_THRESHOLD = 0.7          # probability threshold to make binary built-up mask
# -------------------------

# Feature ordering: must match rasters you have under tile/indices/
FEATURE_FILES = [
    "{base}_MEAN_NDVI.tif",
    "{base}_MEAN_NDBI.tif",
    "{base}_MEAN_BSI.tif",
    "{base}_MEAN_NDWI.tif",
]
# Note: the code expects each tile folder to have an 'indices' subfolder with these files named like:
#  <TILE>_MEAN_NDVI.tif, <TILE>_MEAN_NDBI.tif, etc.

def find_tile_dirs(root: Path):
    """Return a list of tile folders in ROOT (folders only)."""
    return [p for p in sorted(root.iterdir()) if p.is_dir()]

def tile_feature_paths(tile_dir: Path):
    """Given a tile folder, return list of file paths for features in FEATURE_FILES.
       Returns None if any file missing."""
    idx_dir = tile_dir / "indices"
    base = tile_dir.name
    paths = []
    for fmt in FEATURE_FILES:
        p = idx_dir / fmt.format(base=base)
        if not p.exists():
            return None
        paths.append(str(p))
    return paths

def stack_rasters_open(paths):
    """Open rasters and return list of open rasterio datasets and a reference profile."""
    srcs = [rasterio.open(p) for p in paths]
    ref = srcs[0]
    # basic alignment check
    for s in srcs[1:]:
        if s.crs != ref.crs or s.width != ref.width or s.height != ref.height or s.transform != ref.transform:
            # if mismatch is expected in your workflow, you can resample here instead
            raise RuntimeError(f"Raster alignment mismatch in tile {ref.name}")
    return ref.profile, srcs

def sample_points_from_rasters(srcs, pts_gdf):
    """Sample raster values at point geometries. Returns X array (n_points, n_features)."""
    if pts_gdf.empty:
        return None
    coords = [(pt.x, pt.y) for pt in pts_gdf.geometry]
    vals_per_band = []
    for s in srcs:
        vals = np.array([v[0] for v in s.sample(coords)], dtype="float32")
        vals_per_band.append(vals)
    X = np.vstack(vals_per_band).T
    return X

def sample_polygons_from_rasters(srcs, polys_gdf, max_per_poly=100):
    """For each polygon, mask rasters and randomly sample up to max_per_poly valid pixels."""
    X_list = []
    y_list = []
    for _, row in polys_gdf.iterrows():
        geom = [row.geometry]
        bands = []
        valid_any = None
        for s in srcs:
            out, _ = mask(s, geom, crop=True, filled=True, nodata=np.nan)
            band = out[0]   # single band rasters
            bands.append(band)
            valid = ~np.isnan(band)
            valid_any = valid if valid_any is None else (valid_any | valid)
        if not np.any(valid_any):
            continue
        stacked = np.stack(bands, axis=0)              # (nbands, H, W)
        resh = stacked.reshape(stacked.shape[0], -1).T # (H*W, nbands)
        valid_pixels = ~np.any(np.isnan(resh), axis=1)
        idx = np.where(valid_pixels)[0]
        if idx.size == 0:
            continue
        if idx.size > max_per_poly:
            idx = np.random.choice(idx, max_per_poly, replace=False)
        sampled = resh[idx, :]
        X_list.append(sampled)
        y_list.append(np.full(sampled.shape[0], int(row[CLASS_COL]), dtype=np.int32))
    if not X_list:
        return None, None
    return np.vstack(X_list), np.concatenate(y_list)

def build_training_matrix(tile_dirs, train_gdf):
    """Loop tiles, find features, and extract samples from training geometries."""
    X_parts = []
    y_parts = []
    for tile in tile_dirs:
        feat_paths = tile_feature_paths(tile)
        if feat_paths is None:
            print(f"Skipping tile {tile.name}: missing index rasters")
            continue
        try:
            profile, srcs = stack_rasters_open(feat_paths)
        except Exception as e:
            print(f"Error opening rasters for {tile.name}: {e}")
            continue
        try:
            # subset training geometries to tile bounding box (speeds things up)
            tile_bounds = rasterio.open(feat_paths[0]).bounds
            tile_box = box(tile_bounds.left, tile_bounds.bottom, tile_bounds.right, tile_bounds.top)
            sub_gdf = train_gdf.to_crs(srcs[0].crs)
            inter = sub_gdf[sub_gdf.geometry.intersects(tile_box)]
            if inter.empty:
                continue
            pts = inter[inter.geometry.type == "Point"]
            polys = inter[inter.geometry.type.isin(["Polygon", "MultiPolygon"])]

            # sample points
            if not pts.empty:
                Xp = sample_points_from_rasters(srcs, pts)
                if Xp is not None:
                    valid = ~np.any(np.isnan(Xp), axis=1)
                    if valid.any():
                        X_parts.append(Xp[valid])
                        y_parts.append(pts.loc[valid, CLASS_COL].astype(int).values)

            # sample polygons
            if not polys.empty:
                Xpoly, ypoly = sample_polygons_from_rasters(srcs, polys, max_per_poly=MAX_SAMPLES_PER_POLY)
                if Xpoly is not None:
                    X_parts.append(Xpoly)
                    y_parts.append(ypoly)

        finally:
            for s in srcs:
                s.close()

    if not X_parts:
        raise RuntimeError("No training samples extracted. Check training vector and tile coverage.")
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    # drop any rows with NaNs
    valid = ~np.any(np.isnan(X), axis=1)
    return X[valid], y[valid]

def train_and_save_model(X, y, out_model_path: Path):
    """Train RandomForest and save model and basic training summary."""
    print("Training samples:", X.shape, "Class distribution:", np.bincount(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    clf = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Classification report (hold-out):")
    print(classification_report(y_test, y_pred))
    cv = cross_val_score(clf, X_train, y_train, cv=3, scoring="f1", n_jobs=-1)
    print("3-fold CV F1:", cv, "mean:", cv.mean())
    # ensure models dir exists
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(clf, str(out_model_path))
    print("Saved model to", out_model_path)
    return clf

def predict_tile(tile_dir: Path, clf):
    """Predict per tile using raster block windows (writes probability + mask into tile indices folder)."""
    feat_paths = tile_feature_paths(tile_dir)
    if feat_paths is None:
        print(f"Skipping predict for {tile_dir.name}: missing index rasters")
        return
    # open rasters (they are aligned)
    srcs = [rasterio.open(p) for p in feat_paths]
    try:
        profile = srcs[0].profile.copy()
        profile.update(dtype="float32", count=1, compress="DEFLATE")
        out_dir = tile_dir / "indices"
        out_dir.mkdir(parents=True, exist_ok=True)
        base = tile_dir.name
        out_prob = out_dir / f"{base}_BUILTUP_PROB_Mean.tif"
        out_mask = out_dir / f"{base}_BUILTUP_MASK_Mean.tif"

        h = srcs[0].height
        w = srcs[0].width

        # create blank outputs if not exist
        with rasterio.open(out_prob, "w", **profile) as dst:
            dst.write(np.zeros((1, h, w), dtype="float32"), 1)
        with rasterio.open(out_mask, "w", **{**profile, "dtype": "uint8"}) as dst:
            dst.write(np.zeros((1, h, w), dtype="uint8"), 1)

        # iterate windows (blocks) from first raster
        for ji, win in srcs[0].block_windows(1):
            bands_block = [s.read(1, window=win).astype("float32") for s in srcs]
            stacked = np.stack(bands_block, axis=-1)  # (Hwin, Wwin, nfeat)
            resh = stacked.reshape(-1, stacked.shape[-1])
            valid = ~np.any(np.isnan(resh), axis=1)
            probs = np.full((resh.shape[0],), np.nan, dtype="float32")
            labels = np.full((resh.shape[0],), np.nan, dtype="float32")
            if valid.any():
                prob_valid = clf.predict_proba(resh[valid])[:, 1]
                label_valid = (prob_valid >= PROB_THRESHOLD).astype("uint8")
                probs[valid] = prob_valid
                labels[valid] = label_valid
            probs_img = probs.reshape(stacked.shape[0], stacked.shape[1])
            labels_img = labels.reshape(stacked.shape[0], stacked.shape[1])
            # write back into outputs
            with rasterio.open(out_prob, "r+") as dp:
                dp.write(probs_img.astype("float32"), 1, window=win)
            with rasterio.open(out_mask, "r+") as dm:
                dm.write(np.nan_to_num(labels_img, nan=0).astype("uint8"), 1, window=win)
        print("Wrote predictions for", tile_dir.name)

    finally:
        for s in srcs:
            s.close()

def main():
    # load training vector
    if not TRAIN_VECTOR.exists():
        raise FileNotFoundError(f"Training shapefile not found: {TRAIN_VECTOR}")
    train_gdf = gpd.read_file(TRAIN_VECTOR)
    if CLASS_COL not in train_gdf.columns:
        raise RuntimeError(f"Training shapefile must contain '{CLASS_COL}' column with 0/1 labels")

    tile_dirs = find_tile_dirs(ROOT)
    print("Found tiles:", len(tile_dirs))

    # build training data
    X, y = build_training_matrix(tile_dirs, train_gdf)
    print("Built training matrix:", X.shape)

    # train and save model
    clf = train_and_save_model(X, y, OUT_MODEL)

    # predict per tile
    for t in tile_dirs:
        try:
            predict_tile(t, clf)
        except Exception as e:
            print("Prediction failed for", t.name, e)

    # save training summary
    summary = pd.DataFrame({
        "n_samples": [int(X.shape[0])],
        "class0": [int((y == 0).sum())],
        "class1": [int((y == 1).sum())],
    })
    TRAIN_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TRAIN_SUMMARY, index=False)
    print("Saved training summary to", TRAIN_SUMMARY)
    print("Done.")

if __name__ == "__main__":
    main()
