#!/usr/bin/env python3
# +
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
from joblib import dump
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ==================================================
# LOGGING
# ==================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
log = logging.getLogger("builtup-ml")


# ==================================================
# PATHS
# ==================================================
PROJECT_ROOT = Path(".").resolve()
OUTPUT_ROOT = PROJECT_ROOT / "output"
MODELS_DIR = OUTPUT_ROOT / "model"
PREDICTION_DIR = OUTPUT_ROOT / "predictions"
SAMPLES_DIR = OUTPUT_ROOT / "samples"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# HELPERS
# ==================================================
def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def tile_feature_paths(
    tile_dir: Path,
    year: str,
    feature_files: Sequence[str],
) -> Optional[list[Path]]:

    idx_dir = tile_dir / year / "indices"
    base = tile_dir.name
    paths = []

    for tmpl in feature_files:
        p = idx_dir / tmpl.format(base=base)
        if not p.exists():
            log.warning("Missing feature: %s", p)
            return None
        paths.append(p)

    return paths


# ==================================================
# OPTIONAL INDEX FILTER
# ==================================================
def pre_ml_candidate_mask(
    X: np.ndarray,
    feature_files: Sequence[str],
    thresholds: dict,
) -> np.ndarray:

    if not thresholds:
        return np.ones(X.shape[0], dtype=bool)

    mask_ok = np.ones(X.shape[0], dtype=bool)

    for i, fname in enumerate(feature_files):
        for idx_name, (vmin, vmax) in thresholds.items():
            if idx_name in fname:
                mask_ok &= (X[:, i] >= vmin) & (X[:, i] <= vmax)

    return mask_ok


# ==================================================
# TRAINING EXTRACTION — POLYGONS
# ==================================================
def extract_training_polygons(
    tile_dir: Path,
    year: str,
    train_gdf: gpd.GeoDataFrame,
    class_col: str,
    feature_files: Sequence[str],
    max_samples: int,
    thresholds: dict,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

    paths = tile_feature_paths(tile_dir, year, feature_files)
    if paths is None:
        return None, None

    srcs = [rasterio.open(p) for p in paths]

    try:
        tile_box = box(*srcs[0].bounds)
        gdf = train_gdf.to_crs(srcs[0].crs)
        gdf = gdf[gdf.geometry.intersects(tile_box)]

        if gdf.empty:
            log.info("[%s] No intersecting training polygons", tile_dir.name)
            return None, None

        X_all, y_all = [], []

        for _, row in gdf.iterrows():
            bands = []

            for s in srcs:
                out, _ = mask(
                    s,
                    [row.geometry],
                    crop=True,
                    nodata=np.nan,
                    filled=True,
                )
                bands.append(out[0])

            stack = np.stack(bands, axis=0)
            resh = stack.reshape(stack.shape[0], -1).T

            valid = ~np.any(np.isnan(resh), axis=1)
            candidate = pre_ml_candidate_mask(resh, feature_files, thresholds)

            idx = np.where(valid & candidate)[0]
            if idx.size == 0:
                continue

            if idx.size > max_samples:
                idx = np.random.choice(idx, max_samples, replace=False)

            X_all.append(resh[idx])
            y_all.append(np.full(idx.size, int(row[class_col]), dtype=np.int32))

            log.info("[%s] Class %s → %d samples",
                     tile_dir.name, row[class_col], idx.size)

        if not X_all:
            return None, None

        return np.vstack(X_all), np.concatenate(y_all)

    finally:
        for s in srcs:
            s.close()


# ==================================================
# TRAINING EXTRACTION — POINTS
# ==================================================
def extract_training_points(
    tile_dir: Path,
    year: str,
    train_gdf: gpd.GeoDataFrame,
    class_col: str,
    feature_files: Sequence[str],
    thresholds: dict,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

    paths = tile_feature_paths(tile_dir, year, feature_files)
    if paths is None:
        return None, None

    srcs = [rasterio.open(p) for p in paths]

    try:
        gdf = train_gdf.to_crs(srcs[0].crs)
        tile_box = box(*srcs[0].bounds)
        gdf = gdf[gdf.geometry.within(tile_box)]

        if gdf.empty:
            log.info("[%s] No intersecting training points", tile_dir.name)
            return None, None

        coords = [(pt.x, pt.y) for pt in gdf.geometry]

        features = []
        for src in srcs:
            vals = np.array(list(src.sample(coords))).reshape(-1)
            features.append(vals)

        X = np.stack(features, axis=1)
        y = gdf[class_col].astype(int).values

        valid = (
            ~np.any(np.isnan(X), axis=1) &
            ~np.any(np.isinf(X), axis=1)
        )
        X, y = X[valid], y[valid]

        candidate = pre_ml_candidate_mask(X, feature_files, thresholds)
        X, y = X[candidate], y[candidate]

        log.info("[%s] Point samples → %d", tile_dir.name, len(y))
        return X, y

    finally:
        for s in srcs:
            s.close()


# ==================================================
# PREDICTION
# ==================================================
def predict_tile_rasters(
    tile_dir: Path,
    year: str,
    clf: RandomForestClassifier,
    feature_files: Sequence[str],
    prob_threshold: float,
):

    paths = tile_feature_paths(tile_dir, year, feature_files)
    if paths is None:
        return

    srcs = [rasterio.open(p) for p in paths]
    idx_mndwi = next(i for i, f in enumerate(feature_files) if "MNDWI" in f)

    try:
        profile = srcs[0].profile.copy()
        profile.update(dtype="float32", count=1, compress="DEFLATE")

        out_dir = PREDICTION_DIR / tile_dir.name / year
        out_dir.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_dir / "BUILTUP_PROB.tif", "w", **profile) as dp, \
             rasterio.open(out_dir / "BUILTUP_MASK.tif", "w",
                           **{**profile, "dtype": "uint8"}) as dm:

            for _, win in srcs[0].block_windows(1):
                bands = [s.read(1, window=win) for s in srcs]
                stack = np.stack(bands, axis=-1)
                resh = stack.reshape(-1, stack.shape[-1])

                probs = np.zeros(len(resh), dtype="float32")
                valid = ~np.any(np.isnan(resh), axis=1)

                if valid.any():
                    probs[valid] = clf.predict_proba(resh[valid])[:, 1]

                # Hard water mask
                probs[resh[:, idx_mndwi] > 0.15] = 0.0

                prob_img = probs.reshape(stack.shape[:2])
                mask_img = (prob_img >= prob_threshold).astype("uint8")

                dp.write(prob_img, 1, window=win)
                dm.write(mask_img, 1, window=win)

        log.info("[%s] Prediction completed", tile_dir.name)

    finally:
        for s in srcs:
            s.close()


# ==================================================
# MAIN PIPELINE
# ==================================================
def run(
    root: str,
    year: str,
    train_vector: str,
    class_col: str,
    feature_files: Sequence[str],
    tiles: Optional[Sequence[str]] = None,
    index_thresholds: Optional[dict] = None,
    max_samples_per_poly: int = 100,
    n_trees: int = 200,
    prob_threshold: float = 0.8,
    out_model: str = "output/model/builtup_rf.joblib",
    geometry_type: str = "polygon",
):

    root = resolve(root)
    train_vector = resolve(train_vector)
    index_thresholds = index_thresholds or {}

    train_gdf = gpd.read_file(train_vector)

    tile_dirs = [p for p in root.iterdir() if p.is_dir()]
    if tiles:
        tile_dirs = [t for t in tile_dirs if t.name in tiles]

    X_all, y_all = [], []

    for tile in tile_dirs:
        log.info("Extracting training → %s", tile.name)

        if geometry_type == "point":
            X, y = extract_training_points(
                tile, year, train_gdf, class_col,
                feature_files, index_thresholds
            )
        else:
            X, y = extract_training_polygons(
                tile, year, train_gdf, class_col,
                feature_files, max_samples_per_poly,
                index_thresholds
            )

        if X is not None:
            X_all.append(X)
            y_all.append(y)

    if not X_all:
        raise RuntimeError("No training samples extracted")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=n_trees,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(Xtr, ytr)

    log.info("\n%s", classification_report(yte, clf.predict(Xte)))
    dump(clf, resolve(out_model))

    for tile in tile_dirs:
        predict_tile_rasters(
            tile, year, clf, feature_files, prob_threshold
        )

    return clf
