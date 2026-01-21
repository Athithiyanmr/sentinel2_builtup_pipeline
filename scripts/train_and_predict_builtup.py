#!/usr/bin/env python3
# +
# #!/usr/bin/env python3
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ==================================================
# LOGGING
# ==================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("builtup-ml")

# ==================================================
# PATHS
# ==================================================
PROJECT_ROOT = Path(".").resolve()
OUTPUT_ROOT = PROJECT_ROOT / "output"
MODELS_DIR = OUTPUT_ROOT / "model"
PREDICTION_DIR = OUTPUT_ROOT / "predictions"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

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
# INDEX-BASED CANDIDATE FILTER (TRAINING)
# ==================================================
def pre_ml_candidate_mask(
    X: np.ndarray,
    feature_files: Sequence[str],
    thresholds: dict,
) -> np.ndarray:

    mask_ok = np.ones(X.shape[0], dtype=bool)

    for i, fname in enumerate(feature_files):
        for index_name, (vmin, vmax) in thresholds.items():
            if index_name in fname:
                mask_ok &= (X[:, i] >= vmin) & (X[:, i] <= vmax)

    return mask_ok


# ==================================================
# TRAINING SAMPLE EXTRACTION
# ==================================================
def extract_training(
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
        bounds = srcs[0].bounds
        tile_box = box(*bounds)

        gdf = train_gdf.to_crs(srcs[0].crs)
        gdf = gdf[gdf.geometry.intersects(tile_box)]
        if gdf.empty:
            return None, None

        X_all, y_all = [], []

        for _, row in gdf.iterrows():

            bands = []
            for s in srcs:
                out, _ = mask(
                    s,
                    [row.geometry],
                    crop=True,
                    filled=True,
                    nodata=np.nan,
                )
                bands.append(out[0])

            stack = np.stack(bands, axis=0)
            resh = stack.reshape(stack.shape[0], -1).T

            valid = ~np.any(np.isnan(resh), axis=1)
            candidate = pre_ml_candidate_mask(
                resh, feature_files, thresholds
            )

            idx = np.where(valid & candidate)[0]
            if idx.size == 0:
                continue

            if idx.size > max_samples:
                idx = np.random.choice(idx, max_samples, replace=False)

            X_all.append(resh[idx])
            y_all.append(
                np.full(idx.size, int(row[class_col]), dtype=np.int32)
            )

        if not X_all:
            return None, None

        return np.vstack(X_all), np.concatenate(y_all)

    finally:
        for s in srcs:
            s.close()


# ==================================================
# PREDICTION (WITH WATER MASKING)
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

    # Identify index positions
    idx_mndwi = next(
        i for i, f in enumerate(feature_files) if "MNDWI" in f
    )

    try:
        profile = srcs[0].profile.copy()
        profile.update(dtype="float32", count=1, compress="DEFLATE")

        out_dir = PREDICTION_DIR / tile_dir.name / year
        out_dir.mkdir(parents=True, exist_ok=True)

        prob_path = out_dir / "BUILTUP_PROB.tif"
        mask_path = out_dir / "BUILTUP_MASK.tif"

        with rasterio.open(prob_path, "w", **profile) as dp, \
             rasterio.open(mask_path, "w", **{**profile, "dtype": "uint8"}) as dm:

            for _, win in srcs[0].block_windows(1):

                bands = [s.read(1, window=win).astype("float32") for s in srcs]
                stack = np.stack(bands, axis=-1)
                resh = stack.reshape(-1, stack.shape[-1])

                invalid = (
                    np.any(np.isnan(resh), axis=1) |
                    np.any(np.isinf(resh), axis=1)
                )

                probs = np.zeros(len(resh), dtype="float32")

                valid = ~invalid
                if valid.any():
                    probs[valid] = clf.predict_proba(resh[valid])[:, 1]

                # -----------------------------
                # HARD WATER MASK (CRITICAL FIX)
                # -----------------------------
                mndwi = resh[:, idx_mndwi]
                water = mndwi > 0.15
                probs[water] = 0.0

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
):

    index_thresholds = index_thresholds or {}

    root = resolve(root)
    train_vector = resolve(train_vector)

    train_gdf = gpd.read_file(train_vector)

    tile_dirs = [p for p in root.iterdir() if p.is_dir()]
    if tiles:
        tile_dirs = [t for t in tile_dirs if t.name in tiles]

    X_all, y_all = [], []

    for tile in tile_dirs:
        X, y = extract_training(
            tile,
            year,
            train_gdf,
            class_col,
            feature_files,
            max_samples_per_poly,
            index_thresholds,
        )
        if X is not None:
            X_all.append(X)
            y_all.append(y)

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    log.info("Training samples: %s | Class balance: %s", X.shape, np.bincount(y))

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

    out_model = resolve(out_model)
    dump(clf, out_model)
    log.info("Model saved → %s", out_model)

    for tile in tile_dirs:
        log.info("Predicting %s (%s)", tile.name, year)
        predict_tile_rasters(
            tile, year, clf, feature_files, prob_threshold
        )

    return clf
# -


