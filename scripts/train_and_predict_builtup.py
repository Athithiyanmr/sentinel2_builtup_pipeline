#!/usr/bin/env python3
# +
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score

log = logging.getLogger("train-builtup")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -------------------------
# DEFAULTS
# -------------------------
PROJECT_ROOT = Path(".").resolve()

OUTPUT_ROOT = PROJECT_ROOT / "output"
MODELS_DIR = OUTPUT_ROOT / "model"
PREDICTION_DIR = OUTPUT_ROOT / "predictions"

DEFAULT_ROOT = "data/sentinel"
DEFAULT_TRAIN_VECTOR = "data/training/CMDA_overall.shp"
DEFAULT_CLASS_COL = "class"

# feature templates are formatted with {base} = tile name (e.g. T44PLV)
FEATURE_SET_TEMPLATES = {
    "mean": [
        "{base}_MEAN_NDVI.tif",
        "{base}_MEAN_NDBI.tif",
        "{base}_MEAN_BSI.tif",
        "{base}_MEAN_NDWI.tif",
    ],
    "median": [
        "{base}_MEDIAN_NDVI.tif",
        "{base}_MEDIAN_NDBI.tif",
        "{base}_MEDIAN_BSI.tif",
        "{base}_MEDIAN_NDWI.tif",
    ],
    "mean_median": [
        "{base}_MEAN_NDVI.tif",
        "{base}_MEAN_NDBI.tif",
        "{base}_MEAN_BSI.tif",
        "{base}_MEAN_NDWI.tif",
        "{base}_MEDIAN_NDVI.tif",
        "{base}_MEDIAN_NDBI.tif",
        "{base}_MEDIAN_BSI.tif",
        "{base}_MEDIAN_NDWI.tif",
    ],
}

DEFAULT_FEATURE_SET = "mean"
DEFAULT_MAX_SAMPLES_PER_POLY = 100
DEFAULT_N_TREES = 200
DEFAULT_RANDOM_STATE = 42
DEFAULT_PROB_THRESHOLD = 0.7

DEFAULT_OUT_MODEL = "output/model/builtup_rf_default.joblib"
DEFAULT_TRAIN_SUMMARY = "output/model/training_summary_default.csv"


# -------------------------
# Helpers
# -------------------------
def resolve_path(p: str | Path) -> Path:
    """Resolve path relative to project root."""
    p = Path(p)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def find_tile_dirs(root: Path) -> list[Path]:
    """Return a list of tile folders in ROOT (folders only)."""
    return [p for p in sorted(root.iterdir()) if p.is_dir()]


def tile_feature_paths(tile_dir: Path, feature_files: Sequence[str]) -> Optional[list[str]]:
    """Given a tile folder, return list of feature file paths.
       Returns None if any are missing."""
    idx_dir = tile_dir / "indices"
    base = tile_dir.name
    paths: list[str] = []
    for tmpl in feature_files:
        p = idx_dir / tmpl.format(base=base)
        if not p.exists():
            log.warning("Missing feature for tile %s: %s", tile_dir.name, p.name)
            return None
        paths.append(str(p))
    return paths


def stack_rasters_open(paths: Sequence[str]):
    """Open rasters and return list of open rasterio datasets and a reference profile."""
    srcs = [rasterio.open(p) for p in paths]
    ref = srcs[0]
    for s in srcs[1:]:
        if s.crs != ref.crs or s.width != ref.width or s.height != ref.height or s.transform != ref.transform:
            raise RuntimeError(f"Raster alignment mismatch in tile {ref.name}")
    return ref.profile, srcs


def sample_points_from_rasters(srcs, pts_gdf: gpd.GeoDataFrame) -> Optional[np.ndarray]:
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


def sample_polygons_from_rasters(
    srcs,
    polys_gdf: gpd.GeoDataFrame,
    class_col: str,
    max_per_poly: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """For each polygon, mask rasters and randomly sample up to max_per_poly valid pixels."""
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for _, row in polys_gdf.iterrows():
        geom = [row.geometry]
        bands = []
        valid_any = None

        for s in srcs:
            out, _ = mask(s, geom, crop=True, filled=True, nodata=np.nan)
            band = out[0]
            bands.append(band)
            valid = ~np.isnan(band)
            valid_any = valid if valid_any is None else (valid_any | valid)

        if valid_any is None or not np.any(valid_any):
            continue

        stacked = np.stack(bands, axis=0)          # (nbands, H, W)
        resh = stacked.reshape(stacked.shape[0], -1).T  # (H*W, nbands)
        valid_pixels = ~np.any(np.isnan(resh), axis=1)
        idx = np.where(valid_pixels)[0]
        if idx.size == 0:
            continue

        if idx.size > max_per_poly:
            idx = np.random.choice(idx, max_per_poly, replace=False)

        sampled = resh[idx, :]
        X_list.append(sampled)
        y_list.append(
            np.full(sampled.shape[0], int(row[class_col]), dtype=np.int32)
        )

    if not X_list:
        return None, None

    return np.vstack(X_list), np.concatenate(y_list)


def build_training_matrix(
    tile_dirs: Sequence[Path],
    train_gdf: gpd.GeoDataFrame,
    class_col: str,
    feature_files: Sequence[str],
    max_samples_per_poly: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Loop tiles, find features, and extract samples from training geometries."""
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for tile in tile_dirs:
        feat_paths = tile_feature_paths(tile, feature_files)
        if feat_paths is None:
            log.info("Skipping tile %s: missing index rasters", tile.name)
            continue

        try:
            profile, srcs = stack_rasters_open(feat_paths)
        except Exception as e:
            log.error("Error opening rasters for %s: %s", tile.name, e)
            continue

        try:
            # subset training geometries to tile bounding box (speed)
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
                        y_parts.append(pts.loc[valid, class_col].astype(int).values)

            # sample polygons
            if not polys.empty:
                Xpoly, ypoly = sample_polygons_from_rasters(
                    srcs, polys, class_col=class_col, max_per_poly=max_samples_per_poly
                )
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
    valid = ~np.any(np.isnan(X), axis=1)
    return X[valid], y[valid]


def train_and_save_model(
    X: np.ndarray,
    y: np.ndarray,
    out_model_path: Path,
    n_trees: int,
    random_state: int,
) -> Tuple[RandomForestClassifier, pd.DataFrame]:
    """Train RandomForest and save model and basic training summary."""
    log.info("Training samples: %s  Class distribution: %s", X.shape, np.bincount(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    clf = RandomForestClassifier(
        n_estimators=n_trees,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    log.info("Classification report (hold-out):\n%s", classification_report(y_test, y_pred))

    cv = cross_val_score(clf, X_train, y_train, cv=3, scoring="f1", n_jobs=-1)
    log.info("3-fold CV F1: %s (mean=%s)", cv, cv.mean())

    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(clf, str(out_model_path))
    log.info("Saved model to %s", out_model_path)

    summary_df = pd.DataFrame(
        {
            "n_samples": [int(X.shape[0])],
            "class0": [int((y == 0).sum())],
            "class1": [int((y == 1).sum())],
            "n_trees": [n_trees],
            "cv_f1_mean": [float(cv.mean())],
        }
    )
    return clf, summary_df


def predict_tile(
    tile_dir: Path,
    clf: RandomForestClassifier,
    feature_files: Sequence[str],
    prob_threshold: float,
    overwrite_predictions: bool,
    n_trees: int,
):
    """Predict per tile and write probability + mask rasters.

    Outputs go to: output/prediction/<tile_name>/
    """
    feat_paths = tile_feature_paths(tile_dir, feature_files)
    if feat_paths is None:
        log.info("Skipping predict for %s: missing index rasters", tile_dir.name)
        return

    srcs = [rasterio.open(p) for p in feat_paths]
    try:
        profile = srcs[0].profile.copy()
        profile.update(dtype="float32", count=1, compress="DEFLATE")

        # --- NEW OUTPUT LOCATION ---
        # e.g. output/prediction/T44PLV/
        tile_out_dir = PREDICTION_DIR / tile_dir.name
        tile_out_dir.mkdir(parents=True, exist_ok=True)

        base = tile_dir.name

        # suffix based on number of trees (e.g. _200.tif)
        tag = str(n_trees)
        out_prob = tile_out_dir / f"{base}_BUILTUP_PROB_{tag}.tif"
        out_mask = tile_out_dir / f"{base}_BUILTUP_MASK_{tag}.tif"

        if not overwrite_predictions and out_prob.exists() and out_mask.exists():
            log.info("Predictions already exist for %s — skipping.", tile_dir.name)
            return

        # iterate windows (blocks) from first raster
        with rasterio.open(out_prob, "w", **profile) as dp, \
             rasterio.open(out_mask, "w", **{**profile, "dtype": "uint8"}) as dm:

            for ji, win in srcs[0].block_windows(1):
                bands_block = [s.read(1, window=win).astype("float32") for s in srcs]
                stacked = np.stack(bands_block, axis=-1)  # (Hwin, Wwin, nfeat)
                resh = stacked.reshape(-1, stacked.shape[-1])

                valid = ~np.any(np.isnan(resh), axis=1)
                probs = np.full((resh.shape[0],), np.nan, dtype="float32")
                labels = np.zeros((resh.shape[0],), dtype="uint8")

                if valid.any():
                    prob_valid = clf.predict_proba(resh[valid])[:, 1]
                    label_valid = (prob_valid >= prob_threshold).astype("uint8")
                    probs[valid] = prob_valid
                    labels[valid] = label_valid

                probs_img = probs.reshape(stacked.shape[0], stacked.shape[1])
                labels_img = labels.reshape(stacked.shape[0], stacked.shape[1])

                dp.write(np.nan_to_num(probs_img, nan=0).astype("float32"), 1, window=win)
                dm.write(labels_img.astype("uint8"), 1, window=win)

        log.info("Wrote predictions for %s to %s", tile_dir.name, tile_out_dir)

    finally:
        for s in srcs:
            s.close()
# -------------------------
# Public API
# -------------------------
def run(
    root: str = DEFAULT_ROOT,
    train_vector: str = DEFAULT_TRAIN_VECTOR,
    class_col: str = DEFAULT_CLASS_COL,
    tiles: Optional[Sequence[str]] = None,
    feature_files: Optional[Sequence[str]] = None,
    feature_set: str = DEFAULT_FEATURE_SET,
    max_samples_per_poly: int = DEFAULT_MAX_SAMPLES_PER_POLY,
    n_trees: int = DEFAULT_N_TREES,
    random_state: int = DEFAULT_RANDOM_STATE,
    prob_threshold: float = DEFAULT_PROB_THRESHOLD,
    out_model: str = DEFAULT_OUT_MODEL,
    train_summary: str = DEFAULT_TRAIN_SUMMARY,
    overwrite_predictions: bool = False,
) -> Tuple[RandomForestClassifier, pd.DataFrame]:
    """
    Full pipeline: build training matrix, train RF, predict built-up rasters.

    Parameters
    ----------
    root : str
        Root folder with tile subfolders (each having indices/ with index rasters).
    train_vector : str
        Path to training shapefile/GeoPackage.
    class_col : str
        Column in training vector with class labels (0/1).
    tiles : list[str] | None
        Optional list of tile folder names to use for training & prediction.
    feature_files : list[str] | None
        List of filename templates (with {base}) for per-tile index rasters.
        If None, uses templates from `feature_set`.
    feature_set : str
        One of {"mean", "median", "mean_median"} – used only if feature_files is None.
    max_samples_per_poly : int
        Max pixels to sample per training polygon.
    n_trees : int
        Number of trees in RandomForest.
    random_state : int
        Random seed.
    prob_threshold : float
        Probability threshold for built-up mask (0–1).
    out_model : str
        Path to save trained model (.joblib).
    train_summary : str
        Path to save training summary CSV.
    overwrite_predictions : bool
        If True, recompute predictions even if outputs exist.

    Returns
    -------
    clf : RandomForestClassifier
    summary_df : pandas.DataFrame
    """
    root_path = resolve_path(root)
    train_path = resolve_path(train_vector)

    if not root_path.exists():
        raise FileNotFoundError(f"Root folder not found: {root_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Training vector not found: {train_path}")

    if feature_files is None:
        if feature_set not in FEATURE_SET_TEMPLATES:
            raise ValueError(f"Unknown feature_set '{feature_set}'. "
                             f"Use one of {list(FEATURE_SET_TEMPLATES.keys())}.")
        feature_files = FEATURE_SET_TEMPLATES[feature_set]

    log.info("Loading training vector: %s", train_path)
    train_gdf = gpd.read_file(train_path)
    if class_col not in train_gdf.columns:
        raise RuntimeError(f"Training shapefile must contain '{class_col}' column with 0/1 labels")

    tile_dirs = find_tile_dirs(root_path)
    if tiles is not None:
        tile_set = set(tiles)
        tile_dirs = [t for t in tile_dirs if t.name in tile_set]

    log.info("Using %d tiles for training/prediction", len(tile_dirs))
    if not tile_dirs:
        raise RuntimeError("No tile directories found for training/prediction.")

    # build training data
    X, y = build_training_matrix(
        tile_dirs,
        train_gdf,
        class_col=class_col,
        feature_files=feature_files,
        max_samples_per_poly=max_samples_per_poly,
    )
    log.info("Built training matrix: %s", X.shape)

    # train and save model
    out_model_path = resolve_path(out_model)
    clf, summary_df = train_and_save_model(
        X, y, out_model_path, n_trees=n_trees, random_state=random_state
    )

    # predict per tile
    for t in tile_dirs:
        try:
            predict_tile(
                t,
                clf,
                feature_files=feature_files,
                prob_threshold=prob_threshold,
                overwrite_predictions=overwrite_predictions,
                n_trees=n_trees,
            )
        except Exception as e:
            log.error("Prediction failed for %s: %s", t.name, e)

    # save training summary
    train_summary_path = resolve_path(train_summary)
    train_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(train_summary_path, index=False)
    log.info("Saved training summary to %s", train_summary_path)
    log.info("Done.")

    return clf, summary_df


# -------------------------
# CLI main (optional)
# -------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train RF and predict built-up per tile.")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--train-vector", default=DEFAULT_TRAIN_VECTOR)
    p.add_argument("--class-col", default=DEFAULT_CLASS_COL)
    p.add_argument("--tiles", nargs="*", help="Optional list of tile names to use")
    p.add_argument("--feature-set", default=DEFAULT_FEATURE_SET,
                   choices=list(FEATURE_SET_TEMPLATES.keys()))
    p.add_argument("--max-samples-per-poly", type=int, default=DEFAULT_MAX_SAMPLES_PER_POLY)
    p.add_argument("--n-trees", type=int, default=DEFAULT_N_TREES)
    p.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    p.add_argument("--prob-threshold", type=float, default=DEFAULT_PROB_THRESHOLD)
    p.add_argument("--out-model", default=DEFAULT_OUT_MODEL)
    p.add_argument("--train-summary", default=DEFAULT_TRAIN_SUMMARY)
    p.add_argument("--overwrite-predictions", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    import sys
    # ipykernel-safe: ignore Jupyter's -f arg
    if argv is None and "ipykernel" in sys.modules:
        args = parse_args([])
    else:
        args = parse_args(argv)

    run(
        root=args.root,
        train_vector=args.train_vector,
        class_col=args.class_col,
        tiles=args.tiles,
        feature_files=None,
        feature_set=args.feature_set,
        max_samples_per_poly=args.max_samples_per_poly,
        n_trees=args.n_trees,
        random_state=args.random_state,
        prob_threshold=args.prob_threshold,
        out_model=args.out_model,
        train_summary=args.train_summary,
        overwrite_predictions=args.overwrite_predictions,
    )


if __name__ == "__main__":
    main()
# -


