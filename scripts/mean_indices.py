#!/usr/bin/env python3
"""
mean_indices.py

Compute per-pixel MEAN and MEDIAN maps per tile for:
  - NDVI   = (NIR - RED) / (NIR + RED)
  - NDBI   = (SWIR - NIR) / (SWIR + NIR)
  - BSI    = ((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))
  - MNDWI  = (GREEN - SWIR) / (GREEN + SWIR)

Usage:
  python scripts/mean_indices.py --root data/sentinel --overwrite False --use-scl True
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
import logging
import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("mean-indices")

# -----------------------------
# Index functions
# -----------------------------
def safe_div(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.true_divide(num, den)
        out[~np.isfinite(out)] = np.nan
    return out

def ndvi(nir, red): return safe_div(nir - red, nir + red)
def ndbi(swir, nir): return safe_div(swir - nir, swir + nir)
def bsi(swir, red, nir, blue): return safe_div((swir + red) - (nir + blue), (swir + red) + (nir + blue))
def mndwi(green, swir): return safe_div(green - swir, green + swir)

# -----------------------------
# Helpers
# -----------------------------
def find_tile_dirs(root: Path):
    return [p for p in sorted(root.iterdir()) if p.is_dir()]

def read_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        prof = src.profile.copy()
    return arr, prof

def resample_to_profile(src_arr, src_prof, dst_prof, resampling=Resampling.bilinear, threads=2):
    dst = np.empty((dst_prof["height"], dst_prof["width"]), dtype="float32")
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_prof["transform"],
        src_crs=src_prof.get("crs"),
        dst_transform=dst_prof["transform"],
        dst_crs=dst_prof.get("crs"),
        resampling=resampling,
        num_threads=threads,
    )
    return dst

def write_one(path: Path, arr: np.ndarray, profile: dict):
    profile = profile.copy()
    profile.update(dtype="float32", count=1, compress="DEFLATE")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)

# -----------------------------
# Per-tile processing
# -----------------------------
def process_tile(tile: Path, use_scl: bool, overwrite: bool, threads: int):
    log.info("Processing tile %s", tile.name)
    # find band files by suffix inside tile folder (10m under <tile>/10m, 20m under <tile>/20m)
    d10 = tile / "10m"
    d20 = tile / "20m"
    # look for typical filenames: <itemid>_B08.jp2 or <itemid>_B08.tif - pick any B08 to infer profile
    b08_files = list(d10.glob("*_B08.*"))
    if not b08_files:
        log.warning("No B08 found for tile %s — skipping", tile.name)
        return
    # template profile from first B08
    _, template_profile = read_band(str(b08_files[0]))
    target_prof = {"crs": template_profile.get("crs"),
                   "transform": template_profile.get("transform"),
                   "width": template_profile.get("width"),
                   "height": template_profile.get("height")}

    # lists to collect per-acquisition arrays for median stack
    ndvi_stack, ndbi_stack, bsi_stack, ndwi_stack = [], [], [], []
    # accumulators for mean
    h, w = target_prof["height"], target_prof["width"]
    sum_ndvi = np.zeros((h, w), dtype="float64"); count_ndvi = np.zeros((h, w), dtype="int32")
    sum_ndbi = np.zeros_like(sum_ndvi); count_ndbi = np.zeros_like(count_ndvi)
    sum_bsi  = np.zeros_like(sum_ndvi); count_bsi  = np.zeros_like(count_ndvi)
    sum_ndwi = np.zeros_like(sum_ndvi); count_ndwi = np.zeros_like(count_ndvi)

    # find all acquisitions by item id (files named <itemid>_B02 etc.). We will look for matching B02,B03,B04,B08,B11 or B12
    # Simple approach: scan files under d10 and d20
    # Build mapping itemid -> paths
    items = {}
    for p in list(d10.rglob("*")) + list(d20.rglob("*")):
        name = p.stem
        if "_" not in name:
            continue
        itemid, band = name.rsplit("_", 1)
        items.setdefault(itemid, {})[band.upper()] = str(p)

    processed = 0
    for itemid, bands in sorted(items.items()):
        # require 10m bands and SWIR
        if not all(k in bands for k in ("B08","B04","B02","B03")):
            continue
        swir = bands.get("B11") or bands.get("B12")
        if not swir:
            continue
        try:
            b08, p08 = read_band(bands["B08"])
            b04, p04 = read_band(bands["B04"])
            b02, p02 = read_band(bands["B02"])
            b03, p03 = read_band(bands["B03"])
            bswir, pswir = read_band(swir)

            # resample swir to 10m grid
            bswir_rs = resample_to_profile(bswir, pswir, target_prof, resampling=Resampling.bilinear, threads=threads)

            # if any 10m band not aligned, resample them too
            if (p08.get("width") != target_prof["width"] or p08.get("height") != target_prof["height"]
                or p08.get("transform") != target_prof["transform"]):
                b08 = resample_to_profile(b08, p08, target_prof, threads=threads)
            if (p04.get("width") != target_prof["width"] or p04.get("height") != target_prof["height"]
                or p04.get("transform") != target_prof["transform"]):
                b04 = resample_to_profile(b04, p04, target_prof, threads=threads)
            if (p02.get("width") != target_prof["width"] or p02.get("height") != target_prof["height"]
                or p02.get("transform") != target_prof["transform"]):
                b02 = resample_to_profile(b02, p02, target_prof, threads=threads)
            if (p03.get("width") != target_prof["width"] or p03.get("height") != target_prof["height"]
                or p03.get("transform") != target_prof["transform"]):
                b03 = resample_to_profile(b03, p03, target_prof, threads=threads)

            # optional SCL mask to remove clouds/shadows
            mask_valid = np.ones((h,w), dtype=bool)
            if use_scl and "SCL" in bands:
                scl_arr, pscl = read_band(bands["SCL"])
                if (pscl.get("width") != target_prof["width"] or pscl.get("height") != target_prof["height"]
                    or pscl.get("transform") != target_prof["transform"]):
                    scl_rs = resample_to_profile(scl_arr, pscl, target_prof, resampling=Resampling.nearest, threads=threads)
                else:
                    scl_rs = scl_arr
                # classes to exclude; 3 cloud shadow, 8-11 clouds/snow (common)
                invalid = np.isin(scl_rs.astype(np.int32), [3,8,9,10,11])
                mask_valid[invalid] = False

            # set invalid pixels to NaN
            for a in (b08, b04, b02, b03, bswir_rs):
                a[~mask_valid] = np.nan

            # compute indices
            arr_ndvi = ndvi(b08, b04)
            arr_ndbi = ndbi(bswir_rs, b08)
            arr_bsi  = bsi(bswir_rs, b04, b08, b02)
            arr_ndwi = mndwi(b03, bswir_rs)

            # accumulate means
            valid_ndvi = ~np.isnan(arr_ndvi)
            sum_ndvi[valid_ndvi] += arr_ndvi[valid_ndvi]; count_ndvi[valid_ndvi] += 1
            valid_ndbi = ~np.isnan(arr_ndbi)
            sum_ndbi[valid_ndbi] += arr_ndbi[valid_ndbi]; count_ndbi[valid_ndbi] += 1
            valid_bsi = ~np.isnan(arr_bsi)
            sum_bsi[valid_bsi] += arr_bsi[valid_bsi]; count_bsi[valid_bsi] += 1
            valid_ndwi = ~np.isnan(arr_ndwi)
            sum_ndwi[valid_ndwi] += arr_ndwi[valid_ndwi]; count_ndwi[valid_ndwi] += 1

            # append for median
            ndvi_stack.append(arr_ndvi); ndbi_stack.append(arr_ndbi)
            bsi_stack.append(arr_bsi); ndwi_stack.append(arr_ndwi)

            processed += 1
        except Exception as e:
            log.exception("Failed for item %s in %s: %s", itemid, tile.name, e)
            continue

    if processed == 0:
        log.info("No valid acquisitions for tile %s", tile.name)
        return

    # compute means
    mean_ndvi = np.full((h,w), np.nan, dtype="float32"); mean_ndbi = mean_ndvi.copy()
    mean_bsi = mean_ndvi.copy(); mean_ndwi = mean_ndvi.copy()
    mask = count_ndvi > 0; mean_ndvi[mask] = (sum_ndvi[mask] / count_ndvi[mask]).astype("float32")
    mask = count_ndbi > 0; mean_ndbi[mask] = (sum_ndbi[mask] / count_ndbi[mask]).astype("float32")
    mask = count_bsi > 0; mean_bsi[mask] = (sum_bsi[mask] / count_bsi[mask]).astype("float32")
    mask = count_ndwi > 0; mean_ndwi[mask] = (sum_ndwi[mask] / count_ndwi[mask]).astype("float32")

    # compute medians (stack may be memory heavy)
    ndvi_arr = np.stack(ndvi_stack, axis=0); median_ndvi = np.nanmedian(ndvi_arr, axis=0).astype("float32")
    ndbi_arr = np.stack(ndbi_stack, axis=0); median_ndbi = np.nanmedian(ndbi_arr, axis=0).astype("float32")
    bsi_arr  = np.stack(bsi_stack, axis=0);  median_bsi  = np.nanmedian(bsi_arr, axis=0).astype("float32")
    ndwi_arr = np.stack(ndwi_stack, axis=0); median_ndwi = np.nanmedian(ndwi_arr, axis=0).astype("float32")

    # write outputs into tile/indices/
    outdir = tile / "indices"; outdir.mkdir(parents=True, exist_ok=True)
    base = tile.name
    profile = template_profile.copy(); profile.update(count=1, dtype="float32", compress="DEFLATE")
    mapping_out = {
        f"{base}_MEAN_NDVI.tif": mean_ndvi,
        f"{base}_MEDIAN_NDVI.tif": median_ndvi,
        f"{base}_MEAN_NDBI.tif": mean_ndbi,
        f"{base}_MEDIAN_NDBI.tif": median_ndbi,
        f"{base}_MEAN_BSI.tif": mean_bsi,
        f"{base}_MEDIAN_BSI.tif": median_bsi,
        f"{base}_MEAN_NDWI.tif": mean_ndwi,
        f"{base}_MEDIAN_NDWI.tif": median_ndwi,
    }
    for fname, arr in mapping_out.items():
        outp = outdir / fname
        if outp.exists():
            log.info("Skipping existing %s", outp.name)
            continue
        write_one(outp, arr, profile)
        log.info("Wrote %s", outp.name)

# -----------------------------
# CLI and main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Compute mean & median indices per tile")
    p.add_argument("--root", default="data/sentinel", help="Root folder with tile subfolders")
    p.add_argument("--overwrite", default=False, action="store_true", help="Overwrite existing outputs")
    p.add_argument("--use-scl", default=True, action="store_true", help="Use SCL to mask clouds")
    p.add_argument("--threads", default=2, type=int, help="Threads for resampling")
    return p.parse_args()

def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        log.error("Root does not exist: %s", root)
        return
    tiles = find_tile_dirs(root)
    log.info("Found tiles: %d", len(tiles))
    for t in tiles:
        process_tile(t, use_scl=args.use_scl, overwrite=args.overwrite, threads=args.threads)

if __name__ == "__main__":
    main()
