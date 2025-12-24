#!/usr/bin/env python3
# +
# #!/usr/bin/env python3
"""
Monthly → Tile-level Index Aggregation (Sentinel-2)

• Monthly indices computed per year
• Supports one or multiple years
• MEAN across multiple years
• MEAN + MEDIAN only for single year
• Year encoded in folder (not filenames)
• Notebook + CLI safe
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
import logging

import xarray as xr
import rioxarray as rxr
from dask.diagnostics import ProgressBar

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("mean-indices")

# --------------------------------------------------
# Constants
# --------------------------------------------------
MONTHS = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

# --------------------------------------------------
# Index math
# --------------------------------------------------
def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (a - b) / (a + b)
        r[~np.isfinite(r)] = np.nan
    return r

def NDVI(b08, b04): return safe_div(b08, b04)
def NDBI(b11, b08): return safe_div(b11, b08)
def BSI(b11, b04, b08, b02):
    return safe_div((b11 + b04) - (b08 + b02),
                    (b11 + b04) + (b08 + b02))
def MNDWI(b03, b11): return safe_div(b03, b11)
def SAVI(b08, b04, L=0.5): return (1 + L) * (b08 - b04) / (b08 + b04 + L)
def IBI(ndbi, savi, mndwi):
    return (ndbi - (savi + mndwi) / 2) / (ndbi + (savi + mndwi) / 2)

INDEX_FUNCS = {
    "NDVI": lambda b: NDVI(b["B08"], b["B04"]),
    "NDBI": lambda b: NDBI(b["B11"], b["B08"]),
    "BSI":  lambda b: BSI(b["B11"], b["B04"], b["B08"], b["B02"]),
    "MNDWI":lambda b: MNDWI(b["B03"], b["B11"]),
    "SAVI": lambda b: SAVI(b["B08"], b["B04"]),
    "IBI":  lambda b: IBI(b["NDBI"], b["SAVI"], b["MNDWI"]),
}

# --------------------------------------------------
# Raster helpers
# --------------------------------------------------
def read_band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype("float32"), src.profile

def resample(src, src_prof, dst_prof):
    dst = np.empty((dst_prof["height"], dst_prof["width"]), dtype="float32")
    reproject(
        src, dst,
        src_transform=src_prof["transform"],
        src_crs=src_prof["crs"],
        dst_transform=dst_prof["transform"],
        dst_crs=dst_prof["crs"],
        resampling=Resampling.bilinear,
    )
    return dst

def write(path, arr, profile):
    profile = profile.copy()
    profile.update(dtype="float32", count=1, compress="DEFLATE")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)

# --------------------------------------------------
# Monthly existence check
# --------------------------------------------------
def monthly_indices_exist(idx_dir, itemid, indices):
    return all((idx_dir / f"{itemid}_{i}.tif").exists() for i in indices)

# --------------------------------------------------
# STEP 1 — Monthly indices (per YEAR)
# --------------------------------------------------
def compute_monthly_indices(year_dir, indices, overwrite=False):

    for month in MONTHS:
        m10 = year_dir / month / "10m"
        m20 = year_dir / month / "20m"
        if not m10.exists():
            continue

        idx_dir = year_dir / month / "indices"
        idx_dir.mkdir(exist_ok=True)

        b08_files = list(m10.glob("*_B08.tif"))
        if not b08_files:
            continue

        itemid = b08_files[0].stem.replace("_B08", "")

        if not overwrite and monthly_indices_exist(idx_dir, itemid, indices):
            log.info("[%s %s] indices exist → skipped", year_dir.name, month)
            continue

        try:
            bands = {}
            bands["B08"], prof = read_band(m10 / f"{itemid}_B08.tif")
            bands["B04"], _ = read_band(m10 / f"{itemid}_B04.tif")
            bands["B03"], _ = read_band(m10 / f"{itemid}_B03.tif")
            bands["B02"], _ = read_band(m10 / f"{itemid}_B02.tif")

            b11, p11 = read_band(m20 / f"{itemid}_B11.tif")
            if p11["width"] != prof["width"]:
                b11 = resample(b11, p11, prof)
            bands["B11"] = b11

            # precompute shared
            bands["NDBI"] = NDBI(bands["B11"], bands["B08"])
            bands["SAVI"] = SAVI(bands["B08"], bands["B04"])
            bands["MNDWI"] = MNDWI(bands["B03"], bands["B11"])

            for idx in indices:
                arr = INDEX_FUNCS[idx](bands)
                write(idx_dir / f"{itemid}_{idx}.tif", arr, prof)

            log.info("[%s %s] monthly indices processed", year_dir.name, month)

        except Exception as e:
            log.warning("[%s %s] skipped: %s", year_dir.name, month, e)

# --------------------------------------------------
# STEP 2 — Aggregation helpers
# --------------------------------------------------
def aggregate_single_year(tile, year_dir, indices, do_mean, do_median):

    outdir = year_dir / "indices"
    outdir.mkdir(exist_ok=True)

    for idx in indices:
        rasters = sorted(year_dir.glob(f"*/indices/*_{idx}.tif"))
        if not rasters:
            continue

        da = xr.concat(
            [rxr.open_rasterio(r, chunks={"x":512,"y":512}).squeeze()
             for r in rasters],
            dim="time"
        )

        with ProgressBar():
            if do_mean:
                da.mean("time", skipna=True).rio.to_raster(
                    outdir / f"{tile}_MEAN_{idx}.tif", compress="DEFLATE"
                )
            if do_median:
                da.median("time", skipna=True).rio.to_raster(
                    outdir / f"{tile}_MEDIAN_{idx}.tif", compress="DEFLATE"
                )

        log.info("[%s %s] aggregated %s", tile, year_dir.name, idx)

def aggregate_multi_year(tile, year_dirs, indices):

    outdir = year_dirs[0].parent / "indices"
    outdir.mkdir(exist_ok=True)

    for idx in indices:
        rasters = []
        for yd in year_dirs:
            rasters.extend(sorted(yd.glob(f"*/indices/*_{idx}.tif")))

        if not rasters:
            continue

        da = xr.concat(
            [rxr.open_rasterio(r, chunks={"x":512,"y":512}).squeeze()
             for r in rasters],
            dim="time"
        )

        with ProgressBar():
            da.mean("time", skipna=True).rio.to_raster(
                outdir / f"{tile}_MEAN_{idx}.tif", compress="DEFLATE"
            )

        log.info("[%s] multi-year MEAN aggregated %s", tile, idx)

# --------------------------------------------------
# RUN (Notebook-friendly)
# --------------------------------------------------
def run(
    root="data/sentinel",
    tiles=None,
    years=(2025,),
    indices=("NDVI","NDBI","BSI","MNDWI","SAVI","IBI"),
    aggregate_mean=True,
    aggregate_median=False,
    overwrite_monthly=False,
):

    if aggregate_median and len(years) > 1:
        raise ValueError("Median aggregation allowed only for a single year.")

    root = Path(root)
    tile_dirs = [p for p in root.iterdir() if p.is_dir()]

    if tiles:
        tile_dirs = [t for t in tile_dirs if t.name in set(tiles)]

    for tile_dir in tile_dirs:
        tile = tile_dir.name
        year_dirs = [tile_dir / str(y) for y in years if (tile_dir / str(y)).exists()]

        if not year_dirs:
            log.warning("[%s] no valid year folders found", tile)
            continue

        for yd in year_dirs:
            compute_monthly_indices(yd, indices, overwrite_monthly)

        if len(year_dirs) == 1:
            aggregate_single_year(
                tile,
                year_dirs[0],
                indices,
                aggregate_mean,
                aggregate_median
            )
        else:
            if aggregate_mean:
                aggregate_multi_year(tile, year_dirs, indices)

# --------------------------------------------------
# CLI / Script entry
# --------------------------------------------------
if __name__ == "__main__":
    run()
# -


