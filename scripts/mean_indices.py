#!/usr/bin/env python3
# +
# #!/usr/bin/env python3
"""
Compute mean & median indices (NDVI, NDBI, BSI, NDWI) per Sentinel-2 tile.

Usage from terminal:
  python scripts/mean_indices.py --root data/sentinel --threads 4

Usage from notebook:
  from scripts.mean_indices import run as run_indices

  run_indices(
      root="data/sentinel",
      overwrite=False,
      use_scl=True,
      threads=4,
      tiles=["T44PLV"],
      outputs=["mean_ndvi", "mean_bsi", "median_ndwi"],
      compute_median=True,
  )
"""

from __future__ import annotations
import argparse
from pathlib import Path
import logging
import sys
from typing import Iterable, Optional, List, Set

import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("mean-indices")

# All possible output keys
ALL_OUTPUT_KEYS = [
    "mean_ndvi", "median_ndvi",
    "mean_ndbi", "median_ndbi",
    "mean_bsi",  "median_bsi",
    "mean_ndwi", "median_ndwi",
]

# -----------------------------
# Index functions
# -----------------------------
def safe_div(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.true_divide(num, den)
        out[~np.isfinite(out)] = np.nan
    return out

def ndvi(nir, red):
    return safe_div(nir - red, nir + red)

def ndbi(swir, nir):
    return safe_div(swir - nir, swir + nir)

def bsi(swir, red, nir, blue):
    return safe_div((swir + red) - (nir + blue), (swir + red) + (nir + blue))

def mndwi(green, swir):
    return safe_div(green - swir, green + swir)


# -----------------------------
# Helpers
# -----------------------------
def find_tile_dirs(root: Path) -> List[Path]:
    """Return all immediate subdirectories (tiles) under root."""
    return [p for p in sorted(root.iterdir()) if p.is_dir()]

def read_band(path: str):
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
def process_tile(
    tile: Path,
    use_scl: bool,
    overwrite: bool,
    threads: int,
    outputs: Set[str],
):
    """
    Process a single tile folder.

    Parameters
    ----------
    tile : Path
        Tile folder (contains 10m/ and 20m/).
    use_scl : bool
        Use SCL for masking.
    overwrite : bool
        Overwrite existing files.
    threads : int
        Threads used in rasterio.reproject.
    outputs : set[str]
        Subset of ALL_OUTPUT_KEYS to compute.
    """
    log.info("Processing tile %s", tile.name)

    # --- Fast skip: if all requested outputs already exist and overwrite=False ---
    outdir = tile / "indices"
    base = tile.name

    key_to_fname = {
        "mean_ndvi":   f"{base}_MEAN_NDVI.tif",
        "median_ndvi": f"{base}_MEDIAN_NDVI.tif",
        "mean_ndbi":   f"{base}_MEAN_NDBI.tif",
        "median_ndbi": f"{base}_MEDIAN_NDBI.tif",
        "mean_bsi":    f"{base}_MEAN_BSI.tif",
        "median_bsi":  f"{base}_MEDIAN_BSI.tif",
        "mean_ndwi":   f"{base}_MEAN_NDWI.tif",
        "median_ndwi": f"{base}_MEDIAN_NDWI.tif",
    }

    if not overwrite and outdir.exists():
        expected_paths = [outdir / key_to_fname[k] for k in outputs]
        if expected_paths and all(p.exists() for p in expected_paths):
            log.info("All requested outputs already exist for tile %s — skipping.", tile.name)
            return

    # flags: which indices we actually need (mean and/or median)
    need_mean_ndvi   = "mean_ndvi" in outputs
    need_median_ndvi = "median_ndvi" in outputs
    need_ndvi        = need_mean_ndvi or need_median_ndvi

    need_mean_ndbi   = "mean_ndbi" in outputs
    need_median_ndbi = "median_ndbi" in outputs
    need_ndbi        = need_mean_ndbi or need_median_ndbi

    need_mean_bsi    = "mean_bsi" in outputs
    need_median_bsi  = "median_bsi" in outputs
    need_bsi         = need_mean_bsi or need_median_bsi

    need_mean_ndwi   = "mean_ndwi" in outputs
    need_median_ndwi = "median_ndwi" in outputs
    need_ndwi        = need_mean_ndwi or need_median_ndwi

    # if nothing to do for this tile, exit quickly
    if not (need_ndvi or need_ndbi or need_bsi or need_ndwi):
        log.info("No outputs requested for tile %s — skipping.", tile.name)
        return

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
    target_prof = {
        "crs": template_profile.get("crs"),
        "transform": template_profile.get("transform"),
        "width": template_profile.get("width"),
        "height": template_profile.get("height"),
    }

    # stacks only if corresponding median is requested (to save memory)
    ndvi_stack = [] if need_median_ndvi else None
    ndbi_stack = [] if need_median_ndbi else None
    bsi_stack  = [] if need_median_bsi  else None
    ndwi_stack = [] if need_median_ndwi else None

    # accumulators for means only if needed
    h, w = target_prof["height"], target_prof["width"]

    def init_mean():
        return np.zeros((h, w), dtype="float64"), np.zeros((h, w), dtype="int32")

    sum_ndvi, count_ndvi = (init_mean() if need_mean_ndvi else (None, None))
    sum_ndbi, count_ndbi = (init_mean() if need_mean_ndbi else (None, None))
    sum_bsi,  count_bsi  = (init_mean() if need_mean_bsi  else (None, None))
    sum_ndwi, count_ndwi = (init_mean() if need_mean_ndwi else (None, None))

    # find all acquisitions by item id (files named <itemid>_B02 etc.) under 10m and 20m
    items = {}
    for p in list(d10.rglob("*")) + list(d20.rglob("*")):
        if not p.is_file():
            continue
        name = p.stem
        if "_" not in name:
            continue
        itemid, band = name.rsplit("_", 1)
        items.setdefault(itemid, {})[band.upper()] = str(p)

    processed = 0
    for itemid, bands in sorted(items.items()):
        # require 10m bands and SWIR
        if not all(k in bands for k in ("B08", "B04", "B02", "B03")):
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

            # resample SWIR to 10m grid
            bswir_rs = resample_to_profile(
                bswir, pswir, target_prof,
                resampling=Resampling.bilinear,
                threads=threads,
            )

            # if any 10m band not aligned, resample them too
            if (p08.get("width") != target_prof["width"]
                or p08.get("height") != target_prof["height"]
                or p08.get("transform") != target_prof["transform"]):
                b08 = resample_to_profile(b08, p08, target_prof, threads=threads)

            if (p04.get("width") != target_prof["width"]
                or p04.get("height") != target_prof["height"]
                or p04.get("transform") != target_prof["transform"]):
                b04 = resample_to_profile(b04, p04, target_prof, threads=threads)

            if (p02.get("width") != target_prof["width"]
                or p02.get("height") != target_prof["height"]
                or p02.get("transform") != target_prof["transform"]):
                b02 = resample_to_profile(b02, p02, target_prof, threads=threads)

            if (p03.get("width") != target_prof["width"]
                or p03.get("height") != target_prof["height"]
                or p03.get("transform") != target_prof["transform"]):
                b03 = resample_to_profile(b03, p03, target_prof, threads=threads)

            # optional SCL mask to remove clouds/shadows
            mask_valid = np.ones((h, w), dtype=bool)
            if use_scl and "SCL" in bands:
                scl_arr, pscl = read_band(bands["SCL"])
                if (pscl.get("width") != target_prof["width"]
                    or pscl.get("height") != target_prof["height"]
                    or pscl.get("transform") != target_prof["transform"]):
                    scl_rs = resample_to_profile(
                        scl_arr, pscl, target_prof,
                        resampling=Resampling.nearest,
                        threads=threads,
                    )
                else:
                    scl_rs = scl_arr

                # classes to exclude; 3 cloud shadow, 8-11 clouds/snow (common)
                invalid = np.isin(scl_rs.astype(np.int32), [3, 8, 9, 10, 11])
                mask_valid[invalid] = False

            # set invalid pixels to NaN
            for a in (b08, b04, b02, b03, bswir_rs):
                a[~mask_valid] = np.nan

            # compute indices only if needed
            arr_ndvi = ndvi(b08, b04) if (need_ndvi or need_mean_ndvi or need_median_ndvi) else None
            arr_ndbi = ndbi(bswir_rs, b08) if (need_ndbi or need_mean_ndbi or need_median_ndbi) else None
            arr_bsi  = bsi(bswir_rs, b04, b08, b02) if (need_bsi or need_mean_bsi or need_median_bsi) else None
            arr_ndwi = mndwi(b03, bswir_rs) if (need_ndwi or need_mean_ndwi or need_median_ndwi) else None

            # accumulate means
            if need_mean_ndvi and arr_ndvi is not None:
                valid = ~np.isnan(arr_ndvi)
                sum_ndvi[valid] += arr_ndvi[valid]
                count_ndvi[valid] += 1

            if need_mean_ndbi and arr_ndbi is not None:
                valid = ~np.isnan(arr_ndbi)
                sum_ndbi[valid] += arr_ndbi[valid]
                count_ndbi[valid] += 1

            if need_mean_bsi and arr_bsi is not None:
                valid = ~np.isnan(arr_bsi)
                sum_bsi[valid] += arr_bsi[valid]
                count_bsi[valid] += 1

            if need_mean_ndwi and arr_ndwi is not None:
                valid = ~np.isnan(arr_ndwi)
                sum_ndwi[valid] += arr_ndwi[valid]
                count_ndwi[valid] += 1

            # collect for medians (only if requested)
            if need_median_ndvi and arr_ndvi is not None:
                ndvi_stack.append(arr_ndvi)
            if need_median_ndbi and arr_ndbi is not None:
                ndbi_stack.append(arr_ndbi)
            if need_median_bsi and arr_bsi is not None:
                bsi_stack.append(arr_bsi)
            if need_median_ndwi and arr_ndwi is not None:
                ndwi_stack.append(arr_ndwi)

            processed += 1
        except Exception as e:
            log.exception("Failed for item %s in %s: %s", itemid, tile.name, e)
            continue

    if processed == 0:
        log.info("No valid acquisitions for tile %s", tile.name)
        return

    outdir.mkdir(parents=True, exist_ok=True)
    profile = template_profile.copy()
    profile.update(count=1, dtype="float32", compress="DEFLATE")

    mapping_out = {}

    # means
    if need_mean_ndvi:
        mean_ndvi = np.full((h, w), np.nan, dtype="float32")
        mask = count_ndvi > 0
        mean_ndvi[mask] = (sum_ndvi[mask] / count_ndvi[mask]).astype("float32")
        mapping_out["mean_ndvi"] = mean_ndvi

    if need_mean_ndbi:
        mean_ndbi = np.full((h, w), np.nan, dtype="float32")
        mask = count_ndbi > 0
        mean_ndbi[mask] = (sum_ndbi[mask] / count_ndbi[mask]).astype("float32")
        mapping_out["mean_ndbi"] = mean_ndbi

    if need_mean_bsi:
        mean_bsi = np.full((h, w), np.nan, dtype="float32")
        mask = count_bsi > 0
        mean_bsi[mask] = (sum_bsi[mask] / count_bsi[mask]).astype("float32")
        mapping_out["mean_bsi"] = mean_bsi

    if need_mean_ndwi:
        mean_ndwi = np.full((h, w), np.nan, dtype="float32")
        mask = count_ndwi > 0
        mean_ndwi[mask] = (sum_ndwi[mask] / count_ndwi[mask]).astype("float32")
        mapping_out["mean_ndwi"] = mean_ndwi

    # medians (only stack & compute if requested to save memory)
    if need_median_ndvi and ndvi_stack:
        ndvi_arr = np.stack(ndvi_stack, axis=0)
        median_ndvi = np.nanmedian(ndvi_arr, axis=0).astype("float32")
        mapping_out["median_ndvi"] = median_ndvi

    if need_median_ndbi and ndbi_stack:
        ndbi_arr = np.stack(ndbi_stack, axis=0)
        median_ndbi = np.nanmedian(ndbi_arr, axis=0).astype("float32")
        mapping_out["median_ndbi"] = median_ndbi

    if need_median_bsi and bsi_stack:
        bsi_arr = np.stack(bsi_stack, axis=0)
        median_bsi = np.nanmedian(bsi_arr, axis=0).astype("float32")
        mapping_out["median_bsi"] = median_bsi

    if need_median_ndwi and ndwi_stack:
        ndwi_arr = np.stack(ndwi_stack, axis=0)
        median_ndwi = np.nanmedian(ndwi_arr, axis=0).astype("float32")
        mapping_out["median_ndwi"] = median_ndwi

    # write only requested outputs
    for key, arr in mapping_out.items():
        if key not in outputs:
            continue
        fname = key_to_fname[key]
        outp = outdir / fname
        if not overwrite and outp.exists():
            log.info("Skipping existing %s", outp.name)
            continue
        write_one(outp, arr, profile)
        log.info("Wrote %s", outp.name)


# -----------------------------
# Public API for notebooks
# -----------------------------
def normalise_outputs(outputs: Optional[Iterable[str]], compute_median: bool) -> Set[str]:
    """Validate and normalize output keys."""
    if outputs is None:
        out_list = list(ALL_OUTPUT_KEYS)
    else:
        out_list = [o.lower() for o in outputs]

    # drop medians if compute_median=False
    if not compute_median:
        out_list = [o for o in out_list if not o.startswith("median_")]

    # validate
    invalid = [o for o in out_list if o not in ALL_OUTPUT_KEYS]
    if invalid:
        raise ValueError(f"Unknown outputs: {invalid}. "
                         f"Allowed: {ALL_OUTPUT_KEYS}")

    return set(out_list)


def run(
    root: str = "data/sentinel",
    overwrite: bool = False,
    use_scl: bool = True,
    threads: int = 2,
    tiles: Optional[List[str]] = None,
    outputs: Optional[Iterable[str]] = None,
    compute_median: bool = True,
):
    """
    Run mean/median index computation over tiles under `root`.

    Parameters
    ----------
    root : str
        Root folder with tile subfolders (each having 10m/20m).
    overwrite : bool
        If True, recompute and overwrite existing index rasters.
        If False, tiles with all requested outputs already present are skipped.
    use_scl : bool
        Whether to use SCL band to mask clouds/shadows.
    threads : int
        Threads for rasterio.reproject.
    tiles : list[str] | None
        Optional list of tile names (folder names) to process;
        if None, all tiles under root are processed.
    outputs : iterable[str] | None
        Subset of outputs to compute, e.g.
        ["mean_ndvi", "mean_bsi", "median_ndwi"].
        If None, all 8 outputs are computed (subject to compute_median).
    compute_median : bool
        If False, all median_* outputs are ignored even if listed in `outputs`.
    """
    root_path = Path(root)
    if not root_path.exists():
        log.error("Root does not exist: %s", root_path)
        return

    outputs_set = normalise_outputs(outputs, compute_median)

    all_tiles = find_tile_dirs(root_path)
    if tiles is not None:
        tile_set = set(tiles)
        all_tiles = [t for t in all_tiles if t.name in tile_set]

    log.info("Found tiles: %d", len(all_tiles))
    for t in all_tiles:
        process_tile(
            t,
            use_scl=use_scl,
            overwrite=overwrite,
            threads=threads,
            outputs=outputs_set,
        )


# -----------------------------
# CLI and main (optional)
# -----------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compute mean & median indices per tile")
    p.add_argument("--root", default="data/sentinel", help="Root folder with tile subfolders")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--no-scl", dest="use_scl", action="store_false", help="Do NOT use SCL to mask clouds")
    p.add_argument("--threads", default=2, type=int, help="Threads for resampling")
    p.add_argument("--tiles", nargs="*", help="Optional list of tile names to process")
    p.add_argument(
        "--outputs",
        nargs="*",
        help="Subset of outputs to compute, e.g. mean_ndvi mean_bsi median_ndwi"
    )
    p.add_argument(
        "--no-median",
        dest="compute_median",
        action="store_false",
        help="Disable all median_* computations"
    )
    p.set_defaults(use_scl=True, compute_median=True)
    return p.parse_args(argv)

def main(argv=None):
    # ipykernel-safe: ignore Jupyter's -f arg if run from notebook
    if argv is None and "ipykernel" in sys.modules:
        args = parse_args([])
    else:
        args = parse_args(argv)

    run(
        root=args.root,
        overwrite=args.overwrite,
        use_scl=args.use_scl,
        threads=args.threads,
        tiles=args.tiles,
        outputs=args.outputs,
        compute_median=args.compute_median,
    )

if __name__ == "__main__":
    main()
# -


