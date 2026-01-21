# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# #!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
import numpy as np
import logging

# ======================================================
# CONFIG
# ======================================================
RAW_BANDS = ["B02", "B03", "B04", "B08", "B11"]  # Blue, Green, Red, NIR, SWIR
BLOCK = 1024   # Safe for Mac memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("dl-stack")

# ======================================================
# HELPERS
# ======================================================
def collect_band_files(tile_dir: Path, year: int, band: str, month: str | None):
    """
    Returns list of band files.
    If month is provided → only that month.
    Else → all months (year composite).
    """
    if month:
        base = tile_dir / str(year) / month
        if band == "B11":
            return list(base.glob(f"20m/*_{band}.tif"))
        else:
            return list(base.glob(f"10m/*_{band}.tif"))
    else:
        if band == "B11":
            return list(tile_dir.joinpath(str(year)).glob(f"*/20m/*_{band}.tif"))
        else:
            return list(tile_dir.joinpath(str(year)).glob(f"*/10m/*_{band}.tif"))


# ======================================================
# CORE
# ======================================================
def build_stack(tile_dir: Path, year: int, month: str | None = None):

    if month:
        ref_dir = tile_dir / str(year) / month / "10m"
        tag = month
    else:
        ref_dir = tile_dir / str(year) / "January" / "10m"
        tag = "MEAN"

    ref_list = list(ref_dir.glob("*_B02.tif"))
    if not ref_list:
        log.warning("[%s] Reference B02 not found", tile_dir.name)
        return

    ref_path = ref_list[0]

    outdir = tile_dir / str(year) / "dl_stack"
    outdir.mkdir(exist_ok=True)
    outpath = outdir / f"S2_DL_STACK_{year}_{tag}.tif"

    with rasterio.open(ref_path) as ref:

        profile = ref.profile
        profile.update(
            count=len(RAW_BANDS),
            dtype="float32",
            compress="DEFLATE",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            nodata=np.nan
        )

        # ----------------------------------------------
        # Open all sources
        # ----------------------------------------------
        srcs = {}
        for band in RAW_BANDS:
            files = collect_band_files(tile_dir, year, band, month)
            if not files:
                raise RuntimeError(f"[{tile_dir.name}] Missing {band} ({year}, {month})")
            srcs[band] = [rasterio.open(f) for f in files]

        # ----------------------------------------------
        # Block-wise stacking
        # ----------------------------------------------
        with rasterio.open(outpath, "w", **profile) as dst:

            for row in range(0, ref.height, BLOCK):
                for col in range(0, ref.width, BLOCK):

                    h = min(BLOCK, ref.height - row)
                    w = min(BLOCK, ref.width - col)
                    win = Window(col, row, w, h)

                    out_block = []

                    for band in RAW_BANDS:

                        acc = np.zeros((h, w), dtype="float32")
                        cnt = np.zeros((h, w), dtype="uint16")

                        # ---------- Single image (month mode) ----------
                        if len(srcs[band]) == 1:
                            src = srcs[band][0]
                            reproject(
                                rasterio.band(src, 1),
                                acc,
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=ref.window_transform(win),
                                dst_crs=ref.crs,
                                resampling=Resampling.bilinear,
                            )
                            out = acc

                        # ---------- Multi-image mean (year mode) ----------
                        else:
                            for src in srcs[band]:
                                tmp = np.zeros((h, w), dtype="float32")

                                reproject(
                                    rasterio.band(src, 1),
                                    tmp,
                                    src_transform=src.transform,
                                    src_crs=src.crs,
                                    dst_transform=ref.window_transform(win),
                                    dst_crs=ref.crs,
                                    resampling=Resampling.bilinear,
                                )

                                valid = np.isfinite(tmp)
                                acc[valid] += tmp[valid]
                                cnt[valid] += 1

                            out = np.full_like(acc, np.nan, dtype="float32")
                            mask = cnt > 0
                            out[mask] = acc[mask] / cnt[mask]

                        out_block.append(out)

                    stack = np.stack(out_block)
                    dst.write(stack, window=win)

        # ----------------------------------------------
        # Close all files
        # ----------------------------------------------
        for band in srcs:
            for s in srcs[band]:
                s.close()

    log.info("[%s] DL stack built → %s", tile_dir.name, outpath.name)


# ======================================================
# RUNNER
# ======================================================
def run(root="data/sentinel", tiles=None, year=2025, month=None):

    root = Path(root)
    tile_dirs = [p for p in root.iterdir() if p.is_dir()]

    if tiles:
        tiles = set(tiles)
        tile_dirs = [t for t in tile_dirs if t.name in tiles]

    for tile in tile_dirs:
        build_stack(tile, year, month)


if __name__ == "__main__":
    run()

# %%
