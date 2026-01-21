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
from pathlib import Path
import logging
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

PATCH = 256
STRIDE = 128
MIN_BUILT = 50

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("patch-maker")

def rasterize_labels(gdf, ref):
    return rasterize(
        [(g,1) for g in gdf.geometry],
        out_shape=(ref.height, ref.width),
        transform=ref.transform,
        fill=0,
        dtype="uint8"
    )

def process_tile(tile_dir, year, train_gdf, out_img, out_msk):

    stack_path = tile_dir/str(year)/"dl_stack"/"S2_DL_STACK.tif"
    if not stack_path.exists():
        log.warning("[%s] DL stack missing", tile_dir.name)
        return

    with rasterio.open(stack_path) as src:
        img = src.read().astype("float32")
        gdf = train_gdf.to_crs(src.crs)
        gdf = gdf[gdf.geometry.intersects(box(*src.bounds))]

        if gdf.empty:
            log.info("[%s] no polygons", tile_dir.name)
            return

        mask = rasterize_labels(gdf, src)

    C,H,W = img.shape
    pid = 0

    for i in range(0, H-PATCH, STRIDE):
        for j in range(0, W-PATCH, STRIDE):

            x = img[:, i:i+PATCH, j:j+PATCH]
            y = mask[i:i+PATCH, j:j+PATCH]

            if np.count_nonzero(y) < MIN_BUILT:
                continue

            np.save(out_img/f"{tile_dir.name}_{pid}_img.npy", x)
            np.save(out_msk/f"{tile_dir.name}_{pid}_mask.npy", y)
            pid += 1

    log.info("[%s] patches: %s", tile_dir.name, pid)


# ==================================================
# PUBLIC RUN FUNCTION
# ==================================================
def run(
    root="data/sentinel",
    train_vector="data/training/builtup.shp",
    out_dir="data/patches",
    tiles=None,
    year=2025,
):

    root = Path(root)
    train_vector = Path(train_vector)

    out_img = Path(out_dir)/"images"
    out_msk = Path(out_dir)/"masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)

    train_gdf = gpd.read_file(train_vector)

    tile_dirs = [p for p in root.iterdir() if p.is_dir()]
    if tiles:
        tiles = set(tiles)
        tile_dirs = [t for t in tile_dirs if t.name in tiles]

    for tile in tile_dirs:
        process_tile(tile, year, train_gdf, out_img, out_msk)


# ==================================================
# CLI SAFE
# ==================================================
if __name__ == "__main__":
    run()
