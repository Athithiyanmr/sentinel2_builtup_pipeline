#!/usr/bin/env python3
from pathlib import Path
import logging

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
import osmnx as ox

# ==================================================
# CONFIG
# ==================================================
PATCH = 256
STRIDE = 128
MIN_BUILDING_PIXELS = 20
MONTH = "October"   # October

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("osm-patch-maker")

# ==================================================
# FETCH OSM BUILDINGS
# ==================================================
def fetch_osm_buildings(bounds, crs):
    minx, miny, maxx, maxy = bounds

    gdf = ox.features_from_bbox(
        north=maxy, south=miny,
        east=maxx, west=minx,
        tags={"building": True}
    )

    if gdf.empty:
        return gdf

    return gdf.to_crs(crs)

# ==================================================
# RASTERIZE LABELS
# ==================================================
def rasterize_labels(gdf, ref):
    return rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=(ref.height, ref.width),
        transform=ref.transform,
        fill=0,
        dtype="uint8"
    )

# ==================================================
# PROCESS SINGLE TILE
# ==================================================
def process_tile(tile_dir, year, out_img, out_msk):

    stack_path = (
        tile_dir / str(year) / MONTH / "dl_stack" / "S2_DL_STACK.tif"
    )

    if not stack_path.exists():
        log.warning("[%s] stack missing", tile_dir.name)
        return

    with rasterio.open(stack_path) as src:
        img = src.read().astype("float32")

        buildings = fetch_osm_buildings(src.bounds, src.crs)
        if buildings.empty:
            log.info("[%s] no OSM buildings", tile_dir.name)
            return

        mask = rasterize_labels(buildings, src)

    _, H, W = img.shape
    pid = 0

    for i in range(0, H - PATCH, STRIDE):
        for j in range(0, W - PATCH, STRIDE):

            x = img[:, i:i+PATCH, j:j+PATCH]
            y = mask[i:i+PATCH, j:j+PATCH]

            if np.count_nonzero(y) < MIN_BUILDING_PIXELS:
                continue

            np.save(out_img / f"{tile_dir.name}_{pid}_img.npy", x)
            np.save(out_msk / f"{tile_dir.name}_{pid}_mask.npy", y)

            pid += 1

    log.info("[%s] patches created: %d", tile_dir.name, pid)

# ==================================================
# PUBLIC RUN FUNCTION
# ==================================================
def run(
    root="data/sentinel",
    out_dir="data/patches_osm",
    tiles=None,
    MONTH = None,
    year=2025
):

    root = Path(root)
    out_img = Path(out_dir) / "images"
    out_msk = Path(out_dir) / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)

    tile_dirs = [p for p in root.iterdir() if p.is_dir()]

    if tiles:
        tiles = [t.lower() for t in tiles]
        tile_dirs = [t for t in tile_dirs if t.name.lower() in tiles]

    for tile in tile_dirs:
        process_tile(tile, year, out_img, out_msk)

# ==================================================
# CLI SAFE
# ==================================================
if __name__ == "__main__":
    run(
        tiles=["T44PLU"],
        MONTH= "October",
        year=2025
    )