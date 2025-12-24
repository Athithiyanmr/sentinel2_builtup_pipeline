#!/usr/bin/env python3
# +
# #!/usr/bin/env python3
"""
Sentinel-2 monthly downloader with month-wise folders

✔ Tile/MGRS safe
✔ 12 month folders per tile
✔ Adaptive cloud expansion until image found
✔ Select least-cloud image per month
✔ Skip existing item-band files
✔ Per-tile CSV provenance
✔ Jupyter + CLI safe
"""

from __future__ import annotations
import os
import re
import csv
import time
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

import geopandas as gpd
from shapely.geometry import mapping
from pystac_client import Client
import planetary_computer as pc
import requests
import calendar

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEFAULT_OUTDIR = "data/sentinel"
DEFAULT_AOI = "data/aoi/CMDA.shp"
DEFAULT_YEAR = 2025

# progressive cloud expansion
CLOUD_STEPS = (5, 10, 20, 40, 60, 80)

BANDS_10M = ["B02", "B03", "B04", "B08", "TCI"]
BANDS_20M = ["B11", "SCL"]

MAX_WORKERS = 6
RETRY = 2
TIMEOUT = 90
CHUNK_SIZE = 1024 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("sentinel2-monthly")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def read_aoi(path: str):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    return mapping(gdf.unary_union)


def detect_tile_id(item) -> str:
    props = item.properties or {}
    tile = props.get("mgrs:tile")
    if tile:
        return tile
    text = f"{item.id} {props.get('title','')}".upper()
    m = re.search(r"T\d{2}[A-Z]{3}", text)
    if m:
        return m.group(0)
    raise RuntimeError(f"Cannot determine tile ID for {item.id}")


def sign(item):
    try:
        return pc.sign(item)
    except Exception:
        return item


def find_asset(item, band: str) -> Optional[str]:
    item = sign(item)
    for k, a in item.assets.items():
        if band.upper() in k.upper():
            return a.href
    return None


def download(url: str, outpath: Path):
    tmp = outpath.with_suffix(outpath.suffix + ".part")
    for attempt in range(RETRY + 1):
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for c in r.iter_content(chunk_size=CHUNK_SIZE):
                        if c:
                            f.write(c)
            os.replace(tmp, outpath)
            return True, None
        except Exception as e:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt >= RETRY:
                return False, str(e)
            time.sleep(1 + attempt)

# --------------------------------------------------
# MONTHLY SELECTION (EXPANDING CLOUD)
# --------------------------------------------------
def select_monthly_items(items):
    by_month = defaultdict(list)
    for it in items:
        dt = it.properties.get("datetime")
        if not dt:
            continue
        month = datetime.fromisoformat(dt.replace("Z", "")).month
        by_month[month].append(it)

    monthly_best = {}
    report = []

    for m in range(1, 13):
        chosen = None
        used_cloud = None

        for cmax in CLOUD_STEPS:
            candidates = [
                i for i in by_month.get(m, [])
                if i.properties.get("eo:cloud_cover", 100) <= cmax
            ]
            if candidates:
                chosen = min(
                    candidates,
                    key=lambda i: i.properties.get("eo:cloud_cover", 100)
                )
                used_cloud = cmax
                break

        if chosen:
            monthly_best[m] = chosen
            report.append({
                "month": calendar.month_name[m],
                "image_available": "Yes",
                "item_id": chosen.id,
                "datetime": chosen.properties.get("datetime"),
                "cloud_cover": chosen.properties.get("eo:cloud_cover"),
                "cloud_threshold": used_cloud,
            })
        else:
            report.append({
                "month": calendar.month_name[m],
                "image_available": "No",
                "item_id": None,
                "datetime": None,
                "cloud_cover": None,
                "cloud_threshold": None,
            })

    return monthly_best, report

# --------------------------------------------------
# DOWNLOAD TASK
# --------------------------------------------------
def download_task(item, band: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    if list(outdir.glob(f"{item.id}_{band}.*")):
        return item.id, band, "skipped_existing", None

    href = find_asset(item, band)
    if not href:
        return item.id, band, "missing_asset", None

    ext = os.path.splitext(href.split("?")[0])[1] or ".tif"
    outpath = outdir / f"{item.id}_{band}{ext}"

    ok, err = download(href, outpath)
    return item.id, band, "ok" if ok else "error", err

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def run(outdir=DEFAULT_OUTDIR, aoi_path=DEFAULT_AOI, year=DEFAULT_YEAR):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    aoi = read_aoi(aoi_path)
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    search = client.search(
        collections=["sentinel-2-l2a"],
        intersects=aoi,
        datetime=f"{year}-01-01/{year}-12-31",
    )

    items = list(search.get_items())
    log.info("Found %d Sentinel-2 items", len(items))

    items_by_tile = defaultdict(list)
    for it in items:
        items_by_tile[detect_tile_id(it)].append(it)

    for tile, tile_items in items_by_tile.items():
        tile_dir = outdir / tile / str(year)
        tile_dir.mkdir(parents=True, exist_ok=True)

        monthly_items, report = select_monthly_items(tile_items)

        # write tile summary
        with open(tile_dir / "tile_image_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "month", "image_available", "item_id",
                    "datetime", "cloud_cover", "cloud_threshold"
                ],
            )
            writer.writeheader()
            writer.writerows(report)

        # download per month
        for m, item in monthly_items.items():
            month_name = calendar.month_name[m]
            d10 = tile_dir / month_name / "10m"
            d20 = tile_dir / month_name / "20m"

            tasks = []
            for b in BANDS_10M:
                tasks.append((item, b, d10))
            for b in BANDS_20M:
                tasks.append((item, b, d20))

            with ThreadPoolExecutor(MAX_WORKERS) as ex:
                futures = [ex.submit(download_task, *t) for t in tasks]
                for f in as_completed(futures):
                    iid, band, status, msg = f.result()
                    if status == "ok":
                        log.info("[%s %s] downloaded %s", tile, month_name, band)

    log.info("Monthly download completed successfully.")

# --------------------------------------------------
# CLI
# --------------------------------------------------
def main(argv=None):
    if argv is None and "ipykernel" in sys.modules:
        run()
    else:
        p = argparse.ArgumentParser()
        p.add_argument("--outdir", default=DEFAULT_OUTDIR)
        p.add_argument("--aoi", default=DEFAULT_AOI)
        p.add_argument("--year", type=int, default=DEFAULT_YEAR)
        a = p.parse_args(argv)
        run(a.outdir, a.aoi, a.year)

if __name__ == "__main__":
    main()
# -


