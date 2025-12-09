#!/usr/bin/env python3
# +
# #!/usr/bin/env python3
"""
Jupyter-friendly Sentinel-2 downloader for Microsoft Planetary Computer STAC.

Usage:
  # Preferred in a terminal
  python scripts/download_s2_pc_by_tile.py --outdir data/sentinel --aoi data/aoi/CMDA.shp --year 2025

  # In Jupyter: call run(...) directly from a cell (example below).
"""

from __future__ import annotations
import os
import re
import time
import csv
import logging
import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Tuple, Dict

import geopandas as gpd
from shapely.geometry import mapping
from pystac_client import Client
import planetary_computer as pc
import requests

# -------------------------
# CONFIG (edit these only)
# -------------------------
DEFAULT_OUTDIR = "data/sentinel"         # where tiles will be saved
DEFAULT_AOI = "data/aoi/CMDA.shp"        # default AOI in repo
DEFAULT_YEAR = 2025
DEFAULT_CLOUD_MAX = 5.0                  # percent; raise if too strict
DEFAULT_MAX_WORKERS = 6
DEFAULT_RETRY = 2
DEFAULT_TIMEOUT = 90
BANDS_10M = ["B02", "B03", "B04", "B08", "TCI"]         # common 10m bands + TCI
BANDS_20M = ["B11", "SCL"]                             # typical SWIR + SCL (add B12 if needed)
KEEP_TILE_DIRS = True
CHUNK_SIZE = 1024 * 1024
# -------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("simple-pc-downloader")


# -------------------------
# Helper functions
# -------------------------
def read_aoi(aoi_path: str):
    """Read AOI shapefile/geojson and return GeoDataFrame and geometry mapping."""
    gdf = gpd.read_file(aoi_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    geom = gdf.unary_union
    return gdf, mapping(geom)


def detect_tile_id(item) -> str:
    """Extract sentinel tile id from item properties or try a regex fallback."""
    props = getattr(item, "properties", {}) or {}
    tile = props.get("sentinel:tile_id") or props.get("sentinel:tileid") or props.get("sentinel:tileId")
    if tile:
        return tile
    combined = (str(item.id or "") + " " + str(props.get("title") or "")).upper()
    m = re.search(r"T\d{2}[A-Z]{3}", combined)
    return m.group(0) if m else "UNKNOWN_TILE"


def prepare_tile_dirs(outdir: Path, tile: str) -> Tuple[Path, Path]:
    """Create per-tile folders and return 10m and 20m folder paths."""
    tile_root = outdir / tile
    d10 = tile_root / "10m"
    d20 = tile_root / "20m"
    d10.mkdir(parents=True, exist_ok=True)
    d20.mkdir(parents=True, exist_ok=True)
    return d10, d20


def _sign_item(item):
    """Sign the STAC item for Planetary Computer. If signing fails, return the raw item."""
    try:
        return pc.sign(item)
    except Exception:
        return item


def find_asset_href(item, band_key: str) -> Optional[str]:
    """
    Try to find a matching asset href for a requested band key.
    Tries direct asset key, case-insensitive match, substring, and filename match.
    """
    signed = _sign_item(item)
    assets = getattr(signed, "assets", {}) or {}

    # direct key
    if band_key in assets:
        return assets[band_key].href

    bk = band_key.upper()
    # case-insensitive key match
    for k, a in assets.items():
        if k.upper() == bk:
            return a.href

    # synonyms for TCI / visual
    if bk in ("TCI", "VISUAL", "TRUECOLOR"):
        for k, a in assets.items():
            if any(x in k.upper() for x in ("VIS", "TCI", "TRUE")):
                return a.href

    # substring match in asset key
    for k, a in assets.items():
        if bk in k.upper() or k.upper() in bk:
            return a.href

    # try filename in href
    for k, a in assets.items():
        href = str(a.href or "")
        if bk in href.upper():
            return a.href

    return None


def download_to_path(url: str, outpath: Path, timeout: int, retry: int) -> Tuple[bool, Optional[str]]:
    """Download url to outpath with small retry/backoff. Uses .part temp file when writing."""
    tmp = outpath.with_suffix(outpath.suffix + ".part")
    attempt = 0
    while attempt <= retry:
        attempt += 1
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            fh.write(chunk)
            os.replace(str(tmp), str(outpath))
            return True, None
        except Exception as e:
            logger.debug("Attempt %d failed for %s: %s", attempt, url, e)
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            if attempt > retry:
                return False, str(e)
            time.sleep(1 + attempt)
    return False, "unknown error"


def download_task(item, band_key: str, out_folder: Path, timeout: int, retry: int) -> Tuple[str, str, str, Optional[str]]:
    """Wrapper to download one asset for one item into out_folder."""
    out_folder.mkdir(parents=True, exist_ok=True)
    href = find_asset_href(item, band_key)
    if not href:
        return (item.id, band_key, "missing_asset", None)

    # keep original extension if possible (jp2/tif)
    href_base = href.split("?")[0]
    ext = os.path.splitext(href_base)[1] or ".tif"
    outname = f"{item.id}_{band_key}{ext}"
    outpath = out_folder / outname

    if outpath.exists():
        return (item.id, band_key, "skipped", None)

    ok, err = download_to_path(href, outpath, timeout=timeout, retry=retry)
    status = "ok" if ok else "error"
    return (item.id, band_key, status, err)


# -------------------------
# Main flow (simple)
# -------------------------
def run(
    outdir: str = DEFAULT_OUTDIR,
    aoi_path: str = DEFAULT_AOI,
    year: int = DEFAULT_YEAR,
    cloud_max: float = DEFAULT_CLOUD_MAX,
    bands10: List[str] = BANDS_10M,
    bands20: List[str] = BANDS_20M,
    max_workers: int = DEFAULT_MAX_WORKERS,
    retry_count: int = DEFAULT_RETRY,
    timeout: int = DEFAULT_TIMEOUT,
):
    """
    Main downloader function. Call this directly from a notebook or from CLI.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # read AOI and setup STAC client
    gdf, aoi = read_aoi(aoi_path)
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    logger.info("Searching sentinel-2-l2a for year %s (cloud <%s%%)", year, cloud_max)
    search = client.search(
        collections=["sentinel-2-l2a"],
        intersects=aoi,
        datetime=f"{year}-01-01/{year}-12-31",
        query={"eo:cloud_cover": {"lt": cloud_max}},
    )

    items = list(search.get_items())
    logger.info("Found %d items", len(items))
    if not items:
        logger.warning("No items found; check AOI/year/cloud filter.")
        return

    # group by tile
    items_by_tile: Dict[str, List] = {}
    for item in items:
        tile = detect_tile_id(item)
        items_by_tile.setdefault(tile, []).append(item)

    tiles = sorted(items_by_tile.keys())
    logger.info("Tiles detected: %s", tiles)

    # save tile list
    tile_list_path = outdir / "tile_list.txt"
    tile_list_path.write_text("\n".join(tiles))
    logger.info("Wrote tile list: %s", tile_list_path)

    # build tasks
    tasks = []
    for tile, item_list in items_by_tile.items():
        d10, d20 = prepare_tile_dirs(outdir, tile)
        for item in item_list:
            for b in bands10:
                tasks.append((item, b, d10))
            for b in bands20:
                tasks.append((item, b, d20))

    logger.info("Total download tasks: %d", len(tasks))

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(download_task, itm, band, fld, timeout, retry_count) for itm, band, fld in tasks]
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                logger.exception("Task error: %s", e)
                continue
            results.append(res)
            iid, band, status, msg = res
            if status == "ok":
                logger.info("Downloaded: %s %s", iid, band)
            elif status == "skipped":
                logger.debug("Skipped: %s %s", iid, band)
            elif status == "missing_asset":
                logger.warning("Missing asset: %s %s", iid, band)
            else:
                logger.error("Failed: %s %s -> %s", iid, band, msg)

    # summary to CSV
    csv_path = outdir / "download_summary.csv"
    with open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["item_id", "band", "status", "message"])
        for iid, band, status, msg in results:
            w.writerow([iid, band, status, msg or ""])
    logger.info("Wrote download summary to %s", csv_path)

    # done
    logger.info("Download finished. OK=%d skipped=%d missing=%d errors=%d",
                sum(1 for r in results if r[2] == "ok"),
                sum(1 for r in results if r[2] == "skipped"),
                sum(1 for r in results if r[2] == "missing_asset"),
                sum(1 for r in results if r[2] == "error"),
               )


# -------------------------
# CLI wrapper (safe for Jupyter)
# -------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Simple Sentinel-2 downloader (Planetary Computer).")
    p.add_argument("--outdir", "-o", default=DEFAULT_OUTDIR, help="Output root folder (default: data/sentinel)")
    p.add_argument("--aoi", "-a", default=DEFAULT_AOI, help="AOI path (default: data/aoi/CMDA.shp)")
    p.add_argument("--year", "-y", type=int, default=DEFAULT_YEAR, help="Year to search")
    p.add_argument("--cloud", "-c", type=float, default=DEFAULT_CLOUD_MAX, help="Max cloud cover percent")
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Download threads")
    p.add_argument("--retry", type=int, default=DEFAULT_RETRY, help="Retries per file")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout (s)")
    return p.parse_args(argv)


def main(argv=None):
    """
    Entry point that is Jupyter-safe:
    - If run from terminal, argv is None (argparse uses sys.argv)
    - If imported/used in notebook, pass a list or call main(['--outdir', '...'])
    """
    # Detect if running inside ipykernel and when no explicit argv passed, avoid using Jupyter kernel args
    if argv is None and "ipykernel" in sys.modules:
        # run with defaults (safe) — you can still call run(...) manually below in notebook with custom args
        args = parse_args([])
    else:
        args = parse_args(argv)

    run(
        outdir=args.outdir,
        aoi_path=args.aoi,
        year=args.year,
        cloud_max=args.cloud,
        bands10=BANDS_10M,
        bands20=BANDS_20M,
        max_workers=args.max_workers,
        retry_count=args.retry,
        timeout=args.timeout,
    )


    
if __name__ == "__main__":
    main()