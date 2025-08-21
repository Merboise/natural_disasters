#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download GHSL tiles (JRC GHSL, R2023A) by product/epoch/resolution, and zip them.

Features
- Product selectable (default: GHS_POP)
- Single epoch or epoch range with step (e.g., 1975..2020 step 5)
- Resolutions/CRS: 4326_3ss, 4326_30ss, 54009_100, 54009_1km
- Auto-discovers available tiles by scraping the JRC 'tiles/' directory
- Parallel downloads with resume + retries
- Zip per-epoch or a single zip of everything

Docs:
- GHSL download page (tiles available): https://human-settlement.emergency.copernicus.eu/download.php
- GHS-POP R2023A (WGS84 grids at 3″/30″ derived from 100 m Mollweide): https://data.europa.eu/89h/2ff68a52-5b5b-4a22-8f40-c41da8332cfe
"""

import argparse
import concurrent.futures as cf
import fnmatch
import os
import re
import sys
import time
from pathlib import Path
import urllib.robotparser
from urllib.parse import urljoin, urlparse
import requests

# -------- Defaults / constants --------
BASE_FTP = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
PACKAGE = "GHS_POP_GLOBE_R2023A"  # change if needed (e.g., other products use same pattern)
DEFAULT_PRODUCT = "GHS_POP"

# Map short user inputs to directory tokens in GHSL
RES_MAP = {
    "4326_3ss": "4326_3ss",
    "4326_30ss": "4326_30ss",
    "54009_100": "54009_100",   # 100 m Mollweide
    "54009_1km": "54009_1km",   # 1 km Mollweide
}
VERSION = "V1-0"  # as seen in R2023A tile structure
TIMEOUT = 60
RETRIES = 4
WORKERS = min(12, (os.cpu_count() or 4) * 2)

HEADERS = {"User-Agent": "GHSL-downloader/1.0 (+github.com/Merboise/natural_disasters)"}

# -------- Helpers --------
def can_scrape(href_url: str, user_agent: str = "*") -> bool:
    """
    Checks robots.txt for permissions to fetch the given URL.
    """
    parsed = urlparse(href_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    robots_url = base + "/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception as e:
        print(f"[WARN] Could not read robots.txt: {e}")
        return False  # err on safe side
    return rp.can_fetch(user_agent, href_url)

def build_epoch_dir(product: str, epoch: int, res_key: str) -> str:
    """Return the directory path (URL) for the tiles listing for one epoch/res."""
    res_token = RES_MAP[res_key]
    dataset = f"{product}_E{epoch}_GLOBE_R2023A_{res_token}"
    return f"{BASE_FTP}/{PACKAGE}/{dataset}/{VERSION}/tiles/"

def list_tiles(session: requests.Session, tiles_url: str) -> list[str]:
    """
    Fetch the tiles directory HTML and parse all .zip tile filenames.
    We assume Apache-style autoindex with hrefs containing .zip files.
    """
    r = session.get(tiles_url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    # crude parse: extract href="...zip"
    zips = re.findall(r'href="([^"]+\.zip)"', r.text, flags=re.IGNORECASE)
    # normalize: some pages list filenames only, others include full paths
    clean = []
    for z in zips:
        # keep only file-like names (filter out parent links)
        name = z.split("/")[-1]
        if name.lower().endswith(".zip"):
            clean.append(name)
    return sorted(set(clean))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def download_file(session: requests.Session, url: str, dest: Path) -> tuple[str, bool]:
    """
    Download with resume support. Returns (filename, success).
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = HEADERS.copy()
    mode = "wb"

    if tmp.exists():
        existing = tmp.stat().st_size
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    for attempt in range(1, RETRIES + 1):
        try:
            with session.get(url, headers=headers, stream=True, timeout=TIMEOUT) as r:
                if r.status_code == 416:
                    # complete already
                    tmp.rename(dest)
                    return (dest.name, True)
                r.raise_for_status()
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.rename(dest)
            return (dest.name, True)
        except Exception as e:
            if attempt == RETRIES:
                return (dest.name, False)
            time.sleep(2 * attempt)

def zip_output(zip_path: Path, files: list[Path]):
    import zipfile
    ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            # store relative to the zip root
            z.write(f, arcname=f.relative_to(zip_path.parent))

def parse_epoch_list(args) -> list[int]:
    if args.epochs:
        return sorted(set(int(e) for e in args.epochs))
    if args.epoch is not None:
        return [int(args.epoch)]
    if args.epoch_start is not None and args.epoch_end is not None:
        step = int(args.epoch_step or 5)
        return list(range(int(args.epoch_start), int(args.epoch_end) + 1, step))
    # sensible default
    return [2000]

def main():
    ap = argparse.ArgumentParser(description="Download GHSL tiles by product/epoch/resolution and zip the results.")
    ap.add_argument("--product", default=DEFAULT_PRODUCT, help="GHSL product (default: GHS_POP)")
    ap.add_argument("--res", default="4326_3ss", choices=sorted(RES_MAP.keys()),
                    help="Resolution/CRS key (default: 4326_3ss)")
    # epochs: choose one
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--epoch", type=int, help="Single epoch (e.g., 1980)")
    g.add_argument("--epochs", nargs="+", help="Explicit list of epochs (e.g., 1975 1980 2000 2020)")
    g.add_argument("--epoch-range", dest="epoch_range", action="store_true", help="Use --epoch-start/--epoch-end/--epoch-step")
    ap.add_argument("--epoch-start", type=int, help="Range start epoch (with --epoch-range)")
    ap.add_argument("--epoch-end", type=int, help="Range end epoch (with --epoch-range)")
    ap.add_argument("--epoch-step", type=int, default=5, help="Range step (default: 5)")
    ap.add_argument("--out-dir", default="data/GHSL", help="Where to store downloads")
    ap.add_argument("--zip", dest="zip_mode", choices=["none", "per-epoch", "all"], default="per-epoch",
                    help="Zip mode (default: per-epoch)")
    ap.add_argument("--max-workers", type=int, default=WORKERS, help="Parallel downloads (default: auto)")
    ap.add_argument("--filter", nargs="*", help="Optional glob(s) to filter tiles (e.g., 'R10_*' 'R*_*5.zip')")
    ap.add_argument("--dry-run", action="store_true", help="List what would be downloaded")
    args = ap.parse_args()

    epochs = parse_epoch_list(args)
    out_root = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_root)

    all_downloaded = []

    with requests.Session() as session:
        for epoch in epochs:
            tiles_url = build_epoch_dir(args.product, epoch, args.res)
            if not can_scrape(tiles_url):
                print(f"[ERROR] Scraping disallowed by robots.txt: {tiles_url}")
                continue
            try:
                tile_names = list_tiles(session, tiles_url)
            except Exception as e:
                print(f"[WARN] Could not list tiles for {epoch} @ {args.res}: {e}", file=sys.stderr)
                continue

            if args.filter:
                filtered = []
                for pat in args.filter:
                    filtered.extend(fnmatch.filter(tile_names, pat))
                tile_names = sorted(set(filtered))

            epoch_dir = out_root / args.product / f"E{epoch}" / args.res / "tiles"
            ensure_dir(epoch_dir)

            print(f"[INFO] {epoch}: {len(tile_names)} tiles from {tiles_url}")
            if args.dry_run:
                for n in tile_names[:10]:
                    print("  ", n)
                if len(tile_names) > 10:
                    print(f"  ... (+{len(tile_names) - 10} more)")
                continue

            # download in parallel
            jobs = []
            with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                for name in tile_names:
                    url = urljoin(tiles_url, name)
                    dest = epoch_dir / name
                    if dest.exists():
                        all_downloaded.append(dest)
                        continue
                    jobs.append(ex.submit(download_file, session, url, dest))

                for fut in cf.as_completed(jobs):
                    name, ok = fut.result()
                    if ok:
                        all_downloaded.append(epoch_dir / name)
                        print(f"[OK] {name}")
                    else:
                        print(f"[FAIL] {name}", file=sys.stderr)

            # zip per-epoch
            if args.zip_mode == "per-epoch":
                zip_path = out_root / args.product / f"{args.product}_E{epoch}_{args.res}.zip"
                files = sorted((epoch_dir).glob("*.zip"))
                if files:
                    print(f"[ZIP] {zip_path.name} ({len(files)} files)")
                    zip_output(zip_path, files)

    # zip all together
    if args.zip_mode == "all" and all_downloaded:
        # Put the big zip next to product folder
        zip_path = out_root / args.product / f"{args.product}_ALL_{args.res}.zip"
        files = sorted(set(all_downloaded))
        print(f"[ZIP-ALL] {zip_path.name} ({len(files)} files)")
        zip_output(zip_path, files)

    print("[DONE]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
