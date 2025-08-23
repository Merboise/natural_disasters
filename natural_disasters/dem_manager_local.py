# dem_manager_local.py  (only the bits you need to replace/add)
from __future__ import annotations
import os, re, logging
from pathlib import Path
from typing import Iterable, Tuple, Optional, List, Union
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import shapely

WGS84 = "EPSG:4326"
MERIT_TILE_DEG = 5
MERIT_FILENAME_RE = re.compile(r'([ns])(\d{2})([ew])(\d{3})', re.IGNORECASE)

log = logging.getLogger(__name__)

def _sign(letter: str) -> int:
    return -1 if letter.lower() in ("s","w") else 1

def _parse_tile_from_name(name: str):
    """
    Parse e.g. 'n10e045' -> (sw_lat, sw_lon). Returns None if not matched.
    """
    m = MERIT_FILENAME_RE.search(name)
    if not m:
        return None
    ns, lat_str, ew, lon_str = m.groups()
    return (_sign(ns)*int(lat_str), _sign(ew)*int(lon_str))

def _guess_from_path(p: Path):
    # try "n10e045_elv.tif", parent folder (e.g. elv_n00e030), then stem
    for cand in (p.stem, p.name, p.parent.name):
        got = _parse_tile_from_name(cand)
        if got:
            return got
    return None

def _scan_paths(
    roots: Union[str, Path, Iterable[Union[str, Path]]],
    suffix: Tuple[str, ...] = ('.tif', '.tiff'),
) -> List[Path]:
    """
    Recursively scan for MERIT-like tiles under one or many roots.
    Filters to files that (a) match suffix AND (b) contain an NS/EW code.
    """
    if isinstance(roots, (str, Path)):
        roots = [roots]
    paths: List[Path] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            log.warning("DEM root does not exist: %s", root)
            continue
        # Fast scandir-style recursion
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in suffix:
                continue
            if MERIT_FILENAME_RE.search(p.name) or MERIT_FILENAME_RE.search(p.parent.name):
                paths.append(p)
    paths.sort()
    log.info("Found %d DEM tiles under %s", len(paths), roots)
    return paths

def _bbox_from_sw(sw_lat: int, sw_lon: int, tile_deg: int) -> "shapely.geometry.Polygon":
    return box(sw_lon, sw_lat, sw_lon + tile_deg, sw_lat + tile_deg)

def build_local_dem_index(
    roots: Union[str, Path, Iterable[Union[str, Path]]],
    tile_size_deg: int = MERIT_TILE_DEG,
    suffix: Tuple[str, ...] = ('.tif', '.tiff'),
    cache_path: Optional[Union[str, Path]] = None,
    force_rebuild: bool = False,
) -> gpd.GeoDataFrame:
    """
    AUTOPATHING INDEX BUILDER FOR HUGE MERIT TREES

    - roots:      "C:\\...\\Elevation" OR an iterable of specific subfolders.
    - tile_size_deg: 5 for MERIT (default).
    - suffix:     tuple of filename suffixes to include (e.g., ('_elv_tiled.tif','_elv.tif','.tif')).
    - cache_path: optional .gpkg/.parquet/.feather to avoid rescanning every run.
    - force_rebuild: force a rescan even if cache exists.

    Returns GeoDataFrame with columns:
      path, tile_id, sw_lat, sw_lon, ne_lat, ne_lon, center_lat, center_lon, geometry (WGS84)
    """
    # 1) Load from cache if available
    if cache_path and (not force_rebuild) and Path(cache_path).exists():
        cp = str(cache_path)
        ext = Path(cp).suffix.lower()
        log.info("Loading DEM index cache: %s", cp)
        if ext == ".gpkg":
            gdf = gpd.read_file(cp, layer="dem_index")
        elif ext in (".parquet", ".pq"):
            gdf = gpd.read_parquet(cp)
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=WGS84)
        elif ext in (".feather", ".ft"):
            gdf = gpd.read_feather(cp)
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=WGS84)
        else:
            gdf = gpd.read_file(cp)  # best-effort
        if gdf.crs is None or str(gdf.crs) != WGS84:
            gdf = gdf.set_crs(WGS84, allow_override=True)
        return gdf

    # 2) Fresh scan
    paths = _scan_paths(roots, suffix=suffix)
    if not paths:
        raise RuntimeError("No DEM tiles found. Check root(s) and suffix filters.")

    recs = []
    for p in paths:
        sw = _guess_from_path(p)
        if not sw:
            continue
        sw_lat, sw_lon = sw
        geom = _bbox_from_sw(sw_lat, sw_lon, tile_size_deg)
        c = geom.centroid
        tile_id = f"{'n' if sw_lat>=0 else 's'}{abs(sw_lat):02d}{'e' if sw_lon>=0 else 'w'}{abs(sw_lon):03d}"
        recs.append(dict(
            path=str(p),
            tile_id=tile_id.lower(),
            sw_lat=sw_lat, sw_lon=sw_lon,
            ne_lat=sw_lat + tile_size_deg,
            ne_lon=sw_lon + tile_size_deg,
            center_lat=float(c.y), center_lon=float(c.x),
            geometry=geom,
        ))

    if not recs:
        raise RuntimeError("Scan completed but no MERIT-style tiles were parsed.")

    gdf = gpd.GeoDataFrame(pd.DataFrame.from_records(recs), geometry="geometry", crs=WGS84)

    # 3) Save cache if requested
    if cache_path:
        cp = str(cache_path)
        ext = Path(cp).suffix.lower()
        try:
            if ext == ".gpkg":
                gdf.to_file(cp, layer="dem_index", driver="GPKG")
            elif ext in (".parquet", ".pq"):
                gdf.to_parquet(cp, index=False)
            elif ext in (".feather", ".ft"):
                gdf.to_feather(cp)
            else:
                # default to gpkg if user passed unknown extension
                cp2 = str(Path(cp).with_suffix(".gpkg"))
                gdf.to_file(cp2, layer="dem_index", driver="GPKG")
                log.info("Saved DEM index to %s (auto .gpkg)", cp2)
        except Exception as e:
            log.warning("Failed to write DEM index cache (%s): %s", cp, e)

    return gdf
