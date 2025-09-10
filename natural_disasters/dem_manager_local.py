# dem_manager_local.py
from __future__ import annotations
import os, re, logging
from pathlib import Path
from typing import Iterable, Tuple, Optional, List, Union
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

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

def _bbox_from_sw(sw_lat: int, sw_lon: int, tile_deg: int):
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

    Returns GeoDataFrame with columns:
      path, tile_id, sw_lat, sw_lon, ne_lat, ne_lon, center_lat, center_lon, geometry (WGS84)
    """
    # 1) Load cache if present
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
            gdf = gpd.read_file(cp)
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

    # 3) Cache if requested
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
                cp2 = str(Path(cp).with_suffix(".gpkg"))
                gdf.to_file(cp2, layer="dem_index", driver="GPKG")
                log.info("Saved DEM index to %s (auto .gpkg)", cp2)
        except Exception as e:
            log.warning("Failed to write DEM index cache (%s): %s", cp, e)

    return gdf

# --- NEW: quick selector for tiles intersecting a (tiny) gate polygon
def select_tiles_for_gate(index_gdf: gpd.GeoDataFrame, gate_wgs84) -> gpd.GeoDataFrame:
    """
    Return subset of MERIT tiles that intersect the given WGS84 gate polygon.
    Assumes index_gdf.crs == EPSG:4326.
    """
    if index_gdf is None or index_gdf.empty:
        return index_gdf.iloc[[]]
    try:
        idx = list(index_gdf.sindex.query(gate_wgs84, predicate="intersects"))
        out = index_gdf.iloc[idx] if idx else index_gdf.iloc[[]]
    except Exception:
        out = index_gdf[index_gdf.intersects(gate_wgs84)]
    return out.reset_index(drop=True)
