# ----
# dem_manager_local.py
# Local DEM tile manager for MERIT Hydro-style folders
# ----
import os, re, math, logging
from typing import List, Tuple, Optional, Dict
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.windows import from_bounds
from rasterio.merge import merge as rio_merge

_TILE_RE = re.compile(r'([ns])(\d{1,2})([we])(\d{3})', re.IGNORECASE)

def _parse_ll_from_name(name: str) -> Optional[Tuple[float, float]]:
    """
    Parse something like 'n60w075_elv.tif' -> (lat0=+60, lon0=-75)
    Returns the *lower-left* degree of the tile.
    """
    m = _TILE_RE.search(name)
    if not m:
        return None
    ns, lat, we, lon = m.group(1).lower(), int(m.group(2)), m.group(3).lower(), int(m.group(4))
    lat0 = lat if ns == 'n' else -lat
    lon0 = lon if we == 'e' else -lon
    return float(lat0), float(lon0)

def build_local_dem_index(root_dir: str, tile_size_deg: int = 5, suffix: str = '_elv.tif') -> gpd.GeoDataFrame:
    """
    Walk root_dir and index tiles like n60w075_elv.tif (or s##e###_elv.tif).
    tile_size_deg is assumed footprint size of each tile (default 5°).
    Returns GeoDataFrame with columns: [path, lat0, lon0, tile_size, geometry] in EPSG:4326.
    """
    rows = []
    for dirpath, _, files in os.walk(root_dir):
        for f in files:
            if not f.lower().endswith('.tif'):
                continue
            if suffix and not f.lower().endswith(suffix):
                continue
            ll = _parse_ll_from_name(f)
            if not ll:
                continue
            lat0, lon0 = ll
            lat1, lon1 = lat0 + tile_size_deg, lon0 + tile_size_deg
            geom = box(lon0, lat0, lon1, lat1)
            rows.append({"path": os.path.join(dirpath, f), "lat0": lat0, "lon0": lon0, "tile_size": tile_size_deg, "geometry": geom})

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    if gdf.empty:
        logging.warning(f"DEM index: no tiles found under {root_dir}")
    else:
        logging.info(f"DEM index: found {len(gdf)} tiles under {root_dir}")
    return gdf

def select_tiles_for_geom(index_gdf: gpd.GeoDataFrame, geom_wgs84) -> gpd.GeoDataFrame:
    """Return subset of tiles whose footprints intersect the given geometry."""
    if index_gdf.empty:
        return index_gdf
    bbox = gpd.GeoDataFrame([{"geometry": geom_wgs84}], crs="EPSG:4326")
    sel = gpd.sjoin(index_gdf, bbox, how="inner", predicate="intersects").drop(columns=["index_right"])
    return sel

def read_windows_from_tiles(tiles_gdf: gpd.GeoDataFrame, bounds_wgs84: Tuple[float,float,float,float]):
    """
    Read windowed arrays from each tile intersecting bounds_wgs84 (WGS84).
    Returns list of (arr, transform, crs).
    """
    out = []
    for _, r in tiles_gdf.iterrows():
        path = r["path"]
        try:
            with rasterio.open(path) as ds:
                # If your tiles are not EPSG:4326, wrap with WarpedVRT (similar to cloud path)
                if str(ds.crs).upper() != "EPSG:4326":
                    from rasterio.vrt import WarpedVRT
                    from rasterio.warp import Resampling
                    with WarpedVRT(ds, crs="EPSG:4326", resampling=Resampling.bilinear) as vrt:
                        win = from_bounds(*bounds_wgs84, transform=vrt.transform)
                        arr = vrt.read(1, window=win, boundless=True, masked=True)
                        transform = vrt.window_transform(win)
                        out.append((arr, transform, vrt.crs))
                else:
                    win = from_bounds(*bounds_wgs84, transform=ds.transform)
                    arr = ds.read(1, window=win, boundless=True, masked=True)
                    transform = ds.window_transform(win)
                    out.append((arr, transform, ds.crs))
        except Exception as e:
            logging.warning(f"DEM read failed for {path}: {e}")
    return out

def mosaic_arrays(arrs: List[Tuple[np.ndarray, rasterio.Affine, any]]):
    """
    Mosaic several windowed arrays (all EPSG:4326 after read_windows_from_tiles).
    Uses rasterio.merge.merge on in-memory datasets.
    Returns (arr, transform, crs) or (None, None, None) if nothing.
    """
    if not arrs:
        return None, None, None
    if len(arrs) == 1:
        a, t, c = arrs[0]
        return a, t, c

    # wrap arrays as in-memory datasets and merge
    datasets = []
    memfiles = []
    try:
        for a, t, c in arrs:
            mem = rasterio.io.MemoryFile()
            prof = {
                "driver": "GTiff",
                "height": a.shape[0],
                "width": a.shape[1],
                "count": 1,
                "dtype": str(a.dtype),
                "crs": c,
                "transform": t
            }
            ds = mem.open(**prof)
            ds.write(a, 1)
            datasets.append(ds)
            memfiles.append(mem)

        mosaic, out_transform = rio_merge(datasets, method="first")  # or "min"/"max"
        return mosaic[0], out_transform, "EPSG:4326"
    finally:
        for ds in datasets:
            try: ds.close()
            except: pass
        for mf in memfiles:
            try: mf.close()
            except: pass
