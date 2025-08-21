# dem_manager_local.py
# Local DEM tile manager for MERIT-style tiling
import os, re, logging
from typing import List, Tuple, Optional, Iterable
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.warp import transform_bounds

WGS84 = "EPSG:4326"
_TILE_RE = re.compile(r'([ns])(\d{1,2})([we])(\d{3})', re.IGNORECASE)

def _parse_ll_from_name(name: str) -> Optional[Tuple[float, float]]:
    """
    Parse like 'n60w075_elv.tif' -> (lat0=+60, lon0=-75) = lower-left corner (degrees).
    """
    m = _TILE_RE.search(name)
    if not m:
        return None
    ns, lat, we, lon = m.group(1).lower(), int(m.group(2)), m.group(3).lower(), int(m.group(4))
    lat0 = float(lat if ns == 'n' else -lat)
    lon0 = float(lon if we == 'e' else -lon)
    return lat0, lon0

def _try_read_bounds_wgs84(path: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Safe bounds read in EPSG:4326 for odd filenames (fallback).
    """
    try:
        with rasterio.open(path) as ds:
            b = ds.bounds
            crs = ds.crs
            if crs is None:
                # assume WGS84 if missing
                return (b.left, b.bottom, b.right, b.top)
            if str(crs).upper() in ("EPSG:4326", "WGS84"):
                return (b.left, b.bottom, b.right, b.top)
            # reproject bounds
            return transform_bounds(crs, WGS84, b.left, b.bottom, b.right, b.top, densify_pts=21)
    except Exception as e:
        logging.warning(f"[DEM index] Skipping {path}: cannot read bounds ({e})")
        return None

def _suffix_match(name: str, suffixes: Iterable[str]) -> bool:
    lname = name.lower()
    return any(lname.endswith(s.lower()) for s in suffixes)

def build_local_dem_index(
    root_dir: str,
    tile_size_deg: int = 5,
    suffix: str | Iterable[str] = "_elv.tif",
    allow_fallback_read: bool = True
) -> gpd.GeoDataFrame:
    """
    Walk root_dir and index tiles like n60w075_elv.tif (or s##e###_elv.tif).
    Returns GeoDataFrame with columns: [path, lat0, lon0, tile_size, geometry] in EPSG:4326.

    Parameters
    ----------
    suffix : str or iterable of str
        Accepted file endings. Examples:
            "_elv.tif" (default)
            (".tif",)                  # any .tif
            ("_elv.tif","_elv_tiled.tif")
    allow_fallback_read : bool
        If filename doesn’t match regex, try reading bounds from the file.
    """
    if isinstance(suffix, str):
        suffixes = (suffix,)
    else:
        suffixes = tuple(suffix)

    rows = []
    for dirpath, _, files in os.walk(root_dir):
        for f in files:
            if not _suffix_match(f, suffixes):
                continue
            path = os.path.join(dirpath, f)
            ll = _parse_ll_from_name(f)
            if ll:
                lat0, lon0 = ll
                lat1, lon1 = lat0 + tile_size_deg, lon0 + tile_size_deg
                geom = box(lon0, lat0, lon1, lat1)
                rows.append({"path": path, "lat0": lat0, "lon0": lon0, "tile_size": tile_size_deg, "geometry": geom})
            elif allow_fallback_read:
                b = _try_read_bounds_wgs84(path)
                if b:
                    geom = box(*b)
                    rows.append({"path": path, "lat0": None, "lon0": None, "tile_size": None, "geometry": geom})

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=WGS84)
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)]
    gdf.reset_index(drop=True, inplace=True)

    if gdf.empty:
        logging.warning(f"[DEM index] No tiles found under {root_dir} (suffix={suffixes})")
    else:
        logging.info(f"[DEM index] Found {len(gdf)} tiles under {root_dir} (suffix={suffixes})")
    return gdf

# in dem_manager_local.py
def select_tiles_for_geom(index_gdf, geom_wgs84):
    from geopandas import GeoSeries, GeoDataFrame
    WGS84 = "EPSG:4326"

    # repair & normalize
    if not isinstance(index_gdf, GeoDataFrame):
        if "geometry" in getattr(index_gdf, "columns", []):
            index_gdf = GeoDataFrame(index_gdf, geometry="geometry")
        else:
            raise ValueError(f"[DEM select] index has no geometry column; cols={list(getattr(index_gdf,'columns',[]))}")
    if index_gdf.crs is None:
        index_gdf = index_gdf.set_crs(WGS84, allow_override=True)
    elif str(index_gdf.crs) != WGS84:
        index_gdf = index_gdf.to_crs(WGS84)

    geom = GeoSeries([geom_wgs84], crs=WGS84).iloc[0]

    try:
        idx = list(index_gdf.sindex.query(geom, predicate="intersects"))
        cand = index_gdf.iloc[idx] if idx else index_gdf.iloc[[]]
    except Exception:
        tb = GeoSeries([geom], crs=WGS84).total_bounds
        b = index_gdf.geometry.bounds
        cand = index_gdf[(b.minx <= tb[2]) & (b.maxx >= tb[0]) & (b.miny <= tb[3]) & (b.maxy >= tb[1])]

    if cand.empty:
        return cand
    out = cand.loc[cand.intersects(geom)].copy()
    out.reset_index(drop=True, inplace=True)
    return out


# (Optional) helpers you already had; kept intact
def read_windows_from_tiles(tiles_gdf: gpd.GeoDataFrame, bounds_wgs84: Tuple[float,float,float,float]):
    out = []
    for _, r in tiles_gdf.iterrows():
        path = r["path"]
        try:
            with rasterio.open(path) as ds:
                if str(ds.crs).upper() != "EPSG:4326":
                    from rasterio.vrt import WarpedVRT
                    from rasterio.warp import Resampling
                    with WarpedVRT(ds, crs="EPSG:4326", resampling=Resampling.bilinear) as vrt:
                        from rasterio.windows import from_bounds
                        win = from_bounds(*bounds_wgs84, transform=vrt.transform)
                        arr = vrt.read(1, window=win, boundless=True, masked=True)
                        transform = vrt.window_transform(win)
                        out.append((arr, transform, vrt.crs))
                else:
                    from rasterio.windows import from_bounds
                    win = from_bounds(*bounds_wgs84, transform=ds.transform)
                    arr = ds.read(1, window=win, boundless=True, masked=True)
                    transform = ds.window_transform(win)
                    out.append((arr, transform, ds.crs))
        except Exception as e:
            logging.warning(f"DEM read failed for {path}: {e}")
    return out

def mosaic_arrays(arrs: List[Tuple]):
    if not arrs:
        return None, None, None
    if len(arrs) == 1:
        a, t, c = arrs[0]
        return a, t, c
    from rasterio.merge import merge as rio_merge
    datasets, memfiles = [], []
    try:
        for a, t, c in arrs:
            mem = rasterio.io.MemoryFile()
            prof = {"driver": "GTiff","height": a.shape[0],"width": a.shape[1],
                    "count": 1,"dtype": str(a.dtype),"crs": c,"transform": t}
            ds = mem.open(**prof)
            ds.write(a, 1)
            datasets.append(ds); memfiles.append(mem)
        mosaic, out_transform = rio_merge(datasets, method="first")
        return mosaic[0], out_transform, WGS84
    finally:
        for ds in datasets:
            try: ds.close()
            except: pass
        for mf in memfiles:
            try: mf.close()
            except: pass
