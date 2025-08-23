# tsunamis.py — coastline-lines gating, snap back-off, streamed DEM unions
# -----------------------------------------------------------------------
# - Uses split coastline *lines* shapefile and clips it hard to the event gate
# - Antimeridian safe (splits gate, clips piecewise)
# - ROI built from (coast buffer ∩ countries), not global polygons
# - Snap-to-coast with back-off radius until ≥ target acceptance (default 90%)
# - Windows-safe multiprocessing; per-tile unions with heartbeat + ETA
# - DEM window iterator restricted to ROI bounds; GDAL caches tuned

import os
# --- Set GDAL/PROJ *before* geospatial imports
os.environ.setdefault("GDAL_DATA", r"C:\OSGeo4W\share\gdal")
os.environ.setdefault("PROJ_LIB",  r"C:\OSGeo4W\share\proj")
os.environ.setdefault("GPD_READ_FILE_ENGINE",  "pyogrio")
os.environ.setdefault("GPD_WRITE_FILE_ENGINE", "pyogrio")

# parallel IO hints
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
os.environ.setdefault("OGR_NUM_THREADS",  "ALL_CPUS")

# Global GDAL cache ~32 GB (MB units; no commas)
os.environ.setdefault("GDAL_CACHEMAX", "32768")

import logging, math, tempfile, time, warnings
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import (
    Point, Polygon, MultiPolygon, LineString, MultiLineString,
    GeometryCollection, shape, mapping, box
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union, nearest_points, split as _shp_split
import shapely
from shapely import to_wkb

import rasterio
from rasterio import features, windows
from rasterio.enums import Resampling
from rasterio.windows import Window

import fiona
from typing import Optional

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

try:
    from memory_profiler import profile
except Exception:  # pragma: no cover
    def profile(f): return f

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- local helpers / constants
try:
    from .helpers import diagnose_geom as _ext_diagnose_geom, ISO_COL as _EXT_ISO
except Exception:
    _ext_diagnose_geom = None
    _EXT_ISO = None

try:
    from .dem_manager_local import build_local_dem_index
except Exception as _e:
    raise RuntimeError(f"dem_manager_local.build_local_dem_index not importable: {_e}")

# ---------------------------
# Silence noisy warnings/logs
# ---------------------------
warnings.filterwarnings("ignore", category=FutureWarning, module="osgeo.osr")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rasterio")
warnings.filterwarnings("ignore", message=".*'Memory' driver is deprecated since GDAL 3.11.*")
logging.getLogger("rasterio").setLevel(logging.ERROR)
logging.getLogger("rasterio._env").setLevel(logging.ERROR)

# ---------------------------
# Constants & Env dials
# ---------------------------
DEFAULT_DEM_LOCAL_ROOT = os.getenv("DEM_LOCAL_ROOT")
DEFAULT_MAX_INLAND_KM = 1.0     # tsunami inland hard cap
DEFAULT_COAST_TOUCH_M = 200.0   # band width shapes must touch
WGS84 = "EPSG:4326"
DEFAULT_EVENT_SWEEP_KM = float(os.getenv("EVENT_SWEEP_KM", "2000"))
OCEANS_PATH = os.getenv("OCEANS_PATH", None)  # Optional Natural Earth oceans
HULL_PAD_KM = 50.0
COAST_SIMPLIFY_M = float(os.getenv("COAST_SIMPLIFY_M", "500"))   # kept for any line simplification use
IDW_MAX_SAMPLES = int(os.getenv("IDW_MAX_SAMPLES", "20000"))      # cap samples
PER_TILE_GDAL_CACHE_MB = int(os.getenv("PER_TILE_GDAL_CACHE_MB", "8192"))  # 8 GB per worker
MAX_TILE_WORKERS = int(os.getenv("MAX_TILE_WORKERS", "6"))        # up to 6
SNAP_TARGET_ACCEPT = float(os.getenv("SNAP_TARGET_ACCEPT", "0.9"))

# Your split coastline *lines* shapefile:
COASTLINE_PATH = os.getenv(
    "COASTLINE_PATH",
    r"C:\Users\FAAF\Desktop\Python\projects\natural_disasters\coastlines-split-4326\lines.shp"
)

# ---------------------------
# Small logging helpers
# ---------------------------
def _diagnose_geom(tag: str, geom):
    if _ext_diagnose_geom:
        try:
            _ext_diagnose_geom(tag, geom); return
        except Exception:
            pass
    try:
        b = getattr(geom, "bounds", None)
        logging.debug(
            f"[{tag}] type={getattr(geom, 'geom_type', type(geom))}, "
            f"empty={getattr(geom,'is_empty',None)}, valid={getattr(geom,'is_valid',None)}, "
            f"bounds={b}"
        )
    except Exception:
        pass

def _log_df_schema(tag: str, df: pd.DataFrame):
    try:
        logging.info(f"[{tag}] rows={len(df)}, cols={list(df.columns)}")
        if hasattr(df, "dtypes"):
            logging.info(f"[{tag}] dtypes:\n{df.dtypes}")
    except Exception:
        pass

# ---------------------------
# Geometry/GeoDataFrame utils
# ---------------------------
def _crosses_antimeridian(geoms) -> bool:
    try:
        xs = []
        for g in geoms:
            if g is None or g.is_empty:
                continue
            minx, _, maxx, _ = g.bounds
            xs += [minx, maxx]
        if not xs:
            return False
        return (max(xs) - min(xs)) > 300.0
    except Exception:
        return False

def _split_on_dateline(geom):
    """Split geometry on the antimeridian using Shapely split."""
    if geom is None or geom.is_empty:
        return [geom]
    try:
        eps = 1e-9
        mer1 = LineString([(180.0 - eps, -90.0), (180.0 - eps,  90.0)])
        mer2 = LineString([(-180.0 + eps, -90.0), (-180.0 + eps,  90.0)])
        splitters = shapely.geometry.MultiLineString([mer1, mer2])
        res = _shp_split(geom, splitters)
        return list(res.geoms) if hasattr(res, "geoms") else [geom]
    except Exception:
        return [geom]

def _metric_crs_for_bounds(bounds):
    # Use polar stereos near poles, web mercator elsewhere
    _, miny, _, maxy = bounds
    latc = 0.5 * (miny + maxy)
    if latc >= 60.0:
        return "EPSG:3413"
    if latc <= -60.0:
        return "EPSG:3031"
    return "EPSG:3857"

def _geodesic_buffer_wgs84(geom, km: float):
    """Approx geodesic buffer by buffering in a metric CRS chosen by latitude."""
    if km <= 0 or geom is None or geom.is_empty:
        return geom
    b = gpd.GeoSeries([geom], crs=WGS84).total_bounds
    mcrs = _metric_crs_for_bounds(b)
    try:
        gm = gpd.GeoSeries([geom], crs=WGS84).to_crs(mcrs).iloc[0]
        buf = gm.buffer(km * 1000.0)
        return gpd.GeoSeries([buf], crs=mcrs).to_crs(WGS84).iloc[0]
    except Exception:
        return geom.buffer(km / 111.0)

def _build_coast_touch_band(coast_line_wgs: gpd.GeoSeries, coast_touch_m: float = DEFAULT_COAST_TOUCH_M):
    bounds = coast_line_wgs.to_crs(WGS84).total_bounds
    mcrs = _metric_crs_for_bounds(bounds)
    rim_m = coast_line_wgs.to_crs(mcrs).iloc[0].buffer(max(50.0, float(coast_touch_m)))
    return gpd.GeoSeries([rim_m], crs=mcrs).to_crs(WGS84).iloc[0]

def _activate_geometry_column(gdf: gpd.GeoDataFrame, geom_col: str | None = None, crs=WGS84) -> gpd.GeoDataFrame:
    """Ensure GeoDataFrame has a proper GeometryArray (works after pandas ops)."""
    from shapely.geometry import shape as shape_from_geojson

    if not isinstance(gdf, gpd.GeoDataFrame):
        if geom_col is None:
            cand = next((c for c in getattr(gdf, "columns", []) if c.lower() in ("geometry","geom")), None)
            if cand is None:
                raise ValueError(f"No geometry column found; cols={list(getattr(gdf,'columns',[]))}")
            geom_col = cand
        gdf = gpd.GeoDataFrame(gdf, geometry=geom_col, crs=crs)

    gname = getattr(gdf.geometry, "name", "geometry")
    if gname not in gdf.columns:
        raise ValueError(f"Active geometry '{gname}' missing; cols={list(gdf.columns)}")

    # already GeometryArray?
    try:
        from geopandas.array import GeometryDtype
        if isinstance(getattr(gdf.geometry, "dtype", None), GeometryDtype):
            if gdf.crs is None:
                gdf.set_crs(crs, allow_override=True, inplace=True)
            return gdf
    except Exception:
        pass

    # Rehydrate via WKB
    def _to_wkb(v):
        if v is None:
            return None
        if hasattr(v, "wkb"):
            return v.wkb
        try:
            g = shape_from_geojson(v)  # dict-like
            return g.wkb
        except Exception:
            pass
        try:
            from shapely import from_wkt
            return from_wkt(v).wkb if isinstance(v, str) else None
        except Exception:
            return None

    vals = gdf[gname].values
    wkb = [_to_wkb(v) for v in vals]
    from geopandas.array import from_wkb
    ga = from_wkb(np.array(wkb, dtype=object))
    gdf = gdf.set_geometry(gpd.GeoSeries(ga, crs=(gdf.crs or crs)))
    if gdf.crs is None:
        gdf.set_crs(crs, allow_override=True, inplace=True)
    return gdf

def _only_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only Polygon/MultiPolygon (convert others conservatively)."""
    g = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    if g.empty:
        return g
    def _to_poly(geom):
        if geom.geom_type in ("Polygon","MultiPolygon"):
            return geom
        try:
            hull = geom.convex_hull
            if hull.geom_type in ("Polygon","MultiPolygon"):
                return hull
        except Exception:
            pass
        try:
            b0 = geom.buffer(0)
            if b0.geom_type in ("Polygon","MultiPolygon"):
                return b0
        except Exception:
            pass
        return None
    g["geometry"] = g.geometry.apply(_to_poly)
    g = g[g.geometry.notna() & (~g.geometry.is_empty)]
    return g

def _ensure_gdf(df):
    """Coerce to GeoDataFrame (preserve active geometry), ensure WGS84."""
    if isinstance(df, gpd.GeoDataFrame):
        gdf = df.copy()
        geom_name = getattr(gdf, "geometry", None).name if hasattr(gdf, "geometry") else None
        if geom_name is None:
            for cand in ("geometry", "geom", "Geometry", "GEOMETRY"):
                if cand in gdf.columns:
                    gdf = gdf.set_geometry(c= cand, inplace=False)  # type: ignore
                    break
    else:
        cols = list(getattr(df, "columns", []))
        geom_col = None
        for cand in ("geometry", "geom", "Geometry", "GEOMETRY"):
            if cand in cols:
                geom_col = cand
                break
        if geom_col is None:
            raise ValueError(f"DEM index has no geometry column; cols={cols}")
        gdf = gpd.GeoDataFrame(df.copy(), geometry=geom_col)
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84, allow_override=True)
    elif str(gdf.crs) != WGS84:
        gdf = gdf.to_crs(WGS84)
    return gdf

def _normalize_to_polygonal(geom, crs=WGS84):
    """Return a Polygon/MultiPolygon suitable for writing."""
    if geom is None or getattr(geom, "is_empty", True):
        return None
    g = geom
    if g.geom_type == "GeometryCollection":
        polys = [p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")]
        g = unary_union(polys) if polys else g.convex_hull
    if g.geom_type not in ("Polygon","MultiPolygon"):
        try: g = g.convex_hull
        except Exception: pass
    if g.geom_type not in ("Polygon","MultiPolygon"):
        try: g = g.buffer(0)
        except Exception: pass
    if g.geom_type not in ("Polygon","MultiPolygon"):
        return None
    try:
        from shapely import set_precision
        g = set_precision(g.buffer(0), grid_size=1e-8)
    except Exception:
        g = g.buffer(0)
    return None if getattr(g, "is_empty", True) else g

def _haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0088
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2)
    return 2*R*math.asin(math.sqrt(a))

# ---------------------------
# Amplitude onion builders
# ---------------------------
def _fit_decay_params(runups_gdf, source_pt: Point, d0_km=50.0):
    """
    Fit A(d) = K * exp(-α d) / sqrt(d + d0) from runup heights.
    Returns (K, α). Constrained to 1/20000..1/2000 km^-1.
    """
    vals = []
    for _, r in runups_gdf.iterrows():
        g = r.geometry
        if g is None or g.is_empty:
            continue
        d = _haversine_km(source_pt.x, source_pt.y, g.x, g.y)  # km
        z = float(r.get("runupHt", np.nan))
        if np.isfinite(z) and z > 0:
            vals.append((d, z))
    if len(vals) < 3:
        med = float(np.nanmedian(runups_gdf.get("runupHt", np.array([1.0])))) or 1.0
        alpha = 1.0/8000.0
        K = (0.5*med) * math.sqrt(1000.0 + d0_km) * math.exp(alpha*1000.0)
        return K, alpha

    d = np.array([v[0] for v in vals], float)
    z = np.array([v[1] for v in vals], float)
    x = d
    y = np.log(z * np.sqrt(d + d0_km) + 1e-9)  # log(K) - α d
    b, a = np.polyfit(x, y, 1)
    alpha = -float(b)
    K = float(np.exp(a))
    alpha = float(np.clip(alpha, 1/20000.0, 1/2000.0))
    return K, alpha

def _solve_distance_for_level(K, alpha, L, d0_km=50.0, dmax_km=20000.0):
    """Solve L = K * exp(-α d) / sqrt(d + d0) via bisection."""
    def f(d): return K*math.exp(-alpha*d)/math.sqrt(d + d0_km) - L
    lo, hi = 0.0, dmax_km
    if f(lo) < 0: return np.nan
    if f(hi) > 0: return hi
    for _ in range(60):
        mid = 0.5*(lo+hi); fm = f(mid)
        if fm > 0: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

def build_amplitude_gate(source_pt: Point,
                         runups_gdf: gpd.GeoDataFrame,
                         levels_m=(1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01),
                         d0_km=50.0,
                         max_km=20000.0) -> gpd.GeoDataFrame:
    """
    Returns a GeoDataFrame of isochrone 'onion rings' (outermost is ≥ min(levels)).
    """
    sp = Point(float(source_pt.x), float(source_pt.y)) if source_pt else Point(
        float(np.nanmedian(runups_gdf["longitude"])), float(np.nanmedian(runups_gdf["latitude"]))
    )
    K, alpha = _fit_decay_params(runups_gdf, sp, d0_km=d0_km)
    dists = []
    for L in levels_m:
        d = _solve_distance_for_level(K, alpha, float(L), d0_km=d0_km, dmax_km=max_km)
        if not np.isfinite(d) or d <= 0:
            continue
        dists.append((L, float(d)))
    if not dists:
        return gpd.GeoDataFrame({"level":[levels_m[-1]]},
                                 geometry=[_geodesic_buffer_wgs84(sp, 10000.0)], crs=WGS84)
    polys = []
    for L, d_km in dists:
        polys.append((_geodesic_buffer_wgs84(sp, d_km), L))
    gdf = gpd.GeoDataFrame({"level":[p[1] for p in polys]},
                           geometry=[p[0] for p in polys], crs=WGS84).sort_values("level", ascending=False)
    return gdf.reset_index(drop=True)

# ---------------------------
# DEM index / tile selection utils
# ---------------------------
def _safe_select_tiles(index_gdf, geom_wgs84):
    """Select tiles intersecting geom_wgs84 (no sjoin needed)."""
    gdf = _ensure_gdf(index_gdf)
    geom = gpd.GeoSeries([geom_wgs84], crs=WGS84).iloc[0]
    try:
        idx = list(gdf.sindex.query(geom, predicate="intersects"))
        cand = gdf.iloc[idx] if idx else gdf.iloc[[]]
    except Exception:
        minx, miny, maxx, maxy = gpd.GeoSeries([geom], crs=WGS84).total_bounds
        b = gdf.geometry.bounds
        cand = gdf[(b.minx <= maxx) & (b.maxx >= minx) & (b.miny <= maxy) & (b.maxy >= miny)]
    if cand.empty:
        return cand
    mask = cand.intersects(geom)
    out = cand.loc[mask].copy()
    out.reset_index(drop=True, inplace=True)
    return out

def _promote_dem_index(index_df: pd.DataFrame, dem_tile_size_deg: int | None = None) -> gpd.GeoDataFrame:
    """
    Robustly build a GeoDataFrame from any DEM index. Must include 'path'.
    """
    _log_df_schema("DEM index (raw)", index_df)
    if "path" not in index_df.columns:
        for alt in ("PATH", "file", "filepath", "FilePath", "raster", "src"):
            if alt in index_df.columns:
                index_df = index_df.rename(columns={alt: "path"})
                break
    if "path" not in index_df.columns:
        raise ValueError("DEM index must contain a 'path' column.")
    if isinstance(index_df, gpd.GeoDataFrame) and getattr(index_df, "geometry", None) is not None:
        return _activate_geometry_column(index_df, crs=WGS84)
    for c in index_df.columns:
        s = pd.Series(index_df[c]).dropna()
        if s.empty:
            continue
        v = s.iloc[0]
        if hasattr(v, "geom_type"):
            return _activate_geometry_column(gpd.GeoDataFrame(index_df.copy(), geometry=c, crs=WGS84), crs=WGS84)
        if isinstance(v, dict) and "type" in v and "coordinates" in v:
            return _activate_geometry_column(gpd.GeoDataFrame(index_df.copy(), geometry=c, crs=WGS84), crs=WGS84)
        if isinstance(v, str) and v[:6].upper() in ("POINT(", "LINEST", "POLYGO", "MULTIP", "GEOMET"):
            return _activate_geometry_column(gpd.GeoDataFrame(index_df.copy(), geometry=c, crs=WGS84), crs=WGS84)
    for a,b,c,d in (("minx","miny","maxx","maxy"), ("left","bottom","right","top"), ("xmin","ymin","xmax","ymax")):
        if all(col in index_df.columns for col in (a,b,c,d)):
            geoms = [box(float(r[a]), float(r[b]), float(r[c]), float(r[d])) for _, r in index_df.iterrows()]
            return gpd.GeoDataFrame(index_df.copy(), geometry=gpd.GeoSeries(geoms, crs=WGS84))
    if dem_tile_size_deg:
        for cx, cy in (("lon","lat"),("longitude","latitude"),("x","y")):
            if cx in index_df.columns and cy in index_df.columns:
                half = float(dem_tile_size_deg) / 2.0
                geoms = [box(float(r[cx])-half, float(r[cy])-half, float(r[cx])+half, float(r[cy])+half)
                         for _, r in index_df.iterrows()]
                return gpd.GeoDataFrame(index_df.copy(), geometry=gpd.GeoSeries(geoms, crs=WGS84))
    raise ValueError("Could not locate/construct a geometry column in the DEM index.")

# ---------------------------
# Coast / ROI helpers
# ---------------------------
def _principal_dirs_from_runups(runups: gpd.GeoDataFrame,
                                source_pt: Point | None,
                                max_dirs: int = 3,
                                bins: int = 36) -> list[float]:
    """Return up to max_dirs principal bearings (degrees 0..360)."""
    if runups is None or runups.empty:
        return []
    g = runups[runups.geometry.notna() & (~runups.geometry.is_empty)]
    if g.empty:
        return []
    if source_pt is None:
        cx = float(np.nanmedian(g.geometry.x)); cy = float(np.nanmedian(g.geometry.y))
    else:
        cx, cy = float(source_pt.x), float(source_pt.y)
    dx = g.geometry.x.values - cx
    dy = g.geometry.y.values - cy
    ang = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    if 'runupHt' in g.columns:
        w = g['runupHt'].astype(float).values
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    else:
        w = np.ones_like(ang, dtype=float)
    hist, edges = np.histogram(ang, bins=bins, range=(0.0, 360.0), weights=w)
    if not np.any(hist):
        return []
    order = np.argsort(hist)[::-1][:max_dirs]
    centers = ((edges[order] + edges[order + 1]) * 0.5) % 360.0
    out = [float(c) for _, c in sorted(zip(hist[order], centers), key=lambda t: -t[0])]
    return out

def _make_direction_gate(source_pt: Point | None,
                         runups: gpd.GeoDataFrame,
                         half_angle_deg: float = 120.0,
                         max_dirs: int = 3,
                         radius_km: float = 10000.0) -> gpd.GeoSeries:
    """Union of up to max_dirs great-circle 'wedge' sectors pointing from source."""
    dirs = _principal_dirs_from_runups(runups, source_pt, max_dirs=max_dirs)
    if not dirs:
        return gpd.GeoSeries([box(-180, -90, 180, 90)], crs=WGS84)
    m = "EPSG:3857"
    R = radius_km * 1000.0
    c_wgs = Point(float(source_pt.x), float(source_pt.y)) if source_pt else runups.unary_union.centroid
    c_m = gpd.GeoSeries([c_wgs], crs=WGS84).to_crs(m).iloc[0]
    wedges = []
    for brg in dirs:
        base = math.radians(brg); half = math.radians(half_angle_deg)
        angs = np.linspace(base - half, base + half, 64)
        xs = c_m.x + R * np.cos(angs); ys = c_m.y + R * np.sin(angs)
        poly_m = Polygon([(c_m.x, c_m.y), *zip(xs, ys)])
        wedges.append(poly_m)
    gate_m = unary_union(wedges)
    return gpd.GeoSeries([gate_m], crs=m).to_crs(WGS84)

def _snap_points_to_coast(runups_gdf, coast_gdf, max_km=10):
    """Snap runup points to nearest coastline within max_km."""
    line_parts = []
    for geom in coast_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        gt = geom.geom_type
        if gt in ("Polygon", "MultiPolygon"):
            line_parts.append(geom.boundary)
        elif gt in ("LineString", "MultiLineString"):
            line_parts.append(geom)
        elif isinstance(geom, GeometryCollection):
            for sub in geom.geoms:
                if sub.geom_type in ("Polygon", "MultiPolygon"):
                    line_parts.append(sub.boundary)
                elif sub.geom_type in ("LineString", "MultiLineString"):
                    line_parts.append(sub)
    if not line_parts:
        return runups_gdf.iloc[0:0].copy(), gpd.GeoSeries([], crs=coast_gdf.crs)
    merged = unary_union(line_parts)
    if merged.geom_type in ("Polygon", "MultiPolygon"):
        merged = merged.boundary
    coast_lines = gpd.GeoSeries([merged], crs=coast_gdf.crs)
    m_crs = "EPSG:3857"
    coast_m = coast_lines.to_crs(m_crs).iloc[0]

    out = runups_gdf.reset_index(drop=True).copy()
    pts_m = out.to_crs(m_crs).geometry
    snapped_pts, dists_km = [], []
    for p in pts_m:
        s = coast_m.project(p); sp = coast_m.interpolate(s)
        d_km = p.distance(sp) / 1000.0
        if (max_km is None) or (d_km <= float(max_km)):
            snapped_pts.append(sp); dists_km.append(d_km)
        else:
            snapped_pts.append(None); dists_km.append(np.inf)
    keep_mask = np.array([sp is not None for sp in snapped_pts], dtype=bool)
    if not keep_mask.any():
        return out.iloc[0:0].copy(), coast_lines
    out = out.iloc[keep_mask].copy()
    snapped_kept_m = [sp for sp in snapped_pts if sp is not None]
    snapped_series_m = gpd.GeoSeries(snapped_kept_m, crs=m_crs)
    out["snapped_dist_km"] = np.asarray(dists_km, dtype=float)[keep_mask]
    out["snapped_geom"] = snapped_series_m.to_crs(runups_gdf.crs).values
    out = out.set_geometry("snapped_geom")
    return out, coast_lines

def _idw_alongshore(snapped_runups, coast_line, power=2, min_pts=3, step_km=2.0):
    """Alongshore IDW with a hard cap on samples (IDW_MAX_SAMPLES)."""
    if len(snapped_runups) < min_pts:
        return gpd.GeoDataFrame(columns=["geometry", "ru_m"], crs=snapped_runups.crs)

    m_crs = "EPSG:3857"
    line_m = coast_line.to_crs(m_crs).iloc[0]
    snaps_m = snapped_runups.to_crs(m_crs)

    s_known, z_known = [], []
    for _, r in snaps_m.iterrows():
        s = line_m.project(r.geometry); s_known.append(s)
        z = float(r.get("runupHt", np.nan)); z_known.append(z)

    s_known = np.array(s_known, float); z_known = np.array(z_known, float)
    valid = np.isfinite(z_known)
    if valid.sum() < min_pts:
        return gpd.GeoDataFrame(columns=["geometry", "ru_m"], crs=snapped_runups.crs)
    s_known = s_known[valid]; z_known = z_known[valid]

    L_km = line_m.length / 1000.0
    max_samples = int(max(200, IDW_MAX_SAMPLES))
    n_steps = int(np.ceil(L_km / step_km))
    if n_steps > max_samples:
        step_km = max(L_km / max_samples, 0.5)
        n_steps = max_samples
    logging.info(f"[idw] coast_len≈{L_km:,.0f} km; step_km={step_km:.2f}; n_samples={n_steps}")

    t0 = time.perf_counter()
    s_targets = np.linspace(0, line_m.length, max(n_steps, 2))
    pts = [line_m.interpolate(s) for s in s_targets]
    ru_vals = []
    for st in s_targets:
        d = np.abs(s_known - st); d[d == 0] = 1e-6
        w = 1.0 / (d ** power)
        ru_vals.append((w @ z_known) / w.sum())

    coast_samples_m = gpd.GeoDataFrame({"ru_m": ru_vals}, geometry=pts, crs=m_crs)
    out = coast_samples_m.to_crs(snapped_runups.crs)
    logging.info(f"[t] idw: {(time.perf_counter()-t0):.2f}s")
    return out

def _compute_roi(coast_line_wgs: gpd.GeoSeries, country_poly_wgs: MultiPolygon | Polygon, inland_limit_km: float):
    bounds = gpd.GeoSeries([country_poly_wgs], crs=WGS84).total_bounds
    mcrs = _metric_crs_for_bounds(bounds)
    # narrow strip for centroiding
    strip_m = coast_line_wgs.to_crs(mcrs).iloc[0].buffer(2000)
    strip_wgs = gpd.GeoSeries([strip_m], crs=mcrs).to_crs(WGS84).iloc[0]

    # coast buffer in meters using local metric CRS
    coast_buffer_m = coast_line_wgs.to_crs(mcrs).iloc[0].buffer(max(500.0, inland_limit_km * 1000.0))
    coast_buffer_wgs = gpd.GeoSeries([coast_buffer_m], crs=mcrs).to_crs(WGS84).iloc[0]

    try:
        roi = gpd.GeoSeries([country_poly_wgs], crs=WGS84).intersection(coast_buffer_wgs).iloc[0]
        if (roi is None) or roi.is_empty:
            roi = coast_buffer_wgs
    except Exception:
        roi = coast_buffer_wgs
    return strip_wgs, roi

# ---------- NEW: load/coastline clip strictly to gate (antimeridian safe) ----------
def _load_event_coastline_lines(gate_geom):
    """
    Load coastline *lines* and HARD-CLIP them to the event gate.
    Returns a GeoSeries with a single merged LineString/MultiLineString in WGS84.
    Antimeridian-safe: splits the gate and clips piecewise.
    """
    if (not COASTLINE_PATH) or (not os.path.exists(COASTLINE_PATH)):
        raise FileNotFoundError(f"COASTLINE_PATH not found: {COASTLINE_PATH}")

    cl = gpd.read_file(COASTLINE_PATH)
    cl = _activate_geometry_column(cl, crs=WGS84).to_crs(WGS84)
    cl = cl[cl.geometry.notna() & (~cl.geometry.is_empty)]
    if cl.empty:
        raise RuntimeError("Loaded coastline is empty.")

    if (gate_geom is None) or gate_geom.is_empty:
        merged = unary_union(list(cl.geometry))
        if merged.geom_type in ("Polygon", "MultiPolygon"):
            merged = merged.boundary
        return gpd.GeoSeries([merged], crs=WGS84)

    # Padded bbox preselect
    pad_deg = float(os.getenv("COAST_CLIP_PAD_DEG", "2"))
    minx, miny, maxx, maxy = gpd.GeoSeries([gate_geom], crs=WGS84).total_bounds
    bbox = box(minx - pad_deg, miny - pad_deg, maxx + pad_deg, maxy + pad_deg)

    try:
        idx = list(cl.sindex.query(bbox, predicate="intersects"))
        cl = cl.iloc[idx] if idx else cl.iloc[[]]
    except Exception:
        cl = cl[cl.intersects(bbox)]
    if cl.empty:
        raise RuntimeError("Coastline empty after bbox preselect.")

    # Hard clip helper
    def _hard_clip(lines_gdf, poly):
        poly_gdf = gpd.GeoDataFrame(geometry=[poly], crs=WGS84)
        try:
            return gpd.clip(lines_gdf, poly_gdf)
        except Exception:
            try:
                return gpd.overlay(lines_gdf, poly_gdf, how="intersection")
            except Exception:
                out = lines_gdf.copy()
                out["geometry"] = out.geometry.apply(lambda g: g.intersection(poly))
                out = out[out.geometry.notna() & (~out.geometry.is_empty)]
                return out

    # Antimeridian split of the gate
    parts = _split_on_dateline(gate_geom) if _crosses_antimeridian([gate_geom]) else [gate_geom]

    clipped_parts = []
    for part in parts:
        if part is None or part.is_empty:
            continue
        sub = _hard_clip(cl, part)
        if not sub.empty:
            clipped_parts.append(sub)

    if not clipped_parts:
        raise RuntimeError("Coastline empty after gate clip.")

    clipped = pd.concat(clipped_parts, ignore_index=True)
    merged = unary_union(list(clipped.geometry))
    if merged.geom_type in ("Polygon", "MultiPolygon"):
        merged = merged.boundary
    return gpd.GeoSeries([merged], crs=WGS84)

# ---------- NEW: event ROI from *lines* + countries land mask ----------
def _compute_event_roi_from_lines(runups_gdf, countries_gdf, inland_limit_km: float, pre_gate=None):
    """
    Build event ROI using coastlines *lines*:
      - coast_line: merged line clipped to pre_gate
      - roi_wgs: (coast buffer (inland_limit_km)) ∩ (countries union)
      - strip_wgs: thin coastal strip for centroid/wedge work
      - coast_full: GeoDataFrame(lines) suitable for snapping
    """
    # If no gate supplied yet, use the convex hull of runups buffered by DEFAULT_EVENT_SWEEP_KM
    if pre_gate is None or pre_gate.is_empty:
        hull = runups_gdf.unary_union.convex_hull
        pre_gate = _geodesic_buffer_wgs84(hull, DEFAULT_EVENT_SWEEP_KM)

    # Clip coastline lines to gate (strict)
    coast_line = _load_event_coastline_lines(pre_gate)  # GeoSeries[1], WGS84

    # Metric CRS for buffers
    mcrs = _metric_crs_for_bounds(coast_line.total_bounds)

    # A very thin strip (~2 km) along the coast for robust centroiding
    strip_m = coast_line.to_crs(mcrs).iloc[0].buffer(2000)
    strip_wgs = gpd.GeoSeries([strip_m], crs=mcrs).to_crs(WGS84).iloc[0]

    # Coast buffer inland
    coast_buffer_m = coast_line.to_crs(mcrs).iloc[0].buffer(max(500.0, inland_limit_km * 1000.0))
    coast_buffer_wgs = gpd.GeoSeries([coast_buffer_m], crs=mcrs).to_crs(WGS84).iloc[0]

    # Restrict to land using countries union
    try:
        country_union = unary_union(list(countries_gdf.geometry))
    except Exception:
        country_union = unary_union([g for g in countries_gdf.geometry if g is not None and not g.is_empty])

    try:
        roi = gpd.GeoSeries([country_union], crs=WGS84).intersection(coast_buffer_wgs).iloc[0]
        if (roi is None) or roi.is_empty:
            roi = coast_buffer_wgs
    except Exception:
        roi = coast_buffer_wgs

    # Provide a lines GeoDataFrame for snapping
    coast_full = gpd.GeoDataFrame(geometry=coast_line, crs=WGS84)

    # Log length after gating
    try:
        L_km = coast_line.to_crs("EPSG:3857").length.sum() / 1000.0
        logging.info(f"[event] coastline length in gate ≈ {L_km:,.0f} km")
    except Exception:
        pass

    return coast_line, strip_wgs, roi, coast_full

def sector_clip(poly, coast_centroid_wgs, source_point_wgs, half_angle_deg=60, max_range_km=2000):
    """Clip polygon to a wedge facing from coastal centroid toward the source."""
    if poly is None or getattr(poly, "is_empty", True) or source_point_wgs is None or coast_centroid_wgs is None:
        return poly
    m_crs = "EPSG:3857"
    c_m = gpd.GeoSeries([coast_centroid_wgs], crs=WGS84).to_crs(m_crs).iloc[0]
    s_m = gpd.GeoSeries([source_point_wgs], crs=WGS84).to_crs(m_crs).iloc[0]
    dx = s_m.x - c_m.x; dy = s_m.y - c_m.y
    base_ang_rad = math.atan2(dy, dx); half = math.radians(half_angle_deg)
    R = max_range_km * 1000.0
    angles = np.linspace(base_ang_rad - half, base_ang_rad + half, 64)
    xs = c_m.x + R * np.cos(angles); ys = c_m.y + R * np.sin(angles)
    wedge_m = Polygon([(c_m.x, c_m.y), *zip(xs, ys)])
    poly_m = gpd.GeoSeries([poly], crs=WGS84).to_crs(m_crs).iloc[0]
    clipped_m = poly_m.intersection(wedge_m)
    return gpd.GeoSeries([clipped_m], crs=m_crs).to_crs(WGS84).iloc[0]

# ---------------------------
# Window iterator (restricted to ROI bounds)
# ---------------------------
def _iter_windows(ds: rasterio.io.DatasetReader, roi_bounds=None, tile=512):
    """
    Yield fixed-size windows (tile×tile), culled by optional roi_bounds.
    Antimeridian-safe: if ds is geographic and ROI spans ~global width, disable culling.
    """
    width, height = ds.width, ds.height
    if roi_bounds is None:
        for r0 in range(0, height, tile):
            for c0 in range(0, width, tile):
                w = min(tile, width - c0); h = min(tile, height - r0)
                yield Window(c0, r0, w, h)
        return
    try:
        is_geographic = (ds.crs and ds.crs.is_geographic)
    except Exception:
        is_geographic = False
    if is_geographic:
        rb_minx, _, rb_maxx, _ = roi_bounds
        if (rb_maxx - rb_minx) > 300.0:
            roi_bounds = None
            for r0 in range(0, height, tile):
                for c0 in range(0, width, tile):
                    w = min(tile, width - c0); h = min(tile, height - r0)
                    yield Window(c0, r0, w, h)
            return

    # Restrict to subset window covering ROI bounds
    win = windows.from_bounds(*roi_bounds, transform=ds.transform, width=width, height=height)
    win = win.round_offsets().round_lengths()
    r_start = max(0, int(win.row_off))
    r_stop  = min(height, int(win.row_off + win.height))
    c_start = max(0, int(win.col_off))
    c_stop  = min(width,  int(win.col_off + win.width))
    for r0 in range(r_start, r_stop, tile):
        for c0 in range(c_start, c_stop, tile):
            w = min(tile, c_stop - c0); h = min(tile, r_stop - r0)
            if w > 0 and h > 0:
                yield Window(c0, r0, w, h)

def _coast_expand_or_shrink(poly_wgs, coast_line_wgs, inland_limit_km: float, corridor_m: float = 50.0):
    """
    Ensure final polygon lies within inland coastal buffer and is connected to the coast.
    """
    if poly_wgs is None or poly_wgs.is_empty:
        return poly_wgs
    _, coast_roi = _compute_roi(gpd.GeoSeries([coast_line_wgs], crs=WGS84), poly_wgs.buffer(0).envelope, inland_limit_km)
    clipped = gpd.GeoSeries([poly_wgs], crs=WGS84).intersection(coast_roi).iloc[0]
    if clipped is None or clipped.is_empty:
        return clipped
    touch_band = _build_coast_touch_band(gpd.GeoSeries([coast_line_wgs], crs=WGS84))
    if clipped.intersects(touch_band):
        return clipped
    try:
        b = gpd.GeoSeries([clipped], crs=WGS84).total_bounds
        mcrs = _metric_crs_for_bounds(b)
        coast_m = gpd.GeoSeries([coast_line_wgs], crs=WGS84).to_crs(mcrs).iloc[0]
        poly_m  = gpd.GeoSeries([clipped], crs=WGS84).to_crs(mcrs).iloc[0]
        p_poly, p_coast = nearest_points(poly_m, coast_m)
        corridor = LineString([p_poly, p_coast]).buffer(max(10.0, corridor_m))
        bridged_m = poly_m.union(corridor)
        bridged = gpd.GeoSeries([bridged_m], crs=mcrs).to_crs(WGS84).iloc[0]
        final = gpd.GeoSeries([bridged], crs=WGS84).intersection(coast_roi).iloc[0]
        return final if (final is not None and not final.is_empty) else bridged
    except Exception:
        return clipped

# ------------------------------------------------------------
# STREAMED DEM → POLYGON PIPELINE (per-block union; ROI-culling)
# ------------------------------------------------------------
@profile
def stream_flood_polygons_from_dem(
    dem_path: str,
    roi_wgs,                          # country ∩ coastline buffer
    min_elev_m: float,
    max_elev_m: float,
    tmp_vector_path: str | None,
    layer_name: str = "flood",
    simplify_m: float = 0.0,
    min_area_m2: float = 5000.0,
    mode: str = "w",
    min_preview_hits: int = 8,
    min_valid_pixels: int = 800,
    return_union: bool = False,
    gdal_cache_mb: int = PER_TILE_GDAL_CACHE_MB,
    coast_touch_wgs=None              # thin coastal contact band (WGS84)
):
    crs_wgs = WGS84
    if (not return_union) and (mode == "w") and tmp_vector_path and os.path.exists(tmp_vector_path):
        try: os.remove(tmp_vector_path)
        except OSError: pass

    blocks_total = blocks_preview_pass = blocks_polygonized = feats_written = 0
    tile_parts_dem = []
    sink = None
    schema = {"geometry": "Polygon", "properties": {"src": "str:8"}}

    with rasterio.Env(GDAL_CACHEMAX=gdal_cache_mb, NUM_THREADS="ALL_CPUS"):
        ds = rasterio.open(dem_path)
        try:
            nodata = ds.nodata if ds.nodata is not None else -9999.0
            Hmax_f32 = np.float32(max_elev_m); Emin_f32 = np.float32(min_elev_m)
            roi_dem = gpd.GeoSeries([roi_wgs], crs=crs_wgs).to_crs(ds.crs).iloc[0]
            coast_touch_dem = None
            if coast_touch_wgs is not None:
                coast_touch_dem = gpd.GeoSeries([coast_touch_wgs], crs=WGS84).to_crs(ds.crs).iloc[0]
            rb = roi_dem.bounds  # (minx,miny,maxx,maxy)

            if not return_union:
                sink = fiona.open(tmp_vector_path, mode, driver="GPKG",
                                  layer=layer_name, schema=schema, crs=crs_wgs)

            # Choose block iterator:
            use_native_blocks = bool(ds.profile.get("tiled", False))
            if use_native_blocks:
                block_iter = (w for (_, _), w in ds.block_windows(1))
            else:
                block_iter = _iter_windows(ds, roi_bounds=rb, tile=512)

            t0 = time.perf_counter()
            for i, window in enumerate(block_iter, 1):
                # fast bbox reject
                left, bottom, right, top = windows.bounds(window, ds.transform)
                if (right < rb[0]) or (left > rb[2]) or (top < rb[1]) or (bottom > rb[3]):
                    continue
                blocks_total += 1

                # tiny preview
                thumb = ds.read(1, window=window, out_shape=(1, 32, 32),
                                resampling=Resampling.nearest, masked=False).astype(np.float32, copy=False)
                valid_thumb = (thumb != nodata)
                if not valid_thumb.any():
                    continue
                tmin = float(np.min(thumb[valid_thumb])); tmax = float(np.max(thumb[valid_thumb]))
                if (tmin > Hmax_f32) or (tmax < Emin_f32):
                    continue
                hit = (thumb >= Emin_f32) & (thumb <= Hmax_f32)
                if int(hit.sum()) < int(min_preview_hits):
                    continue
                blocks_preview_pass += 1

                # full window
                a = ds.read(1, window=window, masked=False).astype(np.float32, copy=False)
                valid = (a != nodata)
                np.logical_and(valid, a >= Emin_f32, out=valid)
                np.logical_and(valid, a <= Hmax_f32, out=valid)
                n_valid = int(valid.sum())
                if n_valid < int(min_valid_pixels):
                    continue

                mask = valid.astype(np.uint8, copy=False)

                # polygonize positives, clip to ROI & touch rim
                block_geoms = []
                for geom_json, v in features.shapes(mask, transform=windows.transform(window, ds.transform)):
                    if v != 1:
                        continue
                    try:
                        poly_dem = shape(geom_json).buffer(0)
                    except Exception:
                        continue
                    if not poly_dem.intersects(roi_dem):
                        continue
                    if coast_touch_dem is not None and not poly_dem.intersects(coast_touch_dem):
                        continue
                    if min_area_m2 > 0.0:
                        try:
                            area_m2 = gpd.GeoSeries([poly_dem], crs=ds.crs).to_crs("EPSG:3857").area.iloc[0]
                            if area_m2 < float(min_area_m2):
                                continue
                        except Exception:
                            pass
                    block_geoms.append(poly_dem)

                if not block_geoms:
                    continue

                try:
                    block_union = unary_union(block_geoms)
                except Exception:
                    block_union = unary_union([g for g in block_geoms if (g is not None and not g.is_empty)])

                block_union = _normalize_to_polygonal(block_union, crs=str(ds.crs)) or block_union

                if simplify_m and simplify_m > 0:
                    try:
                        block_union = gpd.GeoSeries([block_union], crs=ds.crs)\
                            .to_crs("EPSG:3857").buffer(0).simplify(simplify_m)\
                            .to_crs(ds.crs).iloc[0]
                    except Exception:
                        pass

                blocks_polygonized += 1

                if return_union:
                    tile_parts_dem.append(block_union)
                else:
                    parts = (block_union.geoms if block_union.geom_type == "MultiPolygon" else [block_union])
                    wgs_parts = gpd.GeoSeries(parts, crs=ds.crs).to_crs(crs_wgs)
                    for p in wgs_parts:
                        try:
                            sink.write({"geometry": mapping(p), "properties": {"src": "dem"}})
                            feats_written += 1
                        except Exception as e:
                            logging.error(f"Fiona write failed for {tmp_vector_path}: {e}")

                # periodic per-tile progress (every ~200 windows)
                if (i % 200) == 0:
                    dt = time.perf_counter() - t0
                    logging.info(f"[{os.path.basename(dem_path)}] windows={i} "
                                 f"kept={blocks_preview_pass} polys={blocks_polygonized} "
                                 f"elapsed={dt:.1f}s")

            logging.info(
                f"[stream_flood] {os.path.basename(dem_path)} "
                f"blocks: total={blocks_total}, preview_pass={blocks_preview_pass}, "
                f"polygonized_blocks≈{blocks_polygonized}, feats_written={feats_written}"
            )

            if return_union:
                if not tile_parts_dem:
                    return None
                try:
                    tile_union = unary_union(tile_parts_dem)
                except Exception:
                    tile_union = unary_union([g for g in tile_parts_dem if (g is not None and not g.is_empty)])
                tile_union = _normalize_to_polygonal(tile_union, crs=str(ds.crs)) or tile_union
                return gpd.GeoSeries([tile_union], crs=ds.crs).to_crs(crs_wgs).iloc[0]

            return tmp_vector_path

        finally:
            try:
                if sink is not None:
                    sink.close()
            except Exception:
                pass
            try:
                ds.close()
            except Exception:
                pass

# ------------------------------------------------------------
# Dissolve helper
# ------------------------------------------------------------
def dissolve_vector_to_geom(path, layer_name="flood", dissolve_batch=5000):
    if not os.path.exists(path):
        logging.warning(f"Dissolve skipped: file not found: {path}")
        return None
    try:
        layers = fiona.listlayers(path)
        if layer_name not in layers:
            logging.warning(f"Dissolve skipped: layer '{layer_name}' not in {layers} for {path}")
            return None
    except Exception as e:
        logging.error(f"Failed listing layers for {path}: {e}")
        return None
    try:
        with fiona.open(path, "r", layer=layer_name) as src:
            n_src = len(src)
        if n_src == 0:
            logging.info(f"Dissolve skipped: no features in {path}:{layer_name}")
            return None
    except Exception as e:
        logging.error(f"Failed opening {path}:{layer_name} to count features: {e}")
        return None

    unions, batch = [], []
    try:
        with fiona.open(path, "r", layer=layer_name) as src:
            for feat in src:
                geom = shape(feat["geometry"])
                if not geom.is_empty:
                    batch.append(geom)
                if len(batch) >= dissolve_batch:
                    unions.append(unary_union(batch)); batch = []
            if batch:
                unions.append(unary_union(batch))
    except Exception as e:
        logging.error(f"Failed dissolving {path}:{layer_name}: {e}")
        return None

    if not unions:
        logging.info(f"Dissolve result empty for {path}:{layer_name}")
        return None

    final = unary_union(unions)
    final = _normalize_to_polygonal(final, crs=WGS84) or final
    logging.info(f"Dissolve OK → valid={getattr(final,'is_valid',None)}, empty={getattr(final,'is_empty',None)}")
    return final

def _bands_from_center(height_m, pct_low=0.2, pct_high=0.2):
    return max(0.0, height_m * (1.0 - pct_low)), height_m, height_m * (1.0 + pct_high)

# ---- top-level worker for per-tile DEM union (Windows-safe) ----
def _tile_union_worker(args):
    """
    Top-level function (picklable) to run a single DEM tile streaming and return a WGS84 polygon.
    """
    (dem_path, roi_wkb, coast_touch_wkb, Hmax,
     simplify_m_stream, polygon_min_area_m2,
     min_preview_hits, min_valid_pixels, gdal_cache_mb) = args
    from shapely import from_wkb
    roi_wgs = from_wkb(roi_wkb)
    coast_touch_wgs = from_wkb(coast_touch_wkb) if coast_touch_wkb is not None else None

    return stream_flood_polygons_from_dem(
        dem_path=dem_path, roi_wgs=roi_wgs,
        min_elev_m=-1.0, max_elev_m=Hmax,
        tmp_vector_path=None, layer_name="flood",
        simplify_m=simplify_m_stream, min_area_m2=polygon_min_area_m2,
        min_preview_hits=min_preview_hits, min_valid_pixels=min_valid_pixels,
        return_union=True, coast_touch_wgs=coast_touch_wgs, gdal_cache_mb=gdal_cache_mb
    )

# ---------- NEW: snap back-off until ≥ target acceptance ----------
def _snap_backoff(runups_gdf, coast_gdf, targets=(25, 50, 100, 250, 500), target_accept=SNAP_TARGET_ACCEPT):
    total = len(runups_gdf)
    for R in targets:
        snapped, _ = _snap_points_to_coast(runups_gdf, coast_gdf, max_km=R)
        kept = len(snapped)
        acc = (kept / total) if total else 0.0
        logging.info(f"[snap] radius={R} km → kept={kept}/{total} ({acc:.0%})")
        if acc >= float(target_accept) or R == targets[-1]:
            return snapped, R, acc
    return runups_gdf.iloc[0:0].copy(), targets[-1], 0.0

# ------------------------------------------------------------
# Build inundation per event with coastline-lines ROI
# ------------------------------------------------------------
@profile
def build_tsunami_inundation(
    runups_gdf,
    coast_gdf,
    event_id=None,
    sector_source_pt=None,
    inland_limit_km=10,
    band_percents=(0.2, 0.2),
    use_dem: bool = True,
    dem_local_root: str | None = None,
    dem_tile_size_deg: int = 5,
    tmp_dir: str | None = None,
    keep_temp: bool = False,
    stream_to_disk: bool = False,
    polygon_min_area_m2: float = 20000.0,
    min_preview_hits: int = 8,
    min_valid_pixels: int = 800,
    simplify_m_stream: float = 5.0,
    write_amp_onion: bool = False,
    amp_onion_path: str | None = None,
):
    t_all = time.perf_counter()
    tmp_dir = tmp_dir or tempfile.gettempdir()

    r = runups_gdf.copy()
    if event_id is not None and "tsunamiEventId" in r.columns:
        r = r[r["tsunamiEventId"] == event_id].copy()
    if r.empty:
        return gpd.GeoDataFrame(
            columns=["event_id","band","ru_center","ru_low","ru_high","num_points","method","geometry"],
            crs=WGS84, geometry="geometry"
        )

    if r.geometry.name != "geometry":
        r = gpd.GeoDataFrame(r, geometry=gpd.points_from_xy(r["longitude"], r["latitude"]), crs=WGS84)

    # Enforce tsunami inland cap
    inland_limit_km = float(min(inland_limit_km, DEFAULT_MAX_INLAND_KM))

    # Countries → clean CRS
    coast = coast_gdf.to_crs(WGS84)

    # --- Direction wedge (wide) ---
    dir_half_angle_deg = 60.0
    gate = _make_direction_gate(sector_source_pt, r, half_angle_deg=dir_half_angle_deg,
                                max_dirs=3, radius_km=10000.0)

    # Source point + amplitude onion (outermost for gating)
    source_pt = sector_source_pt or Point(
        float(np.nanmedian(r["longitude"])), float(np.nanmedian(r["latitude"]))
    )
    t0 = time.perf_counter()
    amp_gdf = build_amplitude_gate(
        source_pt=source_pt, runups_gdf=r,
        levels_m=(1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
    )
    amp_gate = amp_gdf.iloc[-1].geometry
    logging.info(f"[t] amp_onion: {(time.perf_counter()-t0):.2f}s")

    # Big gate = direction wedge ∩ amplitude outer
    try:
        big_gate = gpd.GeoSeries([amp_gate], crs=WGS84).intersection(gate.iloc[0]).iloc[0]
        if big_gate is None or big_gate.is_empty:
            big_gate = amp_gate
    except Exception:
        big_gate = amp_gate

    # 1) Event-scale ROI using *coastline lines* with pre-gate
    t0 = time.perf_counter()
    coast_line, strip_wgs, roi_wgs, coast_full = _compute_event_roi_from_lines(
        runups_gdf=r, countries_gdf=coast, inland_limit_km=inland_limit_km, pre_gate=big_gate
    )
    logging.info(f"[t] event_roi: {(time.perf_counter()-t0):.2f}s")

    # 2) Snap runups with back-off until ≥ target acceptance
    snapped, used_km, acc = _snap_backoff(r, coast_full, targets=(25, 50, 100, 250, 500),
                                          target_accept=SNAP_TARGET_ACCEPT)
    logging.info(
        f"Snapped runups: kept={len(snapped)}/{len(r)} ({acc:.1%}); radius_used={used_km} km; "
        f"median_snap_dist_km={np.nanmedian(snapped.get('snapped_dist_km', np.array([np.nan]))):.3f}"
    )

    if snapped.empty:
        # tiny hull fallback (1 km)
        h = gpd.GeoSeries([r.unary_union.convex_hull], crs=WGS84)
        mcrs = _metric_crs_for_bounds(h.total_bounds)
        hull = h.to_crs(mcrs).iloc[0].buffer(1000)
        hull = gpd.GeoSeries([hull], crs=mcrs).to_crs(WGS84).iloc[0]
        if sector_source_pt is not None:
            hull = sector_clip(hull, hull.centroid, sector_source_pt, half_angle_deg=60, max_range_km=inland_limit_km)
        return gpd.GeoDataFrame([{
            "event_id": event_id, "band": "MED", "ru_center": np.nan, "ru_low": np.nan, "ru_high": np.nan,
            "num_points": 0, "method": "points_hull_1km", "geometry": hull
        }], crs=WGS84)

    # 3) IDW alongshore on *event* coastline only (with cap)
    coast_samples = _idw_alongshore(snapped, coast_line)
    if coast_samples.empty:
        h = gpd.GeoSeries([r.unary_union.convex_hull], crs=WGS84)
        mcrs = _metric_crs_for_bounds(h.total_bounds)
        hull = h.to_crs(mcrs).iloc[0].buffer(1000)
        hull = gpd.GeoSeries([hull], crs=mcrs).to_crs(WGS84).iloc[0]
        if sector_source_pt is not None:
            hull = sector_clip(hull, hull.centroid, sector_source_pt, half_angle_deg=60, max_range_km=inland_limit_km)
        return gpd.GeoDataFrame([{
            "event_id": event_id, "band": "MED", "ru_center": np.nan, "ru_low": np.nan, "ru_high": np.nan,
            "num_points": len(snapped), "method": "points_hull_1km", "geometry": hull
        }], crs=WGS84)

    # Representative runup height
    try:
        ru_center = float(np.nanmedian(coast_samples["ru_m"]))
    except Exception:
        ru_center = np.nan
    if not np.isfinite(ru_center):
        fallback_center = float(np.nanmedian(snapped.get("runupHt"))) if "runupHt" in snapped.columns else np.nan
        ru_center = fallback_center if np.isfinite(fallback_center) else 1.0

    low_pct, high_pct = band_percents
    ru_low  = max(0.0, ru_center * (1.0 - low_pct))
    ru_high = ru_center * (1.0 + high_pct)
    logging.info(f"Runup bands (m): LOW={ru_low:.3f}, MED={ru_center:.3f}, HIGH={ru_high:.3f}")

    # Thin coastal-touch rim (enforce sea connectivity)
    coast_touch_wgs = _build_coast_touch_band(coast_line, DEFAULT_COAST_TOUCH_M)

    # Clip ROI by big gate so inland work is minimal
    try:
        roi_wgs = gpd.GeoSeries([roi_wgs], crs=WGS84).intersection(big_gate).iloc[0]
    except Exception:
        pass

    # ==== DEM tiles (index build + selection by BIG gate) ====
    tiles = gpd.GeoDataFrame()
    index_gdf = None
    if use_dem and dem_local_root:
        t0 = time.perf_counter()
        try:
            dem_roots = dem_local_root if isinstance(dem_local_root, (list, tuple)) else [dem_local_root]
            index_gdf = build_local_dem_index(
                roots=dem_roots,
                tile_size_deg=dem_tile_size_deg,
                suffix=('_elv_tiled.tif','_elv.tif','.tif'),
                cache_path=os.path.join(tmp_dir, "dem_index.gpkg"),
                force_rebuild=False
            )
            if index_gdf.crs is None or str(index_gdf.crs) != WGS84:
                index_gdf = index_gdf.set_crs(WGS84, allow_override=True)

            rb = gpd.GeoSeries([big_gate], crs=WGS84).total_bounds
            b = index_gdf.geometry.bounds
            coarse = index_gdf[(b.minx <= rb[2]) & (b.maxx >= rb[0]) &
                               (b.miny <= rb[3]) & (b.maxy >= rb[1])]
            if not coarse.empty:
                try:
                    idx = list(coarse.sindex.query(big_gate, predicate="intersects"))
                    coarse = coarse.iloc[idx] if idx else coarse.iloc[[]]
                except Exception:
                    pass
            tiles = _safe_select_tiles(coarse, big_gate) if not coarse.empty else coarse

            logging.info("Selected %d DEM tiles (big gate; total indexed=%d)",
                         len(tiles), (0 if index_gdf is None else len(index_gdf)))
            if tiles.empty:
                logging.warning("No DEM tiles in big gate; will use fallback buffers.")
        except Exception as e:
            logging.warning(f"Local DEM tile search failed (event {event_id}): {e}")
        logging.info(f"[t] dem_index: {(time.perf_counter()-t0):.2f}s")

    # Optional QA onion
    if write_amp_onion:
        onion_path = amp_onion_path or os.path.join(tmp_dir, f"amp_onion_{event_id}.gpkg")
        try:
            coastline_series = coast_line if isinstance(coast_line, gpd.GeoSeries) else gpd.GeoSeries([coast_line], crs=WGS84)
        except Exception:
            coastline_series = gpd.GeoSeries([], crs=WGS84)
        save_amp_onion(
            amp_gdf=amp_gdf, out_path=onion_path, source_pt=source_pt,
            amp_outer=big_gate, dir_gate=(gate.iloc[0] if hasattr(gate, "iloc") else None),
            roi_wgs=roi_wgs, coast_line=coastline_series,
            tiles_gdf=(tiles if (tiles is not None and not tiles.empty) else None),
            overwrite=True
        )

    results = []

    # --- DEM streaming branch ---
    if use_dem and not tiles.empty:
        from shapely import to_wkb

        max_tile_workers = min(MAX_TILE_WORKERS, (os.cpu_count() or MAX_TILE_WORKERS))
        for band, thr in [("LOW", ru_low), ("MED", ru_center), ("HIGH", ru_high)]:
            Hmax = float(thr) + 1.5
            logging.info(f"[BAND {band}] Hmax={Hmax:.2f} m a.s.l.")

            if stream_to_disk:
                tmp_gpkg = os.path.join(tmp_dir, f"_tmp_flood_{event_id}_{band}.gpkg")
                first = True
                for _, trow in tiles.iterrows():
                    try:
                        rim_tile = gpd.GeoSeries([coast_touch_wgs], crs=WGS84).intersection(trow.geometry.buffer(0.1)).iloc[0]
                    except Exception:
                        rim_tile = coast_touch_wgs
                    stream_flood_polygons_from_dem(
                        dem_path=trow["path"], roi_wgs=roi_wgs,
                        min_elev_m=-1.0, max_elev_m=Hmax,
                        tmp_vector_path=tmp_gpkg, layer_name="flood",
                        simplify_m=simplify_m_stream, min_area_m2=polygon_min_area_m2,
                        mode=("w" if first else "a"),
                        min_preview_hits=min_preview_hits, min_valid_pixels=min_valid_pixels,
                        return_union=False, coast_touch_wgs=rim_tile, gdal_cache_mb=PER_TILE_GDAL_CACHE_MB
                    )
                    first = False

                n = 0
                if os.path.exists(tmp_gpkg):
                    try:
                        if "flood" in fiona.listlayers(tmp_gpkg):
                            with fiona.open(tmp_gpkg, "r", layer="flood") as src:
                                n = len(src)
                    except Exception as e:
                        logging.warning(f"[BAND {band}] Could not count features in {tmp_gpkg}: {e}")

                t0 = time.perf_counter()
                dissolved = dissolve_vector_to_geom(tmp_gpkg, layer_name="flood", dissolve_batch=4000)
                logging.info(f"[BAND {band}] Dissolve wall time: {(time.perf_counter() - t0):.2f}s (features={n})")

                if not keep_temp:
                    try: os.remove(tmp_gpkg)
                    except OSError: pass
                else:
                    logging.info(f"[KEEP_TEMP] Preserved temp flood layer: {tmp_gpkg}")
            else:
                max_tile_workers = min(MAX_TILE_WORKERS, (os.cpu_count() or MAX_TILE_WORKERS))

                roi_wkb = to_wkb(roi_wgs)
                coast_touch_wkb = to_wkb(coast_touch_wgs) if coast_touch_wgs is not None else None

                args_iter = [
                    (trow["path"], roi_wkb, coast_touch_wkb, Hmax,
                    simplify_m_stream, polygon_min_area_m2,
                    min_preview_hits, min_valid_pixels, PER_TILE_GDAL_CACHE_MB)
                    for _, trow in tiles.iterrows()
                ]

                tile_geoms = []
                t_band = time.perf_counter()
                n_total = len(args_iter)

                def _collect_results(executor_cls):
                    tile_geoms.clear()
                    with executor_cls(max_workers=max_tile_workers) as ex:
                        futs = [ex.submit(_tile_union_worker, a) for a in args_iter]
                        for i, f in enumerate(as_completed(futs), 1):
                            try:
                                g = f.result()
                                if g is not None and (not getattr(g, "is_empty", False)):
                                    tile_geoms.append(g)
                            except Exception as e:
                                logging.error(f"[tile] worker failed: {e}")
                            if (i % 5) == 0 or i == n_total:
                                dt = time.perf_counter() - t_band
                                rate = i / max(dt, 1e-6)
                                eta_s = (n_total - i) / max(rate, 1e-6)
                                logging.info(f"[tiles] collected {i}/{n_total} results "
                                            f"(rate={rate:.2f} tiles/s, ETA≈{eta_s/60:.1f} min)")

                # First try processes; if that fails, use threads
                try:
                    _collect_results(ProcessPoolExecutor)
                except Exception as e:
                    logging.error(f"[tiles] ProcessPool failed; falling back to threads. ({e})")
                    _collect_results(ThreadPoolExecutor)

                logging.info(f"[BAND {band}] union start (n={len(tile_geoms)})")
                t_union = time.perf_counter()
                dissolved = unary_union(tile_geoms) if tile_geoms else None
                logging.info(f"[BAND {band}] union done in {time.perf_counter()-t_union:.1f}s")


            if (dissolved is None) or (hasattr(dissolved, "is_empty") and dissolved.is_empty):
                logging.warning(f"[BAND {band}] No dissolved geometry produced. Using fallback buffer.")
                mcrs = _metric_crs_for_bounds(coast_line.total_bounds)
                line_m = coast_line.to_crs(mcrs).iloc[0]
                poly_wgs = gpd.GeoSeries([line_m.buffer(1000, cap_style=2)], crs=mcrs).to_crs(WGS84).iloc[0]
            else:
                poly_wgs = _normalize_to_polygonal(dissolved, crs=WGS84)
                if poly_wgs is None or getattr(poly_wgs, "is_empty", True):
                    logging.warning(f"[BAND {band}] No polygonal dissolved geometry; using fallback buffer.")
                    mcrs = _metric_crs_for_bounds(coast_line.total_bounds)
                    line_m = coast_line.to_crs(mcrs).iloc[0]
                    poly_wgs = gpd.GeoSeries([line_m.buffer(1000, cap_style=2)], crs=mcrs).to_crs(WGS84).iloc[0]

            if sector_source_pt is not None and not poly_wgs.is_empty:
                poly_wgs = sector_clip(poly_wgs, strip_wgs.centroid, sector_source_pt,
                                       half_angle_deg=60, max_range_km=inland_limit_km)

            _diagnose_geom(f"dissolved-{band}", poly_wgs)
            try:
                area_m2 = gpd.GeoSeries([poly_wgs], crs=WGS84).to_crs('EPSG:3857').area.iloc[0]
                logging.info(f"[BAND {band}] Area≈{area_m2:,.0f} m²")
            except Exception:
                pass

            poly_wgs = _coast_expand_or_shrink(poly_wgs,
                                               coast_line.iloc[0] if hasattr(coast_line, "iloc") else coast_line,
                                               inland_limit_km)

            results.append({
                "event_id": event_id, "band": band,
                "ru_center": ru_center, "ru_low": ru_low, "ru_high": ru_high,
                "num_points": int(len(snapped)),
                "method": ("runup_IDW + DEM_threshold(binary) [parallel-tiles]"
                           if not stream_to_disk else
                           "runup_IDW + DEM_threshold(binary) [disk]"),
                "geometry": poly_wgs
            })

    # Fallback: no DEM or no tiles
    if not results:
        slope = 0.015
        D_med_km = float(np.clip(ru_center / max(slope, 1e-3) / 1000.0, 0.5, inland_limit_km))
        D_low_km  = max(0.25, 0.8 * D_med_km)
        D_high_km = min(inland_limit_km, 1.2 * D_med_km)
        line_m = coast_line.to_crs("EPSG:3857").iloc[0]
        for band_name, dist_km in [("LOW", D_low_km), ("MED", D_med_km), ("HIGH", D_high_km)]:
            poly_m = line_m.buffer(dist_km * 1000, cap_style=2)
            poly = gpd.GeoDataFrame(geometry=[poly_m], crs="EPSG:3857").to_crs(WGS84).geometry.iloc[0]
            try:
                poly = gpd.GeoSeries([poly], crs=WGS84).intersection(roi_wgs).iloc[0]
            except Exception:
                pass
            if sector_source_pt is not None:
                poly = sector_clip(poly, strip_wgs.centroid, sector_source_pt,
                                   half_angle_deg=60, max_range_km=inland_limit_km)
            results.append({
                "event_id": event_id, "band": band_name,
                "ru_center": ru_center, "ru_low": ru_low, "ru_high": ru_high,
                "num_points": int(len(snapped)),
                "method": "runup_IDW + coast buffer (no DEM)",
                "geometry": poly
            })

    return gpd.GeoDataFrame(results, crs=WGS84)

# ==== amplitude onion QA writer ==========================================
def save_amp_onion(
    amp_gdf: gpd.GeoDataFrame,
    out_path: str,
    source_pt: Point | None = None,
    amp_outer: BaseGeometry | None = None,
    dir_gate:  BaseGeometry | None = None,
    roi_wgs:   BaseGeometry | None = None,
    coast_line: gpd.GeoSeries | None = None,
    tiles_gdf: gpd.GeoDataFrame | None = None,
    overwrite: bool = True
):
    """Write a multi-layer GPKG for visual QA."""
    import shutil
    try:
        import pyogrio
        _has_ogrio = True
    except Exception:
        _has_ogrio = False

    g = amp_gdf.copy().to_crs(WGS84)
    if "level" not in g.columns:
        g["level"] = np.nan
    g["level_cm"] = (g["level"] * 100.0).round().astype("Int64")
    g["order"] = np.arange(1, len(g) + 1)

    if overwrite and os.path.exists(out_path):
        try:
            os.remove(out_path)
        except Exception:
            try: shutil.rmtree(out_path, ignore_errors=True)
            except Exception: pass

    def _write(df: gpd.GeoDataFrame, layer: str):
        if df is None or df.empty:
            return
        df = df.copy()
        df = df[df.geometry.notna() & (~df.geometry.is_empty)]
        if df.empty:
            return
        if _has_ogrio:
            import pyogrio
            pyogrio.write_dataframe(df, out_path, layer=layer, driver="GPKG")
        else:
            df.to_file(out_path, layer=layer, driver="GPKG")

    _write(g.sort_values("order"), "amp_rings")

    if amp_outer is None and not g.empty:
        amp_outer = g.iloc[-1].geometry
    if amp_outer is not None:
        _write(gpd.GeoDataFrame({"level":[float(g.iloc[-1]["level"]) if not g.empty else np.nan]},
                                geometry=[amp_outer], crs=WGS84), "amp_outer")

    if dir_gate is not None:
        _write(gpd.GeoDataFrame(geometry=[dir_gate], crs=WGS84), "dir_gate")
    if roi_wgs is not None:
        _write(gpd.GeoDataFrame(geometry=[roi_wgs], crs=WGS84), "roi")
    if coast_line is not None and len(coast_line) > 0:
        _write(gpd.GeoDataFrame(geometry=coast_line.geometry, crs=coast_line.crs or WGS84), "coastline")
    if source_pt is not None:
        _write(gpd.GeoDataFrame({"name":["source"]}, geometry=[source_pt], crs=WGS84), "source_pt")
    if tiles_gdf is not None and not tiles_gdf.empty:
        _write(tiles_gdf.to_crs(WGS84)[["path","tile_id","geometry"]]
               if all(c in tiles_gdf.columns for c in ("path","tile_id")) else
               tiles_gdf.to_crs(WGS84)[["geometry"]], "tiles")
    logging.info(f"[amp_onion] wrote QA GPKG → {out_path}")

# ---------------------------
# Write helpers
# ---------------------------
def _ensure_naive_datetime(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    try:
        return s.dt.tz_localize(None)
    except Exception:
        return s

def _prepare_for_write(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = gdf.copy()
    g = _activate_geometry_column(g, crs=WGS84)
    g = g[g.geometry.notna() & (~g.geometry.is_empty)].copy()
    if g.empty:
        return g

    def _fix_geom(geom):
        if geom is None or geom.is_empty:
            return None
        try:
            if hasattr(geom, "has_z") and geom.has_z:
                from shapely.ops import transform
                geom = transform(lambda x, y, z=None: (x, y), geom)
        except Exception:
            pass
        try:
            from shapely import set_precision
            geom = set_precision(geom.buffer(0), grid_size=1e-8)
        except Exception:
            geom = geom.buffer(0)
        return None if geom.is_empty else geom

    g["geometry"] = g.geometry.apply(_fix_geom)
    g = g[g.geometry.notna() & (~g.geometry.is_empty)]
    if "date" in g.columns:
        g["date"] = _ensure_naive_datetime(g["date"])
    return g

def _write_vector(gdf: gpd.GeoDataFrame, path: str, driver: str, layer: str):
    g = _prepare_for_write(gdf)
    if driver in ("FileGDB", "GPKG"):
        g = _only_polygons(g)
    if g.empty:
        logging.warning(f"[write] Nothing to write for {path}:{layer} (empty GeoDataFrame).")
        return

    if driver == "FileGDB":
        try:
            g.to_file(path, driver=driver, layer=layer, engine="fiona",
                      TARGET_ARCGIS_VERSION="ARCGIS_PRO_3_2_OR_LATER", METHOD="SKIP")
            logging.info(f"[fiona] wrote {path} (driver={driver}, layer={layer})")
            return
        except Exception as e:
            logging.warning(f"[fiona] GDB write failed, retry with precision reduction: {e}")
            try:
                from shapely import set_precision
                g2 = g.copy()
                g2["geometry"] = g2.geometry.apply(lambda x: set_precision(x.buffer(0), 1e-6))
                g2.to_file(path, driver=driver, layer=layer, engine="fiona",
                           TARGET_ARCGIS_VERSION="ARCGIS_PRO_3_2_OR_LATER", METHOD="SKIP")
                logging.info(f"[fiona] wrote (precision-reduced) {path} (driver={driver}, layer={layer})")
                return
            except Exception as e2:
                try:
                    dbg = os.path.splitext(path)[0] + "__debug.gpkg"
                    g.to_file(dbg, driver="GPKG", layer=f"{layer}_debug")
                    logging.info(f"[debug] wrote fallback debug GPKG: {dbg}")
                except Exception as e3:
                    logging.error(f"[debug] could not write debug GPKG: {e3}")
                logging.error(f"[fiona] GDB write retry failed: {e2}")
                raise

    try:
        import pyogrio
        pyogrio.write_dataframe(g, path, driver=driver, layer=layer)
        logging.info(f"[pyogrio] wrote {path} (driver={driver}, layer={layer})")
    except Exception as e:
        logging.warning(f"[pyogrio] write failed ({e}); falling back to Fiona.")
        try:
            g.to_file(path, driver=driver, layer=layer, engine="fiona")
            logging.info(f"[fiona] wrote {path} (driver={driver}, layer={layer})")
        except Exception as e2:
            try:
                dbg = os.path.splitext(path)[0] + "__debug.gpkg"
                g.to_file(dbg, driver="GPKG", layer=f"{layer}_debug")
                logging.info(f"[debug] wrote fallback debug GPKG: {dbg}")
            except Exception as e3:
                logging.error(f"[debug] could not write debug GPKG: {e3}")
            raise

# ------------------------------------------------------------
# Top-level tsunami processor (per event; tile-parallel inside)
# ------------------------------------------------------------
@profile
def process_tsunami_data(
    tsunami_events_csv,
    tsunami_runups_csv,
    countries_path,
    dem_dir,  # kept for signature compat (unused; use dem_local_root)
    inland_limit_km=10,
    band_percents=(0.2, 0.2),
    use_dem: bool = True,
    dem_local_root: str | None = None,
    dem_tile_size_deg: int = 5,
    output_folder: str | None = None,
    tmp_dir: str | None = None,
    event_filter: list | set | None = None,
    write_per_event: bool = False,
    per_event_dir: str | None = None,
    per_event_format: str = "gpkg",
    per_event_layer: str = "tsunami",
    write_aggregate: bool = False,
    aggregate_path: str | None = None,
    stream_to_disk: bool = False,
    polygon_min_area_m2: float = 20000.0,
    min_preview_hits: int = 8,
    min_valid_pixels: int = 800,
    simplify_m_stream: float = 5.0,
    n_workers: int = 1   # NOTE: event-level kept sequential; parallelism is per-tile
):
    dem_local_root = dem_local_root or DEFAULT_DEM_LOCAL_ROOT
    out_root = output_folder or "."
    tmp_dir = tmp_dir or out_root or tempfile.gettempdir()

    os.makedirs(out_root, exist_ok=True)
    try:
        os.makedirs(tmp_dir, exist_ok=True)
    except Exception as e:
        logging.error(f"Could not create tmp_dir {tmp_dir}: {e}")

    logging.info(f"Temp dir: {os.path.abspath(tmp_dir)}")
    logging.info(f"DEM_LOCAL_ROOT resolved to: {dem_local_root!r}")

    if write_per_event:
        per_event_dir = per_event_dir or os.path.join(out_root, "tsunami_events")
        os.makedirs(per_event_dir, exist_ok=True)
        per_event_format = per_event_format.lower()
        if per_event_format not in ("gpkg", "gdb"):
            raise ValueError("per_event_format must be 'gpkg' or 'gdb'")

    if write_aggregate and not aggregate_path:
        aggregate_path = os.path.join(out_root, "tsunamis_all.gpkg")

    # load CSVs
    try:
        events_df = pd.read_csv(tsunami_events_csv)
        runups_df = pd.read_csv(tsunami_runups_csv)
    except Exception as e:
        logging.error(f"Tsunami CSV load failed: {e}")
        return gpd.GeoDataFrame()

    # basic checks
    if not {'latitude', 'longitude'}.issubset(runups_df.columns):
        logging.error("Runup CSV missing 'latitude'/'longitude'.")
        return gpd.GeoDataFrame()
    if 'runupHt' not in runups_df.columns:
        runups_df['runupHt'] = np.nan

    if event_filter is not None:
        filt = set(event_filter)
        if 'tsunamiEventId' in runups_df.columns:
            runups_df = runups_df[runups_df['tsunamiEventId'].isin(filt)].copy()
        if 'id' in events_df.columns:
            events_df = events_df[events_df['id'].isin(filt)].copy()

    runups_gdf = gpd.GeoDataFrame(
        runups_df,
        geometry=gpd.points_from_xy(runups_df['longitude'], runups_df['latitude']),
        crs=WGS84
    )
    runups_gdf = _activate_geometry_column(runups_gdf, crs=WGS84)

    # countries
    try:
        countries = gpd.read_file(countries_path).to_crs(WGS84)
        countries = _activate_geometry_column(countries, crs=WGS84)
        try:
            from shapely.validation import make_valid
            countries["geometry"] = countries.geometry.apply(make_valid)
        except Exception:
            countries["geometry"] = countries.geometry.buffer(0)
        countries = countries[
            countries.geometry.notna() & (~countries.geometry.is_empty) & (countries.geometry.is_valid)
        ]
        if countries.empty:
            logging.error("Countries layer cleaned to empty set after validity fix.")
            return gpd.GeoDataFrame()
        ISO_COL = _EXT_ISO or "iso_a3"
        if ISO_COL not in countries.columns:
            logging.error(f"Countries layer missing '{ISO_COL}'.")
            return gpd.GeoDataFrame()
    except Exception:
        logging.exception("Failed loading countries for tsunamis")
        return gpd.GeoDataFrame()

    # ISO assignment: intersects + nearest≤50 km fallback
    try:
        runups_iso = gpd.sjoin(
            runups_gdf, countries[[ISO_COL, "geometry"]],
            how="left", predicate="intersects"
        ).rename(columns={ISO_COL: "iso_a3"}).drop(columns="index_right")
        runups_iso = _activate_geometry_column(runups_iso, crs=WGS84)

        need = runups_iso["iso_a3"].isna()
        n_need = int(need.sum())
        if n_need > 0:
            logging.info(f"ISO: {n_need} runups offshore or not intersecting; resolving via nearest≤50 km.")

            left_proj = runups_iso.loc[need, ["geometry"]].to_crs("EPSG:3857")
            right_proj = countries[[ISO_COL, "geometry"]].to_crs("EPSG:3857")

            def _clean_gdf(gdf):
                g = gdf.copy()
                g = g[g.geometry.notna() & (~g.geometry.is_empty)]
                try:
                    from shapely.validation import make_valid
                    g["geometry"] = g.geometry.apply(make_valid)
                except Exception:
                    g["geometry"] = g.geometry.buffer(0)
                g = g[(~g.geometry.is_empty) & (g.geometry.is_valid)]
                return g

            left_proj = _clean_gdf(left_proj)
            right_proj = _clean_gdf(right_proj)

            if not right_proj.empty and not left_proj.empty:
                try:
                    nearest = gpd.sjoin_nearest(
                        left_proj, right_proj, how="left",
                        distance_col="dist_m", max_distance=50_000
                    ).rename(columns={ISO_COL: "iso_a3"})
                except Exception:
                    right_pts = right_proj.copy()
                    right_pts["geometry"] = right_pts.geometry.representative_point()
                    nearest = gpd.sjoin_nearest(
                        left_proj, right_pts, how="left",
                        distance_col="dist_m", max_distance=50_000
                    ).rename(columns={ISO_COL: "iso_a3"})

                if not nearest.empty:
                    iso_fill = nearest["iso_a3"].dropna()
                    if not iso_fill.empty:
                        runups_iso.loc[iso_fill.index, "iso_a3"] = iso_fill

        logging.info(f"Runups total: {len(runups_gdf)}")
        logging.info(f"Runups with ISO: {runups_iso['iso_a3'].notna().sum()}")
        logging.info(f"Distinct event IDs: {runups_iso['tsunamiEventId'].nunique()}")
        logging.info(
            "Distinct (event, ISO) pairs: "
            f"{runups_iso.dropna(subset=['iso_a3'])[['tsunamiEventId','iso_a3']].drop_duplicates().shape[0]}"
        )
    except Exception:
        logging.exception("ISO spatial-join/nearest failed for runups")
    # group by event (event-level kept sequential for stability)
    results = []
    grouped = runups_iso.groupby('tsunamiEventId', dropna=False)

    # date helper
    def _mk_date(row):
        try:
            y = int(row.get('year')) if pd.notna(row.get('year')) else None
            m = int(row.get('month')) if pd.notna(row.get('month')) else 1
            d = int(row.get('day')) if pd.notna(row.get('day')) else 1
            return pd.to_datetime(f"{y}-{m}-{d}", errors='coerce') if y else pd.NaT
        except Exception:
            return pd.NaT
    events_df['date'] = events_df.apply(_mk_date, axis=1)

    for event_id, grp in grouped:
        if pd.isna(event_id):
            continue
        if event_filter is not None and event_id not in event_filter:
            continue

        grp_gdf = grp if isinstance(grp, gpd.GeoDataFrame) else gpd.GeoDataFrame(grp, geometry="geometry", crs=WGS84)
        grp_gdf = _activate_geometry_column(grp_gdf, crs=WGS84)
        logging.info(f"Processing event {int(event_id)} with {len(grp_gdf)} runups (event-centric)")

        inund = build_tsunami_inundation(
            runups_gdf=grp_gdf[['latitude','longitude','runupHt','tsunamiEventId','geometry']],
            coast_gdf=countries,
            event_id=event_id,
            inland_limit_km=inland_limit_km,
            band_percents=band_percents,
            use_dem=use_dem,
            dem_local_root=dem_local_root,
            dem_tile_size_deg=dem_tile_size_deg,
            tmp_dir=tmp_dir,
            stream_to_disk=stream_to_disk,
            polygon_min_area_m2=polygon_min_area_m2,
            min_preview_hits=min_preview_hits,
            min_valid_pixels=min_valid_pixels,
            simplify_m_stream=simplify_m_stream
        )
        if inund.empty:
            continue

        ev = events_df[events_df['id'] == event_id]
        inund['iso_a3'] = None
        inund['event_type'] = 'tsunami'
        inund['date'] = ev.iloc[0]['date'] if not ev.empty else pd.NaT

        if write_per_event:
            if per_event_format == "gpkg":
                out_path = os.path.join(per_event_dir, f"tsunami_{int(event_id)}.gpkg")
                driver = "GPKG"; layer = per_event_layer
            else:
                gdb_name = f"tsunami_{int(event_id)}.gdb"
                out_path = os.path.join(per_event_dir, gdb_name)
                driver = "FileGDB"; layer = per_event_layer
                try:
                    if "FileGDB" not in getattr(fiona, "supported_drivers", {}):
                        logging.warning("FileGDB writer unavailable; writing GPKG instead.")
                        out_path = os.path.join(per_event_dir, f"tsunami_{int(event_id)}.gpkg")
                        driver = "GPKG"; layer = per_event_layer
                except Exception:
                    logging.warning("Could not verify Fiona drivers; using GPKG for safety.")
                    out_path = os.path.join(per_event_dir, f"tsunami_{int(event_id)}.gpkg")
                    driver = "GPKG"; layer = per_event_layer
            try:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                _write_vector(inund, out_path, driver=driver, layer=per_event_layer)
                logging.info(f"Wrote per-event tsunami: {out_path} (layer={per_event_layer}, driver={driver})")
            except Exception as e:
                logging.error(f"Failed writing per-event output for event {event_id}: {e}")

        results.append(inund)

    if not results:
        logging.warning("No tsunami inundation polygons produced.")
        return gpd.GeoDataFrame()

    out = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs=WGS84)
    out = _activate_geometry_column(out, crs=WGS84)
    logging.info(f"Produced {len(out):,} tsunami polygons (combined).")

    if write_aggregate and aggregate_path:
        try:
            driver = "GPKG"
            if not aggregate_path.lower().endswith(".gpkg"):
                try:
                    if "FileGDB" in getattr(fiona, "supported_drivers", {}):
                        driver = "FileGDB"
                    else:
                        logging.warning("FileGDB writer not available; switching aggregate to GPKG.")
                        aggregate_path = (os.path.splitext(aggregate_path)[0] + ".gpkg")
                except Exception:
                    logging.warning("Could not check Fiona drivers; switching aggregate to GPKG.")
                    aggregate_path = (os.path.splitext(aggregate_path)[0] + ".gpkg")
            layer = "tsunamis_all"
            _write_vector(out, aggregate_path, driver=driver, layer=layer)
            logging.info(f"Wrote aggregate tsunamis: {aggregate_path} (layer={layer}, driver={driver})")
        except Exception as e:
            logging.error(f"Failed writing aggregate tsunamis: {e}")

    return out
