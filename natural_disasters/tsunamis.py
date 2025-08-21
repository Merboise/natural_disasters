# tsunamis.py — streamlined + fast-mode + optional parallel
# ---------------------------------------------------------
# Key improvements:
# - Preview & valid-pixel thresholds to skip near-empty blocks
# - Per-block union before writing (massive dissolve speedup)
# - Optional fast mode: no temp GPKGs; tile unions done in-memory
# - Optional parallelization across (event, ISO) groups
# - More robust logging, normalization (C8), and writer hardening

import os, logging, math, tempfile, time
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import (
    Point, Polygon, MultiPolygon, LineString, MultiLineString,
    GeometryCollection, shape, mapping
)
from shapely.ops import unary_union

import rasterio
from rasterio import features
from rasterio.enums import Resampling
from datetime import datetime  # noqa

import fiona

try:
    from memory_profiler import profile
except Exception:
    # fallback if memory_profiler is absent
    def profile(f): return f

# env (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from .helpers import diagnose_geom, ISO_COL
from .dem_manager_local import build_local_dem_index, select_tiles_for_geom  # noqa: F401 (may be unused)

os.environ.setdefault("GDAL_DATA", r"C:\OSGeo4W\share\gdal")
os.environ.setdefault("PROJ_LIB",  r"C:\OSGeo4W\share\proj")
# Optional: default IO engine choices
os.environ.setdefault("GPD_READ_FILE_ENGINE",  "pyogrio")
os.environ.setdefault("GPD_WRITE_FILE_ENGINE", "pyogrio")  # Fiona forced for FileGDB in _write_vector

DEFAULT_DEM_LOCAL_ROOT = os.getenv("DEM_LOCAL_ROOT")
WGS84 = "EPSG:4326"


# ---------------------------
# Geometry utility functions
# ---------------------------
def _snap_points_to_coast(runups_gdf, coast_gdf, max_km=10):
    """
    Snap runup points to the nearest coastline within max_km.
    Returns (snapped_points_gdf, coast_lines_geoseries)
    """
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

    snapped_pts = []
    dists_km = []
    for p in pts_m:
        s = coast_m.project(p)
        sp = coast_m.interpolate(s)
        d_km = p.distance(sp) / 1000.0
        if (max_km is None) or (d_km <= float(max_km)):
            snapped_pts.append(sp)
            dists_km.append(d_km)
        else:
            snapped_pts.append(None)
            dists_km.append(np.inf)

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


def _diagnose_geom(tag: str, geom):
    try:
        b = getattr(geom, "bounds", None)
        logging.debug(
            f"[{tag}] type={getattr(geom, 'geom_type', type(geom))}, "
            f"empty={getattr(geom,'is_empty',None)}, valid={getattr(geom,'is_valid',None)}, "
            f"bounds={b}"
        )
    except Exception:
        pass


def _activate_geometry_column(gdf: gpd.GeoDataFrame, geom_col: str | None = None, crs="EPSG:4326") -> gpd.GeoDataFrame:
    import numpy as np
    from shapely.geometry import shape as shape_from_geojson

    if not isinstance(gdf, gpd.GeoDataFrame):
        if geom_col is None:
            cand = next((c for c in getattr(gdf, "columns", []) if c.lower() in ("geometry", "geom")), None)
            if cand is None:
                raise ValueError(f"No geometry column found in columns={list(getattr(gdf,'columns',[]))}")
            geom_col = cand
        gdf = gpd.GeoDataFrame(gdf, geometry=geom_col, crs=crs)

    gname = getattr(gdf.geometry, "name", "geometry")
    if gname not in gdf.columns:
        raise ValueError(f"Active geometry '{gname}' missing; cols={list(gdf.columns)}")

    # Detect if already a GeometryArray
    try:
        from geopandas.array import GeometryDtype
        if isinstance(getattr(gdf.geometry, "dtype", None), GeometryDtype):
            if gdf.crs is None:
                gdf.set_crs(crs, allow_override=True, inplace=True)
            return gdf
    except Exception:
        pass

    # Rehydrate any objects (None, shapely, or geojson-like) to a GeometryArray via WKB
    def _to_wkb(v):
        if v is None:
            return None
        if hasattr(v, "wkb"):
            return v.wkb
        # dict-like or mapping?
        try:
            g = shape_from_geojson(v)
            return g.wkb
        except Exception:
            pass
        # WKT string?
        try:
            from shapely import from_wkt
            g = from_wkt(v) if isinstance(v, str) else None
            return g.wkb if g is not None else None
        except Exception:
            return None

    vals = gdf[gname].values
    wkb_bytes = [_to_wkb(v) for v in vals]
    from geopandas.array import from_wkb
    ga = from_wkb(np.array(wkb_bytes, dtype=object))
    gdf = gdf.set_geometry(gpd.GeoSeries(ga, crs=(gdf.crs or crs)))

    if gdf.crs is None:
        gdf.set_crs(crs, allow_override=True, inplace=True)
    return gdf


def _only_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Keep only Polygon/MultiPolygon (convert others conservatively).
    Prevents “unknown geometry type” on write with Fiona/GDB.
    """
    g = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    if g.empty:
        return g

    def _to_poly(geom):
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            return geom
        # try convex hull then buffer(0)
        try:
            hull = geom.convex_hull
            if hull.geom_type in ("Polygon", "MultiPolygon"):
                return hull
        except Exception:
            pass
        try:
            b0 = geom.buffer(0)
            if b0.geom_type in ("Polygon", "MultiPolygon"):
                return b0
        except Exception:
            pass
        return None

    g["geometry"] = g.geometry.apply(_to_poly)
    g = g[g.geometry.notna() & (~g.geometry.is_empty)]
    return g


def _log_geom_mix(tag, g):
    try:
        mix = g.geometry.geom_type.value_counts().to_dict()
        logging.info(f"[{tag}] geometry mix: {mix}")
    except Exception:
        pass


def _ensure_gdf(df):
    """
    Coerce df to a GeoDataFrame with a valid geometry column and CRS=WGS84.
    IMPORTANT: we do NOT rename the geometry column here to avoid
    GeoPandas 'Unknown column geometry' issues on some versions.
    """
    if isinstance(df, gpd.GeoDataFrame):
        gdf = df.copy()
        # If, for any reason, the active geometry is unset, try to set it.
        geom_name = getattr(gdf, "geometry", None).name if hasattr(gdf, "geometry") else None
        if geom_name is None:
            for cand in ("geometry", "geom", "Geometry", "GEOMETRY"):
                if cand in gdf.columns:
                    gdf = gdf.set_geometry(cand, inplace=False)
                    break
    else:
        # Find a plausible geometry column
        cols = list(getattr(df, "columns", []))
        geom_col = None
        for cand in ("geometry", "geom", "Geometry", "GEOMETRY"):
            if cand in cols:
                geom_col = cand
                break
        if geom_col is None:
            raise ValueError(f"DEM index has no geometry column; cols={cols}")
        gdf = gpd.GeoDataFrame(df.copy(), geometry=geom_col)

    # DO NOT rename geometry; keep whatever name is active.
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84, allow_override=True)
    elif str(gdf.crs) != WGS84:
        gdf = gdf.to_crs(WGS84)
    return gdf


def _safe_select_tiles(index_gdf, geom_wgs84):
    """Select tiles intersecting geom_wgs84 without using sjoin (avoids geometry-column issues)."""
    gdf = _ensure_gdf(index_gdf)
    geom = gpd.GeoSeries([geom_wgs84], crs=WGS84).iloc[0]

    # Try spatial index first
    try:
        idx = list(gdf.sindex.query(geom, predicate="intersects"))
        cand = gdf.iloc[idx] if idx else gdf.iloc[[]]
    except Exception:
        # Fallback: coarse bbox filter
        minx, miny, maxx, maxy = gpd.GeoSeries([geom], crs=WGS84).total_bounds
        b = gdf.geometry.bounds
        cand = gdf[(b.minx <= maxx) & (b.maxx >= minx) & (b.miny <= maxy) & (b.maxy >= miny)]

    if cand.empty:
        return cand

    mask = cand.intersects(geom)
    out = cand.loc[mask].copy()
    out.reset_index(drop=True, inplace=True)
    return out


def _idw_alongshore(snapped_runups, coast_line, power=2, min_pts=3, step_km=2.0):
    if len(snapped_runups) < min_pts:
        return gpd.GeoDataFrame(columns=["geometry", "ru_m"], crs=snapped_runups.crs)
    m_crs = "EPSG:3857"
    line_m = coast_line.to_crs(m_crs).iloc[0]
    snaps_m = snapped_runups.to_crs(m_crs)

    s_known, z_known = [], []
    for _, r in snaps_m.iterrows():
        s = line_m.project(r.geometry)
        s_known.append(s)
        z_known.append(float(r.get("runupHt", np.nan)))

    s_known = np.array(s_known, dtype=float)
    z_known = np.array(z_known, dtype=float)
    valid = np.isfinite(z_known)
    if valid.sum() < min_pts:
        return gpd.GeoDataFrame(columns=["geometry", "ru_m"], crs=snapped_runups.crs)
    s_known = s_known[valid]
    z_known = z_known[valid]

    n_steps = int(np.ceil(line_m.length / (step_km * 1000.0)))
    s_targets = np.linspace(0, line_m.length, max(n_steps, 2))
    pts = [line_m.interpolate(s) for s in s_targets]

    ru_vals = []
    for st in s_targets:
        d = np.abs(s_known - st)
        d[d == 0] = 1e-6
        w = 1.0 / (d ** power)
        ru_vals.append((w @ z_known) / w.sum())

    coast_samples_m = gpd.GeoDataFrame({"ru_m": ru_vals}, geometry=pts, crs=m_crs)
    return coast_samples_m.to_crs(snapped_runups.crs)


def _coastal_strip(coast_geom, width_m=2000):
    return coast_geom.buffer(width_m)


def sector_clip(poly, coast_centroid_wgs, source_point_wgs, half_angle_deg=60, max_range_km=2000):
    """
    Clip polygon to a wedge facing from coastal centroid toward source.
    All inputs expected in WGS84. Returns original poly if inputs missing.
    """
    if poly is None or poly.is_empty or source_point_wgs is None or coast_centroid_wgs is None:
        return poly

    m_crs = "EPSG:3857"
    c_m = gpd.GeoSeries([coast_centroid_wgs], crs="EPSG:4326").to_crs(m_crs).iloc[0]
    s_m = gpd.GeoSeries([source_point_wgs], crs="EPSG:4326").to_crs(m_crs).iloc[0]

    dx = s_m.x - c_m.x
    dy = s_m.y - c_m.y
    base_ang_rad = math.atan2(dy, dx)
    half = math.radians(half_angle_deg)
    R = max_range_km * 1000.0

    angles = np.linspace(base_ang_rad - half, base_ang_rad + half, 64)
    xs = c_m.x + R * np.cos(angles)
    ys = c_m.y + R * np.sin(angles)
    wedge_m = Polygon([(c_m.x, c_m.y), *zip(xs, ys)])

    poly_m = gpd.GeoSeries([poly], crs="EPSG:4326").to_crs(m_crs).iloc[0]
    clipped_m = poly_m.intersection(wedge_m)
    return gpd.GeoSeries([clipped_m], crs=m_crs).to_crs("EPSG:4326").iloc[0]

def _log_df_schema(tag: str, df: pd.DataFrame, n_preview: int = 5):
    try:
        logging.info(f"[{tag}] type={type(df).__name__}, rows={len(df)}, cols={list(df.columns)}")
        if hasattr(df, "dtypes"):
            logging.info(f"[{tag}] dtypes=\n{df.dtypes}")
        # show first non-null sample per column
        for c in list(df.columns)[:32]:
            s = pd.Series(df[c]).dropna()
            if s.empty:
                logging.debug(f"[{tag}] {c}: all null")
                continue
            v = s.iloc[0]
            vtype = type(v).__name__
            vs = str(v)
            if len(vs) > 120:
                vs = vs[:117] + "..."
            logging.debug(f"[{tag}] sample {c}: {vtype} = {vs}")
        # quick guess of geometry-like columns
        geomish = []
        for c in df.columns:
            s = pd.Series(df[c]).dropna()
            if s.empty:
                continue
            v = s.iloc[0]
            if hasattr(v, "geom_type"):
                geomish.append((c, "shapely"))
            elif isinstance(v, dict) and "type" in v and "coordinates" in v:
                geomish.append((c, "geojson"))
            elif isinstance(v, str) and v[:6].upper() in ("POINT(", "LINEST", "POLYGO", "MULTIP", "GEOMET"):
                geomish.append((c, "wkt?"))
        if geomish:
            logging.info(f"[{tag}] geometry-like candidates: {geomish}")
        else:
            logging.info(f"[{tag}] no obvious geometry-like columns detected")
    except Exception as e:
        logging.warning(f"[{tag}] schema logging failed: {e}")

# ------------------------------------------------------------
# STREAMED DEM → POLYGON PIPELINE (with per-block union)
# ------------------------------------------------------------
@profile
def stream_flood_polygons_from_dem(
    dem_path: str,
    strip_wgs,                         # WGS84 coastal strip geometry
    min_elev_m: float,                 # Emin (e.g., -1.0)
    max_elev_m: float,                 # Hmax (ceil(ru + 1.5))
    inland_limit_km: float,            # inland cap for clip
    tmp_vector_path: str | None,       # on-disk target (GeoPackage path) if writing
    layer_name: str = "flood",
    simplify_m: float = 0.0,           # optional simplify in meters (after union)
    min_area_m2: float = 5000.0,       # drop tiny slivers (bigger default)
    mode: str = "w",                   # "w" first tile, then "a" for subsequent tiles
    # new skip thresholds:
    min_preview_hits: int = 8,         # require some hits in 32x32 preview
    min_valid_pixels: int = 200,       # require N true pixels in full block
    # fast mode:
    return_union: bool = False,        # if True, do not write; return unioned geometry (WGS84) for this tile
):
    """
    Streamed threshold→polygon pipeline for MERIT-Hydro (float32 EGM96, nodata=-9999).
    - Per-block union drastically reduces downstream dissolve work.
    - In 'return_union=True' mode, the function returns a unioned shapely geometry (WGS84)
      for the entire tile and writes nothing to disk.
    """
    crs_wgs = "EPSG:4326"
    if (not return_union) and (mode == "w") and tmp_vector_path and os.path.exists(tmp_vector_path):
        try:
            os.remove(tmp_vector_path)
        except OSError:
            pass

    blocks_total = 0
    blocks_bbox_pass = 0
    blocks_preview_pass = 0
    blocks_polygonized = 0
    feats_written = 0

    tile_parts_dem = []  # only used when return_union=True

    # Open data sources / sink conditionally
    sink = None
    schema = {"geometry": "Polygon", "properties": {"src": "str:8"}}
    try:
        ds = rasterio.open(dem_path)
        nodata = ds.nodata if ds.nodata is not None else -9999.0
        Hmax_f32 = np.float32(max_elev_m)
        Emin_f32 = np.float32(min_elev_m)

        # Precompute clips in DEM CRS + fast bbox for block culling
        strip_dem = gpd.GeoSeries([strip_wgs], crs=crs_wgs).to_crs(ds.crs).iloc[0]
        inland_clip_dem = gpd.GeoSeries([strip_wgs], crs=crs_wgs) \
            .to_crs("EPSG:3857").buffer(inland_limit_km * 1000.0) \
            .to_crs(ds.crs).iloc[0]
        ic_minx, ic_miny, ic_maxx, ic_maxy = inland_clip_dem.bounds

        if not return_union:
            sink = fiona.open(tmp_vector_path, mode, driver="GPKG",
                              layer=layer_name, schema=schema, crs=crs_wgs)

        for (j, i), window in ds.block_windows(1):
            # heartbeat
            if (j * 10_000 + i) % 500 == 0:
                logging.debug(f"Scanning blocks… j={j}, i={i}")
            blocks_total += 1

            # fast bbox reject
            left, bottom, right, top = rasterio.windows.bounds(window, ds.transform)
            if (right < ic_minx) or (left > ic_maxx) or (top < ic_miny) or (bottom > ic_maxy):
                continue
            blocks_bbox_pass += 1

            # tiny preview
            thumb = ds.read(1, window=window, out_shape=(1, 32, 32),
                            resampling=Resampling.nearest, masked=False).astype(np.float32, copy=False)
            valid_thumb = (thumb != nodata)
            if not valid_thumb.any():
                continue
            tmin = float(np.min(thumb[valid_thumb])); tmax = float(np.max(thumb[valid_thumb]))
            # quick range reject
            if (tmin > Hmax_f32) or (tmax < Emin_f32):
                continue
            # require some "in-range" pixels in preview
            hit = (thumb >= Emin_f32) & (thumb <= Hmax_f32)
            if int(hit.sum()) < int(min_preview_hits):
                continue
            blocks_preview_pass += 1

            # read the real block
            a = ds.read(1, window=window, masked=False).astype(np.float32, copy=False)
            valid = (a != nodata)
            np.logical_and(valid, a >= Emin_f32, out=valid)
            np.logical_and(valid, a <= Hmax_f32, out=valid)
            n_valid = int(valid.sum())
            if n_valid < int(min_valid_pixels):
                continue

            mask = valid.astype(np.uint8, copy=False)

            # collect all positives for this block, then union once
            block_geoms = []
            for geom_js, v in features.shapes(mask, transform=rasterio.windows.transform(window, ds.transform)):
                if v != 1:
                    continue
                try:
                    poly_dem = shape(geom_js)
                    # enforce polygonal & validity early (prevents odd collections/lines)
                    poly_dem = poly_dem.buffer(0)
                except Exception:
                    continue
                # clip tests
                if not (poly_dem.intersects(inland_clip_dem) and poly_dem.intersects(strip_dem)):
                    continue
                # sliver cut
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

            # per-block union
            try:
                block_union = unary_union(block_geoms)
            except Exception:
                # fallback: if union fails, keep parts
                block_union = MultiPolygon([g for g in block_geoms if g.geom_type == "Polygon"]) \
                    if any(g.geom_type == "Polygon" for g in block_geoms) else unary_union(block_geoms)

            # make sure the union is polygonal & valid (C8)
            block_union = _normalize_to_polygonal(block_union, crs=str(ds.crs)) or block_union

            # optional simplify in projected meters
            if simplify_m and simplify_m > 0:
                try:
                    block_union = gpd.GeoSeries([block_union], crs=ds.crs) \
                        .to_crs("EPSG:3857").buffer(0).simplify(simplify_m) \
                        .to_crs(ds.crs).iloc[0]
                except Exception:
                    pass

            blocks_polygonized += 1  # once per block producing geometry

            if return_union:
                # collect for later union (do not return inside the block loop)
                tile_parts_dem.append(block_union)
            else:
                # write minimal number of features (parts) in WGS84
                parts = (block_union.geoms if block_union.geom_type == "MultiPolygon" else [block_union])
                wgs_parts = gpd.GeoSeries(parts, crs=ds.crs).to_crs(crs_wgs)
                for p in wgs_parts:
                    try:
                        sink.write({"geometry": mapping(p), "properties": {"src": "dem"}})
                        feats_written += 1
                    except Exception as e:
                        logging.error(f"Fiona write failed for {tmp_vector_path}: {e}")

        # log summary
        logging.info(
            f"[stream_flood] {os.path.basename(dem_path)} "
            f"blocks: total={blocks_total}, bbox_pass={blocks_bbox_pass}, "
            f"preview_pass={blocks_preview_pass}, polygonized_blocks≈{blocks_polygonized}, "
            f"feats_written={feats_written}" + (f" → {os.path.basename(tmp_vector_path)}" if tmp_vector_path else "")
        )

        if return_union:
            if not tile_parts_dem:
                return None
            try:
                tile_union = unary_union(tile_parts_dem)
            except Exception:
                # defensive union
                tile_union = unary_union([g for g in tile_parts_dem if (g is not None and not g.is_empty)])
            # normalize and return in WGS84
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


def _normalize_to_polygonal(geom, crs="EPSG:4326"):
    """
    Return a Polygon/MultiPolygon suitable for writing.
    - Accepts any shapely geometry (incl. GeometryCollection/mixed)
    - Extracts polygonal parts; if none, uses convex hull as last resort
    - Fixes validity via buffer(0) and optional precision reduce
    """
    if geom is None:
        return None
    if getattr(geom, "is_empty", True):
        return None

    # 1) Extract polygonal parts if needed
    g = geom
    if g.geom_type == "GeometryCollection":
        polys = [p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")]
        if polys:
            g = unary_union(polys)
        else:
            # no polygonal content → use convex hull
            g = g.convex_hull

    # 2) If still non-polygon, try convex hull / buffer(0)
    if g.geom_type not in ("Polygon", "MultiPolygon"):
        try:
            g = g.convex_hull
        except Exception:
            pass
    if g.geom_type not in ("Polygon", "MultiPolygon"):
        try:
            g = g.buffer(0)
        except Exception:
            pass

    # 3) Still not polygonal? give up cleanly
    if g.geom_type not in ("Polygon", "MultiPolygon"):
        return None

    # 4) Fix tiny self-intersections & precision noise
    try:
        # Shapely 2.x has set_precision
        from shapely import set_precision
        g = set_precision(g.buffer(0), grid_size=1e-8)
    except Exception:
        g = g.buffer(0)

    # Final guard
    if getattr(g, "is_empty", True):
        return None
    return g


def dissolve_vector_to_geom(path, layer_name="flood", dissolve_batch=5000):
    """
    Read polygons in batches from path:layer and return a single dissolved Shapely geometry.
    Keeps memory bounded by dissolving incrementally.
    """
    if not os.path.exists(path):
        logging.warning(f"Dissolve skipped: file not found: {path}")
        return None

    # Ensure layer exists
    try:
        layers = fiona.listlayers(path)
        if layer_name not in layers:
            logging.warning(f"Dissolve skipped: layer '{layer_name}' not in {layers} for {path}")
            return None
    except Exception as e:
        logging.error(f"Failed listing layers for {path}: {e}")
        return None

    # Early-out empty layer?
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
                    unions.append(unary_union(batch))
                    batch = []
            if batch:
                unions.append(unary_union(batch))
    except Exception as e:
        logging.error(f"Failed dissolving {path}:{layer_name}: {e}")
        return None

    if not unions:
        logging.info(f"Dissolve result empty for {path}:{layer_name}")
        return None

    final = unary_union(unions)
    final = _normalize_to_polygonal(final, crs="EPSG:4326") or final
    logging.info(f"Dissolve OK for {path}:{layer_name} → valid={getattr(final,'is_valid',None)}, empty={getattr(final,'is_empty',None)}")
    return final


def _bands_from_center(height_m, pct_low=0.2, pct_high=0.2):
    return max(0.0, height_m * (1.0 - pct_low)), height_m, height_m * (1.0 + pct_high)


# ------------------------------------------------------------
# Build inundation per (event, ISO)
# ------------------------------------------------------------
@profile
def build_tsunami_inundation(
    runups_gdf,
    coast_gdf,
    event_id=None,
    sector_source_pt=None,              # shapely Point (WGS84) or None
    inland_limit_km=10,
    band_percents=(0.2, 0.2),
    use_dem: bool = True,
    dem_local_root: str | None = None,
    dem_tile_size_deg: int = 5,
    tmp_dir: str | None = None,
    keep_temp: bool = False,
    # fast/skip controls for streaming
    stream_to_disk: bool = False,       # False = in-memory unions; True = write temp GPKGs and dissolve
    polygon_min_area_m2: float = 5000.0,
    min_preview_hits: int = 8,
    min_valid_pixels: int = 200,
    simplify_m_stream: float = 0.0,     # optional simplify per-block (meters)
):
    """
    Returns GeoDataFrame with bands {LOW, MED, HIGH}.
    If use_dem and local tiles available:
      - Default (fast): tile unions & final union are done in memory (no temp files).
      - Legacy: stream to temp GPKG and dissolve on disk (stream_to_disk=True).
    Otherwise → slope-based coastline buffers (no DEM).
    """
    tmp_dir = tmp_dir or tempfile.gettempdir()

    r = runups_gdf.copy()
    if event_id is not None and "tsunamiEventId" in r.columns:
        r = r[r["tsunamiEventId"] == event_id].copy()
    if r.empty:
        return gpd.GeoDataFrame(
            columns=["event_id", "band", "ru_center", "ru_low", "ru_high", "num_points", "method", "geometry"],
            crs="EPSG:4326", geometry="geometry"
        )

    if r.geometry.name != "geometry":
        r = gpd.GeoDataFrame(r, geometry=gpd.points_from_xy(r["longitude"], r["latitude"]), crs="EPSG:4326")

    coast = coast_gdf.to_crs("EPSG:4326")

    # Snap & IDW alongshore
    snapped, coast_line = _snap_points_to_coast(r, coast)
    logging.info(
        f"Snapped runups: kept={len(snapped)}/{len(r)}; "
        f"median_snap_dist_km={np.nanmedian(snapped.get('snapped_dist_km', np.array([np.nan]))):.3f}"
    )
    if snapped.empty:
        hull = r.unary_union.convex_hull.buffer(1609)  # ~1 mile
        if sector_source_pt is not None:
            hull = sector_clip(hull, hull.centroid, sector_source_pt, half_angle_deg=60, max_range_km=inland_limit_km)
        return gpd.GeoDataFrame([{
            "event_id": event_id, "band": "MED", "ru_center": np.nan, "ru_low": np.nan, "ru_high": np.nan,
            "num_points": 0, "method": "points_hull_1mile", "geometry": hull
        }], crs="EPSG:4326")

    coast_samples = _idw_alongshore(snapped, coast_line)
    if coast_samples.empty:
        hull = r.unary_union.convex_hull.buffer(1609)
        if sector_source_pt is not None:
            hull = sector_clip(hull, hull.centroid, sector_source_pt, half_angle_deg=60, max_range_km=inland_limit_km)
        return gpd.GeoDataFrame([{
            "event_id": event_id, "band": "MED", "ru_center": np.nan, "ru_low": np.nan, "ru_high": np.nan,
            "num_points": len(snapped), "method": "points_hull_1mile", "geometry": hull
        }], crs="EPSG:4326")

    # Representative runup height (meters a.s.l., EGM96)
    try:
        ru_center = float(np.nanmedian(coast_samples["ru_m"]))
    except Exception:
        ru_center = np.nan
    if not np.isfinite(ru_center):
        fallback_center = float(np.nanmedian(snapped.get("runupHt"))) if "runupHt" in snapped.columns else np.nan
        ru_center = fallback_center if np.isfinite(fallback_center) else 1.0

    low_pct, high_pct = band_percents
    ru_low = max(0.0, ru_center * (1.0 - low_pct))
    ru_high = ru_center * (1.0 + high_pct)

    logging.info(f"Runup bands (m): LOW={ru_low:.3f}, MED={ru_center:.3f}, HIGH={ru_high:.3f}")

    # Coastal strip (compute BEFORE tile selection)
    strip_m = _coastal_strip(coast_line.to_crs("EPSG:3857").iloc[0], width_m=2000)
    strip_wgs = gpd.GeoSeries([strip_m], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]

    # Find intersecting tiles (no in-memory mosaic)
    tiles = gpd.GeoDataFrame()
    if use_dem and dem_local_root:
        try:
            index_gdf = build_local_dem_index(
                dem_local_root,
                tile_size_deg=dem_tile_size_deg,
                suffix=('_elv_tiled.tif', '_elv.tif', '.tif')
            )
            _log_df_schema("DEM index (raw)", index_gdf)

            # Manually choose the geometry column
            # 1) quick pick, if present
            if isinstance(index_gdf, gpd.GeoDataFrame):
                geom_col = getattr(getattr(index_gdf, "geometry", None), "name", None)
                if geom_col not in getattr(index_gdf, "columns", []):
                    geom_col = None
            else:
                geom_col = None

            # 2) fallback: try common names, then detect
            if geom_col is None:
                for cand in ("geometry", "geom", "GEOMETRY", "Geom", "wkt", "WKT"):
                    if cand in index_gdf.columns:
                        geom_col = cand
                        break
            if geom_col is None:
                # last-resort sniff: pick first column that looks shapely/geojson/WKT
                for c in index_gdf.columns:
                    s = pd.Series(index_gdf[c]).dropna()
                    if s.empty: 
                        continue
                    v = s.iloc[0]
                    if hasattr(v, "geom_type"):
                        geom_col = c; break
                    if isinstance(v, dict) and "type" in v and "coordinates" in v:
                        geom_col = c; break
                    if isinstance(v, str) and v[:6].upper() in ("POINT(", "LINEST", "POLYGO", "MULTIP", "GEOMET"):
                        geom_col = c; break

            if geom_col is None:
                raise ValueError("Could not locate a geometry column in the DEM index.")

            logging.info(f"[DEM index] using geometry column: {geom_col!r}")

            # Promote to GeoDataFrame safely
            index_gdf = _activate_geometry_column(index_gdf, geom_col=geom_col, crs="EPSG:4326")
            _log_df_schema("DEM index (activated)", index_gdf)

            index_gdf = _ensure_gdf(index_gdf)
            logging.info(
                f"DEM index: crs={index_gdf.crs}, geom_col={index_gdf.geometry.name}, "
                f"cols={list(index_gdf.columns)} n={len(index_gdf)}"
            )

            tiles = _safe_select_tiles(index_gdf, strip_wgs)
            logging.info(f"Selected {len(tiles)} DEM tiles for event {event_id}")

            if tiles.empty:
                logging.warning(f"No local DEM tiles intersect coastal strip (event {event_id}); using fallback buffers.")
        except Exception as e:
            logging.warning(f"Local DEM tile search failed (event {event_id}): {e}")


    results = []

    # --- DEM streaming branch ---
    if use_dem and not tiles.empty:
        # iterate three bands
        for band, thr in [("LOW", ru_low), ("MED", ru_center), ("HIGH", ru_high)]:
            Hmax = float(thr) + 1.5
            logging.info(f"[BAND {band}] Hmax={Hmax:.2f} m a.s.l.")

            dissolved = None

            if stream_to_disk:
                # legacy on-disk temp path for debugging/inspection
                tmp_gpkg = os.path.join(tmp_dir, f"_tmp_flood_{event_id}_{band}.gpkg")
                first = True

                for _, t in tiles.iterrows():
                    stream_flood_polygons_from_dem(
                        dem_path=t["path"],
                        strip_wgs=strip_wgs,
                        min_elev_m=-1.0,
                        max_elev_m=Hmax,
                        inland_limit_km=inland_limit_km,
                        tmp_vector_path=tmp_gpkg,
                        layer_name="flood",
                        simplify_m=simplify_m_stream,
                        min_area_m2=polygon_min_area_m2,
                        mode=("w" if first else "a"),
                        min_preview_hits=min_preview_hits,
                        min_valid_pixels=min_valid_pixels,
                        return_union=False
                    )
                    first = False

                # Count features (informational)
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
                t1 = time.perf_counter()
                logging.info(f"[BAND {band}] Dissolve wall time: {(t1 - t0):.2f}s (features={n})")

                if not keep_temp:
                    try:
                        os.remove(tmp_gpkg)
                    except OSError:
                        pass
                else:
                    logging.info(f"[KEEP_TEMP] Preserved temp flood layer: {tmp_gpkg}")

            else:
                # FAST MODE: do not write temp files; union per-tile in memory
                tile_geoms = []
                for _, t in tiles.iterrows():
                    g_tile = stream_flood_polygons_from_dem(
                        dem_path=t["path"],
                        strip_wgs=strip_wgs,
                        min_elev_m=-1.0,
                        max_elev_m=Hmax,
                        inland_limit_km=inland_limit_km,
                        tmp_vector_path=None,
                        simplify_m=simplify_m_stream,
                        min_area_m2=polygon_min_area_m2,
                        min_preview_hits=min_preview_hits,
                        min_valid_pixels=min_valid_pixels,
                        return_union=True
                    )
                    if g_tile is not None and (not getattr(g_tile, "is_empty", False)):
                        tile_geoms.append(g_tile)

                if tile_geoms:
                    try:
                        dissolved = unary_union(tile_geoms)
                    except Exception:
                        dissolved = unary_union([x for x in tile_geoms if (x is not None and not x.is_empty)])

            # Choose dissolved or fallback; normalize to polygonal; optional sector clip
            if (dissolved is None) or (hasattr(dissolved, "is_empty") and dissolved.is_empty):
                logging.warning(f"[BAND {band}] No dissolved geometry produced. Using fallback buffer.")
                poly_wgs = coast_line.buffer(1609, cap_style=2).to_crs("EPSG:4326").iloc[0]
            else:
                poly_wgs = _normalize_to_polygonal(dissolved, crs="EPSG:4326")
                if poly_wgs is None or getattr(poly_wgs, "is_empty", True):
                    logging.warning(f"[BAND {band}] No polygonal dissolved geometry; using fallback buffer.")
                    poly_wgs = coast_line.buffer(1609, cap_style=2).to_crs("EPSG:4326").iloc[0]

            # one-time sector wedge clip
            if sector_source_pt is not None and not poly_wgs.is_empty:
                poly_wgs = sector_clip(poly_wgs, strip_wgs.centroid, sector_source_pt,
                                       half_angle_deg=60, max_range_km=inland_limit_km)

            _diagnose_geom(f"dissolved-{band}", poly_wgs)
            try:
                area_m2 = gpd.GeoSeries([poly_wgs], crs='EPSG:4326').to_crs('EPSG:3857').area.iloc[0]
                logging.info(f"[BAND {band}] Area≈{area_m2:,.0f} m²")
            except Exception as e:
                logging.debug(f"[BAND {band}] area calc skipped: {e}")

            results.append({
                "event_id": event_id, "band": band,
                "ru_center": ru_center, "ru_low": ru_low, "ru_high": ru_high,
                "num_points": int(len(snapped)),
                "method": ("runup_IDW + DEM_threshold(binary) [fast]" if not stream_to_disk
                           else "runup_IDW + DEM_threshold(binary) [disk]"),
                "geometry": poly_wgs
            })

    # --- Fallback: no DEM (or no tiles) → slope-based buffers ---
    if not results:
        slope = 0.015  # 1.5% nominal coastal slope
        D_med_km = float(np.clip(ru_center / max(slope, 1e-3) / 1000.0, 0.5, inland_limit_km))
        D_low_km = max(0.25, 0.8 * D_med_km)
        D_high_km = min(inland_limit_km, 1.2 * D_med_km)

        line_m = coast_line.to_crs("EPSG:3857").iloc[0]
        for band_name, dist_km in [("LOW", D_low_km), ("MED", D_med_km), ("HIGH", D_high_km)]:
            poly_m = line_m.buffer(dist_km * 1000, cap_style=2)
            poly = gpd.GeoSeries([poly_m], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
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

    return gpd.GeoDataFrame(results, crs="EPSG:4326")


def _ensure_naive_datetime(series: pd.Series) -> pd.Series:
    """Coerce to pandas datetime64[ns] and strip timezone (some drivers reject tz-aware)."""
    s = pd.to_datetime(series, errors="coerce")
    try:
        # pandas Series.dt only exists for datetime dtype
        return s.dt.tz_localize(None)
    except Exception:
        return s


def _prepare_for_write(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = gdf.copy()

    # ensure active geometry + CRS
    g = _activate_geometry_column(g, crs="EPSG:4326")

    # drop empties/None up front
    g = g[g.geometry.notna() & (~g.geometry.is_empty)].copy()
    if g.empty:
        return g

    # try to ensure 2D (drop Z/M) & validity
    def _fix_geom(geom):
        if geom is None or geom.is_empty:
            return None
        # 2.5D → 2D
        try:
            if hasattr(geom, "has_z") and geom.has_z:
                from shapely.ops import transform
                _2d = transform(lambda x, y, z=None: (x, y), geom)
                geom = _2d
        except Exception:
            pass
        # validity & precision
        try:
            from shapely import set_precision
            geom = set_precision(geom.buffer(0), grid_size=1e-8)
        except Exception:
            geom = geom.buffer(0)
        return None if geom.is_empty else geom

    g["geometry"] = g.geometry.apply(_fix_geom)
    g = g[g.geometry.notna() & (~g.geometry.is_empty)]
    # normalize datetime columns if present
    if "date" in g.columns:
        g["date"] = _ensure_naive_datetime(g["date"])
    return g


def _write_vector(gdf: gpd.GeoDataFrame, path: str, driver: str, layer: str):
    g = _prepare_for_write(gdf)
    _log_geom_mix("pre-only-polygons", g)
    if driver in ("FileGDB", "GPKG"):
        g = _only_polygons(g)
    _log_geom_mix("pre-write", g)

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
            logging.warning(f"[fiona] primary GDB write failed, retrying with lower precision: {e}")
            try:
                from shapely import set_precision
                g2 = g.copy()
                g2["geometry"] = g2.geometry.apply(lambda x: set_precision(x.buffer(0), 1e-6))
                g2.to_file(path, driver=driver, layer=layer, engine="fiona",
                           TARGET_ARCGIS_VERSION="ARCGIS_PRO_3_2_OR_LATER", METHOD="SKIP")
                logging.info(f"[fiona] wrote (precision-reduced) {path} (driver={driver}, layer={layer})")
                return
            except Exception as e2:
                # write a debug GPKG for inspection before raising
                try:
                    dbg = os.path.splitext(path)[0] + "__debug.gpkg"
                    g.to_file(dbg, driver="GPKG", layer=f"{layer}_debug")
                    logging.info(f"[debug] wrote fallback debug GPKG: {dbg}")
                except Exception as e3:
                    logging.error(f"[debug] could not write debug GPKG: {e3}")
                logging.error(f"[fiona] GDB write retry failed: {e2}")
                raise

    # pyogrio for GPKG or other
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
            # write a debug GPKG for inspection
            try:
                dbg = os.path.splitext(path)[0] + "__debug.gpkg"
                g.to_file(dbg, driver="GPKG", layer=f"{layer}_debug")
                logging.info(f"[debug] wrote fallback debug GPKG: {dbg}")
            except Exception as e3:
                logging.error(f"[debug] could not write debug GPKG: {e3}")
            raise


# ------------------------------------------------------------
# Top-level tsunami processor (optional parallel)
# ------------------------------------------------------------
@profile
def process_tsunami_data(
    tsunami_events_csv,
    tsunami_runups_csv,
    countries_path,
    dem_dir,  # retained for signature compatibility (unused)
    inland_limit_km=10,
    band_percents=(0.2, 0.2),
    use_dem: bool = True,
    dem_local_root: str | None = None,
    dem_tile_size_deg: int = 5,
    output_folder: str | None = None,
    tmp_dir: str | None = None,
    # NEW controls
    event_filter: list | set | None = None,     # process only these event IDs
    write_per_event: bool = False,              # write each (event, ISO) as it's produced
    per_event_dir: str | None = None,           # defaults to <output_folder>/tsunami_events
    per_event_format: str = "gpkg",             # "gpkg" or "gdb"
    per_event_layer: str = "tsunami",
    write_aggregate: bool = False,              # also write a combined file at the end
    aggregate_path: str | None = None,          # defaults to <output_folder>/tsunamis_all.gpkg
    # Performance knobs
    stream_to_disk: bool = False,               # False = fast in-memory unions
    polygon_min_area_m2: float = 5000.0,
    min_preview_hits: int = 8,
    min_valid_pixels: int = 200,
    simplify_m_stream: float = 0.0,
    n_workers: int = 1                          # >1 to parallelize (experimental)
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

    # defaults for writing
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
        crs="EPSG:4326"
    )
    runups_gdf = _activate_geometry_column(runups_gdf, crs="EPSG:4326")

    # countries (clean)
    try:
        countries = gpd.read_file(countries_path).to_crs("EPSG:4326")
        countries = _activate_geometry_column(countries, crs="EPSG:4326")
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

        if ISO_COL not in countries.columns:
            logging.error(f"Countries layer missing '{ISO_COL}'.")
            return gpd.GeoDataFrame()
    except Exception:
        logging.exception("Failed loading countries for tsunamis")
        return gpd.GeoDataFrame()

    # ISO assignment: intersects + nearest≤50km fallback
    try:
        # intersects
        runups_iso = gpd.sjoin(
            runups_gdf, countries[[ISO_COL, "geometry"]],
            how="left", predicate="intersects"
        ).rename(columns={ISO_COL: "iso_a3"}).drop(columns="index_right")
        runups_iso = _activate_geometry_column(runups_iso, crs="EPSG:4326")

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
                    # fallback: nearest to representative points
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
        return gpd.GeoDataFrame()

    # event dates
    def _mk_date(row):
        try:
            y = int(row.get('year')) if pd.notna(row.get('year')) else None
            m = int(row.get('month')) if pd.notna(row.get('month')) else 1
            d = int(row.get('day')) if pd.notna(row.get('day')) else 1
            return pd.to_datetime(f"{y}-{m}-{d}", errors='coerce') if y else pd.NaT
        except Exception:
            return pd.NaT

    events_df['date'] = events_df.apply(_mk_date, axis=1)

    # Helper to run one (event, ISO) group
    def _run_group(event_id_val, iso_val, grp_df):
        coast_sel = countries[countries[ISO_COL] == iso_val]
        if coast_sel.empty:
            logging.warning(f"No coast polygon for ISO {iso_val} (event {event_id_val}).")
            return gpd.GeoDataFrame()

        inund = build_tsunami_inundation(
            runups_gdf=grp_df[['latitude', 'longitude', 'runupHt', 'tsunamiEventId', 'geometry']],
            coast_gdf=coast_sel,
            event_id=event_id_val,
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
            return inund

        ev = events_df[events_df['id'] == event_id_val]
        inund['iso_a3'] = iso_val
        inund['event_type'] = 'tsunami'
        inund['date'] = ev.iloc[0]['date'] if not ev.empty else pd.NaT
        return inund

    # iterate per (event_id, iso)
    results = []
    grouped = runups_iso.groupby(['tsunamiEventId', 'iso_a3'], dropna=False)

    # SERIAL by default (n_workers=1)
    if n_workers <= 1:
        for (event_id_val, iso_val), grp in grouped:
            if pd.isna(event_id_val) or pd.isna(iso_val):
                continue
            if event_filter is not None and event_id_val not in event_filter:
                continue

            # ensure grp is a proper GeoDataFrame with an active geometry
            if not isinstance(grp, gpd.GeoDataFrame):
                grp = gpd.GeoDataFrame(grp, geometry="geometry", crs="EPSG:4326")
            grp = _activate_geometry_column(grp, crs="EPSG:4326")
            logging.info(f"Processing event {int(event_id_val)} / ISO {iso_val} with {len(grp)} runups")
            inund = _run_group(event_id_val, iso_val, grp)
            if inund.empty:
                continue

            # per-event write
            if write_per_event:
                if per_event_format == "gpkg":
                    out_path = os.path.join(per_event_dir, f"tsunami_{int(event_id_val)}_{iso_val}.gpkg")
                    driver = "GPKG"; layer = per_event_layer
                else:
                    gdb_name = f"tsunami_{int(event_id_val)}_{iso_val}.gdb"
                    out_path = os.path.join(per_event_dir, gdb_name)
                    driver = "FileGDB"; layer = per_event_layer
                    try:
                        if "FileGDB" not in getattr(fiona, "supported_drivers", {}):
                            logging.warning("FileGDB writer unavailable; writing GPKG instead.")
                            out_path = os.path.join(per_event_dir, f"tsunami_{int(event_id_val)}_{iso_val}.gpkg")
                            driver = "GPKG"
                    except Exception:
                        logging.warning("Could not verify Fiona drivers; using GPKG for safety.")
                        out_path = os.path.join(per_event_dir, f"tsunami_{int(event_id_val)}_{iso_val}.gpkg")
                        driver = "GPKG"

                try:
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    _write_vector(inund, out_path, driver=driver, layer=layer)
                    logging.info(f"Wrote per-event tsunami: {out_path} (layer={layer}, driver={driver})")
                except Exception as e:
                    logging.error(f"Failed writing per-event output for event {event_id_val}, {iso_val}: {e}")
                    # debug dump
                    try:
                        dbg = os.path.join(per_event_dir, f"__debug_{int(event_id_val)}_{iso_val}.gpkg")
                        gpd.GeoDataFrame(inund, geometry="geometry", crs="EPSG:4326").to_file(
                            dbg, driver="GPKG", layer="debug"
                        )
                        logging.info(f"[debug] wrote {dbg}")
                    except Exception as e2:
                        logging.error(f"[debug] could not write debug gpkg: {e2}")

            results.append(inund)

    else:
        # PARALLEL (experimental): to avoid heavy pickles, run serially for writing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Prepare tasks as small dataframes (grp.copy()) – expect moderate sizes per group
        tasks = [((int(eid), iso), grp.copy()) for (eid, iso), grp in grouped
                 if (pd.notna(eid) and pd.notna(iso) and (event_filter is None or eid in event_filter))]

        logging.info(f"Dispatching {len(tasks)} (event, ISO) groups across {n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = []
            for (eid_iso, grp_df) in tasks:
                eid, iso_val = eid_iso
                futs.append(ex.submit(_run_group, eid, iso_val, grp_df))
            for f in as_completed(futs):
                try:
                    res = f.result()
                    if not res.empty:
                        results.append(res)
                except Exception as e:
                    logging.error(f"Worker failed: {e}")

        # Per-event writes (do AFTER gather to avoid file locks)
        if write_per_event:
            for inund in results:
                eid = int(inund["event_id"].iloc[0])
                iso_val = inund["iso_a3"].iloc[0]
                if per_event_format == "gpkg":
                    out_path = os.path.join(per_event_dir, f"tsunami_{eid}_{iso_val}.gpkg")
                    driver = "GPKG"; layer = per_event_layer
                else:
                    gdb_name = f"tsunami_{eid}_{iso_val}.gdb"
                    out_path = os.path.join(per_event_dir, gdb_name)
                    driver = "FileGDB"; layer = per_event_layer
                    try:
                        if "FileGDB" not in getattr(fiona, "supported_drivers", {}):
                            logging.warning("FileGDB writer unavailable; writing GPKG instead.")
                            out_path = os.path.join(per_event_dir, f"tsunami_{eid}_{iso_val}.gpkg")
                            driver = "GPKG"
                    except Exception:
                        logging.warning("Could not verify Fiona drivers; using GPKG for safety.")
                        out_path = os.path.join(per_event_dir, f"tsunami_{eid}_{iso_val}.gpkg")
                        driver = "GPKG"

                try:
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    _write_vector(inund, out_path, driver=driver, layer=layer)
                    logging.info(f"Wrote per-event tsunami: {out_path} (layer={layer}, driver={driver})")
                except Exception as e:
                    logging.error(f"Failed writing per-event output for event {eid}, {iso_val}: {e}")
                    # debug dump
                    try:
                        dbg = os.path.join(per_event_dir, f"__debug_{eid}_{iso_val}.gpkg")
                        gpd.GeoDataFrame(inund, geometry="geometry", crs="EPSG:4326").to_file(
                            dbg, driver="GPKG", layer="debug"
                        )
                        logging.info(f"[debug] wrote {dbg}")
                    except Exception as e2:
                        logging.error(f"[debug] could not write debug gpkg: {e2}")

    # nothing produced
    if not results:
        logging.warning("No tsunami inundation polygons produced.")
        return gpd.GeoDataFrame()

    # aggregate in-memory
    out = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs="EPSG:4326")
    out = _activate_geometry_column(out, crs="EPSG:4326")
    logging.info(f"Produced {len(out)} tsunami polygons (combined).")

    # optional combined write
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
