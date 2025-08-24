# ----
# helpers.py (constants, logging, geometry helpers, optional ISO/time joins, GDB writers)
# ----
import os, sys, logging, psutil, math
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Optional
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import transform as shp_transform, unary_union
from shapely.validation import explain_validity, make_valid
from pathlib import Path
from pyogrio.errors import DataSourceError as _PgDataSourceError

try:
    import pyogrio
    HAVE_PYOGRIO = True
except Exception:
    HAVE_PYOGRIO = False

try:
    from shapely import set_precision as shp_set_precision
    HAVE_PRECISION = True
except Exception:
    HAVE_PRECISION = False

# ---- constants ----
NM_TO_KM = 1.852
AGENCY_PREF = ['USA','BOM','REUNION','TOKYO','CMA','HKO','KMA','NADI']
RAD_TIERS   = ['R64','R50','R34']
QUADS       = ['NE','SE','SW','NW']
WGS84 = "EPSG:4326"
CANON_COLS = [
    "event_id", "event_type", "start_time", "end_time",
    "band", "geom_method", "geom_confidence", "area_km2", "geometry"
]

# Natural Earth: EU-harmonized ISO3, avoids pseudo-codes like CH1/FR1
ISO_COL = "ISO_A3_EH"
ISO_NORMALIZE = {
    "UKM": "GBR",   # keep a small normalizer for rare oddities
}

# crude climatology (kept here so storms can import)
CLIMO_R34_NM = {
    "ATL": {0:60, 1:75, 2:90, 3:110, 4:120, 5:130},
    "EP":  {0:55, 1:70, 2:85, 3:100, 4:115, 5:125},
    "WP":  {0:60, 1:80, 2:95, 3:115, 4:130, 5:140},
    "SI":  {0:55, 1:70, 2:85, 3:105, 4:120, 5:130},
    "SP":  {0:55, 1:70, 2:85, 3:100, 4:115, 5:125},
    "NI":  {0:50, 1:65, 2:80, 3:95,  4:110, 5:120},
}
SSHS_SCALE_K = {0:1.8, 1:2.0, 2:2.2, 3:2.4, 4:2.6, 5:2.8}  # rmw→storm-size factor

# ---- logging ----
def setup_logging(output_folder: str, level=logging.INFO) -> None:
    """Console + file logging, safe for repeat calls."""
    logger = logging.getLogger()
    logger.setLevel(level)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt); ch.setLevel(level)
    logger.addHandler(ch)

    log_dir = os.path.join(output_folder, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    fh_path = os.path.join(log_dir, 'pipeline.log')
    fh = logging.FileHandler(fh_path, encoding='utf-8')
    fh.setFormatter(fmt); fh.setLevel(level)
    logger.addHandler(fh)

    logging.info(f"Logging to file: {fh_path}")

def log_mem(note=""):
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss / (1024**3)
    logging.info(f"[MEM] {note} RSS={rss:.2f} GB")

# ---- pathing ----
# Base folder for local datasets (default: ./data). Can override via env var.
DATA_DIR = Path(os.getenv("NATDIS_DATA_DIR", "data")).expanduser().resolve()

def data_path(*parts: str) -> Path:
    """
    Build a path inside the data directory. Accepts either "a/b" or "a\\b".
    Example: data_path("Top 5 Percent EMDAT.csv")
             data_path("DFO", "FloodArchive_region.shp")
    """
    p = Path(parts[0])
    for q in parts[1:]:
        p = p / q
    return (DATA_DIR / p).resolve()

# Optional: outputs go here by default
OUTPUT_DIR = Path(os.getenv("NATDIS_OUTPUT_DIR", "disaster_output")).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def output_path(*parts: str) -> Path:
    p = Path(parts[0])
    for q in parts[1:]:
        p = p / q
    return (OUTPUT_DIR / p).resolve()

# ---- small utils ----
def bounds_ok(lat: float, lon: float) -> bool:
    return (-90 <= lat <= 90) and (-180 <= lon <= 180)

def clean_radius(val):
    try:
        v = float(val)
        if np.isfinite(v) and v > 0:
            return v
    except Exception:
        pass
    return np.nan

def first_positive(row, names):
    for n in names:
        if n in row:
            v = clean_radius(row[n])
            if pd.notna(v):
                return v
    return np.nan

def _geom_present_mask(obj) -> pd.Series:
    """
    Geometry-present mask without calling GeoSeries.notna() to avoid warnings.
    Treats 'present' as: geometry is not None AND not empty.
    """
    s = obj.geometry if isinstance(obj, gpd.GeoDataFrame) else obj
    return s.apply(lambda g: (g is not None) and (getattr(g, "is_empty", True) is False))


def build_circle(lat, lon, radius_nm, points=180) -> Optional[Polygon]:
    """Approximate circle (WGS84 degrees) centered at (lat,lon); radius in nautical miles."""
    if pd.isna(radius_nm) or radius_nm <= 0:
        return None
    r_km = radius_nm * NM_TO_KM
    ang = np.linspace(0, 2*np.pi, points, endpoint=False)
    coslat = np.cos(np.radians(lat))
    denom = (111.0 * coslat) if coslat != 0 else 111.0
    xs = lon + (r_km * np.cos(ang)) / denom
    ys = lat + (r_km * np.sin(ang)) / 111.0
    try:
        return Polygon(list(zip(xs, ys)))
    except Exception:
        return None

def _coords_ok(g):
    import numpy as np
    if g is None or g.is_empty:
        return False
    try:
        # Shapely 2.x
        from shapely import get_coordinates
        A = get_coordinates(g)
    except Exception:
        # Fallback for Shapely 1.8
        def _poly_coords(poly):
            from itertools import chain
            xs = list(poly.exterior.coords)
            for ring in poly.interiors:
                xs += list(ring.coords)
            return xs
        import numpy as np
        if g.geom_type == "Polygon":
            A = np.array(_poly_coords(g), dtype=float)
        elif g.geom_type == "MultiPolygon":
            pts = []
            for p in g.geoms:
                pts.extend(_poly_coords(p))
            A = np.array(pts, dtype=float) if pts else np.empty((0,2))
        else:
            try:
                A = np.array(list(g.coords), dtype=float)
            except Exception:
                A = np.empty((0,2))

    if A.size == 0:
        return False
    if not np.isfinite(A).all():
        return False
    x, y = A[:, 0], A[:, 1]
    return (np.all(x >= -180.000001) and np.all(x <= 180.000001)
            and np.all(y >= -90.000001) and np.all(y <= 90.000001))

def _keep_surface_parts(g):
    """
    Keep polygonal parts only. Returns Polygon/MultiPolygon or None.
    """
    if g is None or getattr(g, "is_empty", True):
        return None
    gt = getattr(g, "geom_type", "")
    if gt in ("Polygon", "MultiPolygon"):
        return g
    if gt == "GeometryCollection":
        parts = [p for p in g.geoms if getattr(p, "geom_type", "") in ("Polygon", "MultiPolygon")]
        if not parts:
            return None
        try:
            return unary_union(parts)
        except Exception:
            # fallback: pairwise unions with buffer(0) rescue
            u = parts[0]
            for p in parts[1:]:
                try:
                    u = u.union(p)
                except Exception:
                    try:
                        u = u.buffer(0).union(p.buffer(0))
                    except Exception:
                        pass
            return u
    return None


def _safe_clean(g):
    """
    Make geometry valid, strip to polygonal surfaces, and robustify with buffer(0).
    Returns Polygon/MultiPolygon or None.
    """
    if g is None or getattr(g, "is_empty", True):
        return None
    try:
        if not g.is_valid:
            g = make_valid(g)
        g = _keep_surface_parts(g)
        if g is None or g.is_empty:
            return None
        g = g.buffer(0)
        if g is None or g.is_empty:
            return None
        return _keep_surface_parts(g)
    except Exception:
        try:
            gg = g.buffer(0)
            return None if (gg is None or gg.is_empty) else _keep_surface_parts(gg)
        except Exception:
            return None


def _safe_unary_union(geoms):
    """
    Clean each geometry (valid + polygonal) then unary_union with robust fallbacks.
    Returns Polygon/MultiPolygon or None.
    """
    cleaned = []
    for g in geoms:
        gg = _safe_clean(g)
        if gg is not None and not gg.is_empty:
            cleaned.append(gg)
    if not cleaned:
        return None
    try:
        return unary_union(cleaned)
    except Exception:
        u = cleaned[0]
        for g in cleaned[1:]:
            try:
                u = u.union(g)
            except Exception:
                try:
                    u = u.buffer(0).union(g.buffer(0))
                except Exception:
                    continue
        return u


def _union_polys(geoms):
    """
    Union only the polygonal members of `geoms`, after cleaning.
    Returns Polygon/MultiPolygon or None.
    """
    polys = []
    for g in geoms:
        if g is None or getattr(g, "is_empty", True):
            continue
        gt = getattr(g, "geom_type", "")
        if gt in ("Polygon", "MultiPolygon"):
            polys.append(_safe_clean(g))
    if not polys:
        return None
    return _safe_unary_union(polys)

def _clean_for_fgdb(gdf):
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon
    g = gdf.copy()

    # 1) ensure polygons & 2D only
    try:
        from shapely import wkb
        g["geometry"] = g.geometry.apply(
            lambda geom: wkb.loads(wkb.dumps(geom, output_dimension=2)) if geom and not geom.is_empty else geom
        )
    except Exception:
        pass

    # promote Polygon → MultiPolygon (FGDB likes consistent types)
    def _promote(mp):
        if mp is None or mp.is_empty:
            return mp
        if mp.geom_type == "Polygon":
            return MultiPolygon([mp])
        return mp
    g["geometry"] = g.geometry.apply(_promote)

    # 3) make valid
    try:
        from shapely.validation import make_valid
        g["geometry"] = g.geometry.apply(make_valid)
    except Exception:
        g["geometry"] = g.buffer(0)

    # 4) drop empties/invalids
    g = g[_geom_present_mask(g) & (g.geometry.is_valid)].copy()

    # 5) drop non-finite / out-of-range lon/lat
    bad = ~g.geometry.apply(_coords_ok)
    if bad.any():
        logging.warning(f"Dropping {int(bad.sum())} feature(s) with non-finite or out-of-range coordinates before GDB write.")
        g = g[~bad]

    # 6) optional precision trim to reduce parts & polygon re-org cost
    try:
        from shapely import set_precision
        g["geometry"] = g.geometry.apply(lambda geom: set_precision(geom, grid_size=1e-7))
    except Exception:
        pass

    # 7) nudge dtypes for ArcGIS (avoid int64→float64)
    for col in ("event_id", "num_points"):
        if col in g.columns:
            if np.issubdtype(g[col].dtype, np.integer):
                # If fits in int32, downcast; else let TARGET_ARCGIS_VERSION handle int64
                if g[col].dropna().empty:
                    continue
                vmin, vmax = g[col].min(), g[col].max()
                if vmin >= -2147483648 and vmax <= 2147483647:
                    g[col] = g[col].astype("int32")

    # enforce CRS
    if g.crs is None or str(g.crs) != "EPSG:4326":
        g = g.to_crs("EPSG:4326")

    return g

def extract_polygons_only(geom):
    if geom is None:
        return None
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if g.geom_type in ['Polygon','MultiPolygon']]
        if not polys:
            return None
        return polys[0] if len(polys)==1 else MultiPolygon(polys)
    elif getattr(geom, "geom_type", None) in ['Polygon','MultiPolygon']:
        return geom
    return None

def ensure_multipolygon(geom):
    if geom is None or getattr(geom, "is_empty", True):
        return None
    gt = getattr(geom, "geom_type", "")
    if gt == 'Polygon':
        return MultiPolygon([geom])
    if gt == 'MultiPolygon':
        return geom
    return None

# ---- geometry build logging ----
def log_polygon_failure(kind: str, sid: str, reason: str, output_folder: str):
    logging.error(f"{kind} polygon failed for {sid}: {reason}")
    os.makedirs(os.path.join(output_folder, "fallback_layers"), exist_ok=True)
    csv_path = os.path.join(output_folder, "fallback_layers", f"skipped_{kind.lower()}s.csv")
    import csv
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ID","reason"])
        if not file_exists:
            w.writeheader()
        w.writerow({"ID": sid, "reason": reason})

def diagnose_geom(geom):
    if geom is None:
        return "geom_none"
    if geom.is_empty:
        return "geom_empty"
    if not geom.is_valid:
        return f"geom_invalid:{explain_validity(geom)}"
    if geom.area == 0:
        return "geom_zero_area"
    return "geom_ok"

# ---- optional time helpers ----
def assign_decade(dt):
    if pd.isna(dt):
        return "unknown"
    y = dt.year if isinstance(dt, pd.Timestamp) else (int(dt) if pd.notna(dt) else None)
    if not y:
        return "unknown"
    return f"{y - (y % 10)}s"

def has_temporal_overlap(a_start, a_end, b_start, b_end, tolerance_days=7):
    """Closed-interval overlap with ±tolerance on B."""
    if pd.isna(a_start) or pd.isna(a_end) or pd.isna(b_start) or pd.isna(b_end):
        return False
    b_start_ext = b_start - pd.Timedelta(days=tolerance_days)
    b_end_ext   = b_end   + pd.Timedelta(days=tolerance_days)
    return max(a_start, b_start_ext) <= min(a_end, b_end_ext)

def filter_by_emdat_overlap(
    gdf: gpd.GeoDataFrame,
    emdat_df: pd.DataFrame,
    start_col: str,
    end_col: Optional[str] = None,
    emdat_start_col: str = "start_date",
    emdat_end_col: str = "end_date",
    tolerance_days: int = 7,
) -> gpd.GeoDataFrame:
    """
    Return only features whose [start_col,end_col or start_col] overlaps any EM-DAT record.
    If end_col is None, treat start_col as instantaneous (use same for start/end).
    """
    if gdf.empty or emdat_df.empty or start_col not in gdf.columns:
        return gdf

    endc = end_col if end_col and end_col in gdf.columns else start_col

    keep_idx = []
    for i, r in gdf.iterrows():
        a_start, a_end = r[start_col], r[endc]
        overlaps = emdat_df.apply(
            lambda e: has_temporal_overlap(a_start, a_end, e.get(emdat_start_col), e.get(emdat_end_col), tolerance_days),
            axis=1
        )
        if overlaps.any():
            keep_idx.append(i)
    return gdf.loc[keep_idx].copy()

# ---- spatial join helpers ----
def normalize_iso(a3: str) -> str:
    if not isinstance(a3, str): return "UNK"
    a3 = a3.strip().upper()
    return ISO_NORMALIZE.get(a3, a3)

def spatial_join_iso_single(gdf: gpd.GeoDataFrame, countries: gpd.GeoDataFrame, iso_col: str, label: str):
    if gdf is None or gdf.empty:
        logging.info(f"{label}: nothing to write (GeoDataFrame empty).")
        return gpd.GeoDataFrame()
    if gdf.crs != countries.crs:
        gdf = gdf.to_crs(countries.crs)
    joined = gpd.sjoin(gdf, countries[[iso_col, "geometry"]], how="inner", predicate="intersects")
    joined = joined.drop(columns="index_right").rename(columns={iso_col: "iso_a3"})
    joined["iso_a3"] = joined["iso_a3"].apply(normalize_iso)
    logging.info(f"{label}: Spatial join assigned ISO codes to {len(joined)} features.")
    return joined

def spatial_join_iso(gdf, countries, iso_col, label):
    if gdf is None or gdf.empty:
        logging.warning(f"{label} GeoDataFrame is empty.")
        return gpd.GeoDataFrame()
    try:
        if gdf.crs != countries.crs:
            gdf = gdf.to_crs(countries.crs)
        joined = gpd.sjoin(gdf, countries[[iso_col, "geometry"]], how="inner", predicate="intersects")
        joined = joined.drop(columns="index_right").rename(columns={iso_col: "iso_a3"})
        joined["iso_a3"] = joined["iso_a3"].apply(normalize_iso)
        logging.info(f"{label}: Spatial join assigned ISO codes to {len(joined)} features.")
        return joined
    except Exception as e:
        logging.error(f"{label}: Spatial join failed: {e}")
        gdf = gdf.copy()
        gdf["iso_a3"] = "UNK"
        return gdf

def _wrap_lonlat_geom(geom):
    if geom is None or geom.is_empty:
        return geom
    def _f(x, y, z=None):
        if not (math.isfinite(x) and math.isfinite(y)):
            return (np.nan, np.nan) if z is None else (np.nan, np.nan, z)
        x2 = ((x + 180.0) % 360.0) - 180.0
        y2 = max(-90.0, min(90.0, y))
        return (x2, y2) if z is None else (x2, y2, z)
    return shp_transform(_f, geom)

def _bounds_bad(bdf):
    minx, miny, maxx, maxy = bdf[['minx','miny','maxx','maxy']].T.values
    finite = np.isfinite(minx) & np.isfinite(miny) & np.isfinite(maxx) & np.isfinite(maxy)
    oob = (minx < -180.001) | (maxx > 180.001) | (miny < -90.001) | (maxy > 90.001)
    return (~finite) | oob

def _first_nonnull(series, prefer=None):
    vals = series.dropna()
    if prefer is not None:
        for p in prefer:
            m = (series.index.get_level_values(0) == p) if hasattr(series.index, "get_level_values") else None
    return vals.iloc[0] if not vals.empty else None

def _union_polys(geoms):
    polys = []
    for g in geoms:
        if g is None or getattr(g, "is_empty", True):
            continue
        gt = getattr(g, "geom_type", "")
        if gt in ("Polygon", "MultiPolygon"):
            polys.append(_safe_clean(g))
    if not polys:
        return None
    return _safe_unary_union(polys)

# --- add in helpers.py (near other utils) -------------------------------
def _prepare_fields_for_ogr(df: gpd.GeoDataFrame, prefer: set | None = None):
    """
    Return (df2, rename_map) with column names made OGR-safe and
    case-insensitively unique. Canonical columns in `prefer` keep their names;
    conflicting "raw" columns get a suffix like `_raw`, then `_raw2`, etc.
    """
    import re
    if prefer is None:
        prefer = set(CANON_COLS)

    cols = list(df.columns)
    rename = {}
    used_lower = set()

    def sanitize(name: str) -> str:
        # keep geometry untouched
        if name == "geometry":
            return name
        s = str(name).strip()
        # collapse whitespace & punctuation
        s = re.sub(r"[^\w]+", "_", s)          # non [0-9A-Za-z_]
        s = re.sub(r"_+", "_", s).strip("_")   # dedupe underscores
        if not s:
            s = "field"
        if s[0].isdigit():
            s = "f_" + s
        # GPKG is fine with long names, but keep reasonable
        if len(s) > 60:
            s = s[:60]
        return s

    # First pass: propose sanitized names
    proposed = {c: sanitize(c) for c in cols}

    # Second pass: enforce case-insensitive uniqueness with preference
    for c in cols:
        if c == "geometry":
            continue
        base = proposed[c]
        key = base.lower()

        # If name already taken in a case-insensitive way:
        if key in used_lower:
            # keep canonical column names if they collide
            if c in prefer:
                # find a unique variant with numeric suffix
                i = 2
                base_i = base
                while (base_i.lower() in used_lower):
                    base_i = f"{base}_{i}"
                    i += 1
                base = base_i
            else:
                # raw column colliding with (likely) canonical; tag as _raw (then _raw2, ...)
                base_i = f"{base}_raw"
                i = 2
                while (base_i.lower() in used_lower):
                    base_i = f"{base}_raw{i}"
                    i += 1
                base = base_i

        used_lower.add(base.lower())
        rename[c] = base

    df2 = df.rename(columns=rename)
    # log a tiny hint in case you need to trace names later
    try:
        collisions = {k: v for k, v in rename.items() if k != v and k != "geometry"}
        if collisions:
            logging.info(f"Field rename(s) for OGR safety: {collisions}")
    except Exception:
        pass

    return df2, rename

def _gpkg_remove_layer(path: str, layer_name: str) -> bool:
    """
    Delete a layer from a GeoPackage if it exists. Returns True if a layer was removed.
    """
    try:
        import pyogrio
        from osgeo import ogr
    except Exception:
        return False

    try:
        names = pyogrio.list_layers(path)[0]  # returns (names, geometry_types[, ...])
    except Exception:
        names = []
    if layer_name not in names:
        return False

    try:
        ds = ogr.Open(path, update=1)
        if ds is None:
            return False
        # Find layer index by name
        for i in range(ds.GetLayerCount()):
            lyr = ds.GetLayerByIndex(i)
            if lyr is not None and lyr.GetName() == layer_name:
                ds.DeleteLayer(i)
                ds = None
                return True
        ds = None
        return False
    except Exception:
        return False

def write_gpkg(df: gpd.GeoDataFrame, path: str, layer_name: str, *, overwrite: bool = True, append: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # NEW: sanitize/dedupe fields before writing
    df, _ = _prepare_fields_for_ogr(df, prefer=set(CANON_COLS))

    if HAVE_PYOGRIO:
        try:
            lco = {"SPATIAL_INDEX": "YES"}
            if overwrite:
                lco = {"SPATIAL_INDEX": "YES", "OVERWRITE": "YES"}
            pyogrio.write_dataframe(df, path, driver="GPKG", layer=layer_name,
                                    layer_options=lco, append=bool(append and not overwrite))
            return
        except Exception:
            # Fallback: remove the layer, then write fresh
            if overwrite and os.path.exists(path) and _gpkg_remove_layer(path, layer_name):
                pyogrio.write_dataframe(df, path, driver="GPKG", layer=layer_name,
                                        layer_options={"SPATIAL_INDEX": "YES"})
                return
            raise
    else:
        if overwrite and os.path.exists(path):
            _gpkg_remove_layer(path, layer_name)
        df.to_file(path, driver="GPKG", layer=layer_name)

def write_single_hazard_gdb(
    gdf: gpd.GeoDataFrame,
    output_path: str,
    layer_name: str,
    fix_mode: str = "wrap",
    precision: float = 1e-7,
    verbose_geometry_logging: bool = False
):
    if gdf is None or gdf.empty:
        logging.warning("Nothing to write.")
        return

    df = gdf.copy()

    # enforce CRS
    if df.crs is None or str(df.crs) != WGS84:
        df = df.to_crs(WGS84)

    # drop empties early
    df = df[_geom_present_mask(df)].copy()

    # clean geometry: make_valid + optional precision snap
    try:
        df["geometry"] = df["geometry"].apply(make_valid)
        if precision and precision > 0 and HAVE_PRECISION:
            df["geometry"] = df["geometry"].apply(lambda g: shp_set_precision(g, precision))
    except Exception:
        df["geometry"] = df.buffer(0)

    # drop empties again if any
    df = df[_geom_present_mask(df)].copy()

    # wrap or drop out-of-range coords
    bad = _bounds_bad(df.bounds)
    if bad.any():
        if fix_mode == "wrap":
            idx = df.index[bad]
            df.loc[idx, "geometry"] = df.loc[idx, "geometry"].apply(_wrap_lonlat_geom)
            still_bad = _bounds_bad(df.loc[idx].bounds)
            if still_bad.any():
                df = df.drop(index=df.loc[idx[still_bad]].index)
        else:
            df = df.drop(index=df.index[bad])

    if verbose_geometry_logging:
        logging.info(f"{layer_name}: {len(df)} feature(s) after clean.")

    df, _ = _prepare_fields_for_ogr(df, prefer=set(CANON_COLS))

    # choose writer by extension
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".gdb":
        # Try FileGDB; fall back to GPKG if unavailable
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            lco = ["TARGET_ARCGIS_VERSION=ARCGIS_PRO_3_2_OR_LATER", "METHOD=SKIP"]
            if HAVE_PYOGRIO:
                pyogrio.write_dataframe(df, output_path, driver="FileGDB",
                                        layer=layer_name, layer_options=lco)
            else:
                df.to_file(output_path, driver="FileGDB", layer=layer_name)
            return
        except Exception as e:
            # inside write_single_hazard_gdb(), in the FileGDB exception fallback:
            fallback = output_path[:-4] + ".gpkg"
            logging.warning(f"FileGDB write failed ({e}); falling back to GPKG: {fallback}")
            try:
                write_gpkg(df, fallback, layer_name, overwrite=True)   # <— ensure overwrite
                logging.info(f"Wrote fallback GPKG: {fallback}")
                return
            except Exception as e2:
                logging.error(f"GPKG fallback also failed: {e2}")
                raise


    # Non-.gdb → write GPKG directly
    try:
        write_gpkg(df, output_path, layer_name)
    except Exception as e:
        logging.error(f"GPKG write failed for {output_path}: {e}")
        raise


# --- unifed additions
def _ensure_cols(gdf, cols):
    for c in cols:
        if c not in gdf.columns:
            gdf[c] = None
    return gdf

def _area_km2(geom):
    try:
        if geom is None or getattr(geom, "is_empty", True) or not getattr(geom, "is_valid", True):
            return None
        return gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:6933").area.iloc[0] / 1e6
    except Exception:
        return None

def unify_schema_storms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty: return gdf
    out = gdf.copy()
    if "SID" in out.columns:
        out = out.rename(columns={"SID": "event_id"})
    out["event_type"] = "storm"
    out = _ensure_cols(out, ["geom_method","geom_confidence","start_time","end_time","storm_date"])
    out["start_time"] = out["start_time"].fillna(out["storm_date"])
    out["end_time"]   = out["end_time"].fillna(out["storm_date"])
    out["band"] = None
    out["area_km2"] = out.geometry.apply(_area_km2)
    out = _ensure_cols(out, CANON_COLS)
    return out[CANON_COLS]

def unify_schema_quakes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty: return gdf
    out = gdf.copy()
    if "event_id" not in out.columns:
        out["event_id"] = (out["id"].astype(str) if "id" in out.columns else out.index.astype(str))
    out["event_type"] = "earthquake"
    dt = out.get("eq_date", out.get("date"))
    out["start_time"] = dt
    out["end_time"]   = dt
    out["band"] = None
    out["geom_method"] = None
    out["geom_confidence"] = None
    out["area_km2"] = out.geometry.apply(_area_km2)
    out = _ensure_cols(out, CANON_COLS)
    return out[CANON_COLS]

def unify_schema_tsunamis(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty: return gdf
    out = gdf.copy()
    out["event_type"] = "tsunami"
    out = _ensure_cols(out, ["start_time","end_time","band"])
    if "method" in out.columns and "geom_method" not in out.columns:
        out["geom_method"] = out["method"]
    out["geom_confidence"] = None
    out["area_km2"] = out.geometry.apply(_area_km2)
    out = _ensure_cols(out, CANON_COLS)
    return out[CANON_COLS]

# helpers.py
# helpers.py
def unify_schema_floods(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Non-destructive canonicalizer for floods:
    - Preserves ALL existing columns (incl. enriched fields).
    - Only fills canonical fields if missing.
    - Computes area_km2 ONLY where it's NaN.
    """
    if gdf is None or gdf.empty:
        return gdf if isinstance(gdf, gpd.GeoDataFrame) else gpd.GeoDataFrame(columns=CANON_COLS, geometry=[])

    out = gdf.copy()

    # event_type
    if "event_type" not in out.columns or out["event_type"].isna().all():
        out["event_type"] = "flood"

    # start/end time: never drop original *_date columns; just ensure canonical *_time exist
    st = pd.to_datetime(out.get("start_time"), errors="coerce")
    if "start_time" not in out.columns or st.isna().all():
        st2 = pd.to_datetime(out.get("start_date"), errors="coerce")
        out["start_time"] = st.fillna(st2)

    en = pd.to_datetime(out.get("end_time"), errors="coerce")
    if "end_time" not in out.columns or en.isna().all():
        en2 = pd.to_datetime(out.get("end_date"), errors="coerce")
        out["end_time"] = en.fillna(en2)

    # fill end_time with start_time when missing
    m = out["end_time"].isna() & out["start_time"].notna()
    if m.any():
        out.loc[m, "end_time"] = out.loc[m, "start_time"]

    # geom_method: only set from footprint_method if geom_method missing/empty
    if ("geom_method" not in out.columns) or out["geom_method"].isna().all():
        if "footprint_method" in out.columns:
            out["geom_method"] = out.get("geom_method", None)
            fill_idx = out["geom_method"].isna()
            if fill_idx.any():
                out.loc[fill_idx, "geom_method"] = out.loc[fill_idx, "footprint_method"]

    # defaults that don't overwrite
    if "band" not in out.columns:
        out["band"] = None
    if "geom_confidence" not in out.columns:
        out["geom_confidence"] = None

    # area_km2: compute equal-area only where NaN
    out["area_km2"] = pd.to_numeric(out.get("area_km2"), errors="coerce")
    try:
        need = out["area_km2"].isna()
        if need.any():
            areas = gpd.GeoSeries(out.geometry, crs="EPSG:4326").to_crs("EPSG:6933").area / 1e6
            out.loc[need, "area_km2"] = areas[need]
    except Exception:
        # leave as-is on failure; we don't overwrite existing values
        pass

    # ensure canonical columns exist
    for c in CANON_COLS:
        if c not in out.columns:
            out[c] = None

    # reorder: canonical first, then every other column (preserved)
    extras = [c for c in out.columns if c not in CANON_COLS]
    return out[CANON_COLS + extras]

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorized great-circle distance in km.
    Works for any mix of scalars/vectors by broadcasting all inputs
    to a common shape before masking.
    """
    R = 6371.0
    A1, B1, A2, B2 = np.broadcast_arrays(
        np.asarray(lat1, dtype=float),
        np.asarray(lon1, dtype=float),
        np.asarray(lat2, dtype=float),
        np.asarray(lon2, dtype=float),
    )
    out = np.full(A1.shape, np.nan, dtype=float)

    m = np.isfinite(A1) & np.isfinite(B1) & np.isfinite(A2) & np.isfinite(B2)
    if not np.any(m):
        return out

    p1 = np.radians(A1[m]); p2 = np.radians(A2[m])
    dlat = p2 - p1
    dlon = np.radians(B2[m] - B1[m])

    with np.errstate(invalid="ignore"):
        a = np.sin(dlat/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2.0)**2
        out[m] = 2 * R * np.arcsin(np.sqrt(a))
    return out

def _centroid_latlon(gdf: gpd.GeoDataFrame):
    if gdf.empty:
        return (np.array([]), np.array([]))
    g = gdf
    if g.crs is None or str(g.crs) != "EPSG:4326":
        g = g.to_crs("EPSG:4326")
    pts = g.geometry.representative_point()   # safer than centroid
    pts = gpd.GeoSeries(pts, crs=g.crs).to_crs("EPSG:4326")
    return np.asarray(pts.y), np.asarray(pts.x)

def spatial_temporal_filter(
    gdf: gpd.GeoDataFrame,
    em: pd.DataFrame,
    gdf_start: str, gdf_end: str,
    em_start: str = "start_date", em_end: str = "end_date",
    gdf_lat: str = None, gdf_lon: str = None,
    em_lat: str = "em_lat", em_lon: str = "em_lon",
    max_km: float = 250.0
) -> gpd.GeoDataFrame:
    """Keep rows whose time windows overlap any EM-DAT row and (if lat/lon available) are within max_km."""
    if gdf.empty or em.empty: return gdf
    g = gdf.copy()
    g[gdf_start] = pd.to_datetime(g[gdf_start], errors="coerce")
    g[gdf_end]   = pd.to_datetime(g[gdf_end], errors="coerce")
    em = em.copy()
    em[em_start] = pd.to_datetime(em[em_start], errors="coerce")
    em[em_end]   = pd.to_datetime(em[em_end], errors="coerce")

    if gdf_lat is None or gdf_lon is None:
        clat, clon = _centroid_latlon(g)
    else:
        clat = pd.to_numeric(g[gdf_lat], errors="coerce").values
        clon = pd.to_numeric(g[gdf_lon], errors="coerce").values

    if em_lat not in em.columns:
        for a in ["em_lat","latitude","lat","LAT","Lat"]:
            if a in em.columns: em_lat = a; break
    if em_lon not in em.columns:
        for o in ["em_lon","longitude","lon","LON","Lon","LONGITUDE"]:
            if o in em.columns: em_lon = o; break

    keep_idx = []
    have_em_ll = (em_lat in em.columns) and (em_lon in em.columns)
    for i, row in g.iterrows():
        st, et = row[gdf_start], row[gdf_end]
        if pd.isna(st) or pd.isna(et): continue
        overlaps = (em[em_start] <= et) & (em[em_end] >= st)
        if not overlaps.any(): continue
        if have_em_ll and len(clat) > 0 and len(clon) > 0:
            lat0 = float(clat[i % len(clat)]) if len(clat) else np.nan
            lon0 = float(clon[i % len(clon)]) if len(clon) else np.nan
            if not (np.isnan(lat0) or np.isnan(lon0)):
                sub = em.loc[overlaps, [em_lat, em_lon]].dropna()
                if not sub.empty:
                    d = haversine_km(lat0, lon0, sub[em_lat].values, sub[em_lon].values)
                    if (d <= max_km).any():
                        keep_idx.append(i)
                        continue
        keep_idx.append(i)
    return g.loc[keep_idx]

def audit_emdat_matches(
    kept_gdf: gpd.GeoDataFrame,
    emdat_df: pd.DataFrame,
    gdf_start: str, gdf_end: str,
    gdf_lat: str = None, gdf_lon: str = None,
    em_start: str = "start_date", em_end: str = "end_date",
    em_lat: str = "em_lat", em_lon: str = "em_lon",
    max_km: float = 250.0,
) -> set:
    """
    Return a set of EM-DAT DataFrame index values that match at least one kept feature
    by temporal overlap and (if available) lat/lon proximity.
    """
    if kept_gdf.empty or emdat_df.empty:
        return set()

    g = kept_gdf.copy()
    g[gdf_start] = pd.to_datetime(g[gdf_start], errors="coerce")
    g[gdf_end]   = pd.to_datetime(g[gdf_end], errors="coerce")

    em = emdat_df.copy()
    em[em_start] = pd.to_datetime(em[em_start], errors="coerce")
    em[em_end]   = pd.to_datetime(em[em_end], errors="coerce")

    # GDF lat/lon
    if gdf_lat is None or gdf_lon is None:
        glat, glon = _centroid_latlon(g)
    else:
        glat = pd.to_numeric(g[gdf_lat], errors="coerce").values
        glon = pd.to_numeric(g[gdf_lon], errors="coerce").values

    # EM-DAT lat/lon detection
    if em_lat not in em.columns:
        for a in ["em_lat","latitude","lat","LAT","Lat"]:
            if a in em.columns: em_lat = a; break
    if em_lon not in em.columns:
        for o in ["em_lon","longitude","lon","LON","Lon","LONGITUDE"]:
            if o in em.columns: em_lon = o; break

    have_em_ll = (em_lat in em.columns) and (em_lon in em.columns)
    matched_idx = set()

    for i, prow in g.iterrows():
        st, et = prow[gdf_start], prow[gdf_end]
        if pd.isna(st) or pd.isna(et):
            continue
        temporal = (em[em_start] <= et) & (em[em_end] >= st)
        if not temporal.any():
            continue

        idxs = em.index[temporal]
        if have_em_ll and len(glat) > 0 and len(glon) > 0:
            lat0 = float(glat[i % len(glat)]) if len(glat) else np.nan
            lon0 = float(glon[i % len(glon)]) if len(glon) else np.nan
            if not (np.isnan(lat0) or np.isnan(lon0)):
                sub = em.loc[idxs, [em_lat, em_lon]].dropna()
                if not sub.empty:
                    d = haversine_km(lat0, lon0, sub[em_lat].values, sub[em_lon].values)
                    matched_idx.update(sub.index[(d <= max_km)])
                    continue
        matched_idx.update(idxs)  # fallback: temporal-only if no lat/lon
    return matched_idx

def write_exclusions(
    output_folder: str,
    emdat_df: pd.DataFrame,
    matched_em_idx: set,
    storms_before: gpd.GeoDataFrame, storms_after: gpd.GeoDataFrame,
    quakes_before: gpd.GeoDataFrame,  quakes_after: gpd.GeoDataFrame,
    tsu_before: gpd.GeoDataFrame,     tsu_after: gpd.GeoDataFrame,
    floods_before: gpd.GeoDataFrame, floods_after: gpd.GeoDataFrame,
):
    """Write: (1) unmatched/matched EM-DAT CSVs, (2) excluded features per hazard to a GPKG."""
    os.makedirs(output_folder, exist_ok=True)
    # 1) EM-DAT audit CSVs
    try:
        emdat_df.assign(__idx=emdat_df.index).to_csv(os.path.join(output_folder, "audit_emdat_all.csv"), index=False)
        em_matched   = emdat_df.loc[list(matched_em_idx)]
        em_unmatched = emdat_df.drop(index=list(matched_em_idx), errors="ignore")
        em_matched.to_csv(os.path.join(output_folder, "audit_emdat_matched.csv"), index=False)
        em_unmatched.to_csv(os.path.join(output_folder, "audit_emdat_unmatched.csv"), index=False)
    except Exception as e:
        logging.warning(f"Failed writing EM-DAT audit CSVs: {e}")

    # 2) Excluded features per hazard
    gpkg = os.path.join(output_folder, "audit_excluded.gpkg")
    try:
        if storms_before is not None and hasattr(storms_before, "empty") and not storms_before.empty:
            excl = storms_before.loc[storms_before.index.difference(storms_after.index if not storms_after.empty else [])]
            if not excl.empty: excl.to_file(gpkg, layer="storms_excluded", driver="GPKG")
        if quakes_before is not None and hasattr(quakes_before, "empty") and not quakes_before.empty:
            excl = quakes_before.loc[quakes_before.index.difference(quakes_after.index if not quakes_after.empty else [])]
            if not excl.empty: excl.to_file(gpkg, layer="earthquakes_excluded", driver="GPKG")
        if tsu_before is not None and hasattr(tsu_before, "empty") and not tsu_before.empty:
            excl = tsu_before.loc[tsu_before.index.difference(tsu_after.index if not tsu_after.empty else [])]
            if not excl.empty: excl.to_file(gpkg, layer="tsunamis_excluded", driver="GPKG")
        if not floods_before.empty:
            excl = floods_before.loc[floods_before.index.difference(floods_after.index if not floods_after.empty else [])]
            if not excl.empty: excl.to_file(gpkg, layer="floods_excluded", driver="GPKG")
    except Exception as e:
        logging.warning(f"Failed writing excluded features GPKG: {e}")

# ---- writer: combined -> GDB (storms + quakes + tsunamis) ----
def combine_disasters_to_gdb(
    storm_gdf: gpd.GeoDataFrame,
    earthquake_gdf: gpd.GeoDataFrame,
    countries_path: str,
    output_gdb: str,
    tsunami_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(),
    output_folder: str = "disaster_output",
    verbose_geometry_logging: bool = False,
    do_spatial_iso: bool = False,          # toggle
    do_temporal_bucketing: bool = False,   # toggle
):
    """
    Combine the three hazards and write one GDB. Spatial ISO tagging and decade bucketing
    are controlled by toggles and default to False.
    """
    logging.info(f"Combining disasters and writing to {output_gdb} ...")

    countries = None
    if do_spatial_iso:
        try:
            countries = gpd.read_file(countries_path).to_crs("EPSG:4326")
            if ISO_COL not in countries.columns:
                logging.error(f"Missing '{ISO_COL}' in countries shapefile.")
                return gpd.GeoDataFrame()
        except Exception as e:
            logging.error(f"Failed loading countries: {e}")
            return gpd.GeoDataFrame()

    # tag event types + date normalize
    def normalize_block(gdf, label, date_src):
        if gdf is None or gdf.empty:
            return gpd.GeoDataFrame()
        gdf = gdf.copy()
        if 'event_type' not in gdf.columns:
            gdf['event_type'] = label.lower()
        if date_src and date_src in gdf.columns and 'date' not in gdf.columns:
            gdf = gdf.rename(columns={date_src: 'date'})
        if 'date' not in gdf.columns:
            gdf['date'] = pd.NaT

        # spatial ISO (optional)
        if do_spatial_iso:
            out = spatial_join_iso(gdf, countries, ISO_COL, label)
        else:
            out = gdf
            if "iso_a3" not in out.columns:
                out["iso_a3"] = "UNK"
        # temporal (optional)
        if do_temporal_bucketing:
            out["decade"] = out["date"].apply(assign_decade)
        else:
            if "decade" not in out.columns:
                out["decade"] = "unknown"
        return out

    storms_iso = normalize_block(storm_gdf,     "Storms",      "storm_date")
    quakes_iso = normalize_block(earthquake_gdf,"Earthquakes", "eq_date")
    tsu_iso    = normalize_block(tsunami_gdf,   "Tsunamis",    "date")

    combined = pd.concat(
        [df for df in [storms_iso, quakes_iso, tsu_iso] if df is not None and not df.empty],
        ignore_index=True
    )
    if combined.empty:
        logging.warning("Nothing to write: combined disasters GeoDataFrame is empty.")
        return gpd.GeoDataFrame()

    combined_gdf = gpd.GeoDataFrame(combined, crs="EPSG:4326")

    # group into layers
    from collections import defaultdict
    grouped = defaultdict(list)
    for _, row in combined_gdf.iterrows():
        etype  = row.get("event_type", "unknown")
        decade = row.get("decade", "unknown")
        iso    = row.get("iso_a3", "UNK")
        band   = (row.get("band") or "").lower() if 'band' in combined_gdf.columns else ""
        key    = (etype, decade, iso, band)
        grouped[key].append(row)

    layers = {}
    for (etype, decade, iso, band), rows in grouped.items():
        gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
        base = f"{etype}_{decade}_{iso}"
        layer_name = (f"{base}_{band}" if band else base).lower().replace("-", "_").replace(" ", "_")[:60]
        layers[layer_name] = gdf
        logging.info(f"Prepared layer '{layer_name}' with {len(gdf)} features.")

    # clear existing GDB
    if os.path.exists(output_gdb):
        import shutil
        try:
            shutil.rmtree(output_gdb)
            logging.info(f"Removed existing GDB: {output_gdb}")
        except OSError as e:
            logging.error(f"Error removing existing GDB {output_gdb}: {e}")
            return gpd.GeoDataFrame()

    # write
    for lname, lgdf in layers.items():
        try:
            lgdf = lgdf.copy()
            lgdf = lgdf[_geom_present_mask(lgdf)].copy()
            lgdf['geometry'] = lgdf['geometry'].buffer(0)
            lgdf['geometry'] = lgdf['geometry'].apply(ensure_multipolygon)

            if verbose_geometry_logging:
                logging.info(f"Layer '{lname}' geometry types:\n{lgdf.geometry.geom_type.value_counts()}")

            # store 'date' as 'event_date' for GDB friendliness
            if "date" in lgdf.columns:
                lgdf = lgdf.rename(columns={"date": "event_date"})
            lgdf.to_file(output_gdb, driver="OpenFileGDB", layer=lname)
        except Exception as e:
            fb = os.path.join(output_folder, "failed_layers", f"{lname}.shp")
            os.makedirs(os.path.dirname(fb), exist_ok=True)
            logging.error(f"Failed to write layer '{lname}': {e}")
            try:
                lgdf.to_file(fb)
                logging.warning(f"  Saved fallback shapefile: {fb}")
            except Exception as e2:
                logging.error(f"  Also failed fallback for '{lname}': {e2}")

    logging.info("Finished writing GeoDatabase.")
    return combined_gdf
