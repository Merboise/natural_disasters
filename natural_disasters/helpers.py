# ----
# helpers.py (constants, logging, geometry helpers, optional ISO/time joins, GDB writers)
# ----
import os, sys, logging, psutil
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Optional
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.validation import explain_validity

# ---- constants ----
NM_TO_KM = 1.852
AGENCY_PREF = ['USA','BOM','REUNION','TOKYO','CMA','HKO','KMA','NADI']
RAD_TIERS   = ['R64','R50','R34']
QUADS       = ['NE','SE','SW','NW']

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

# ---- writer: single hazard -> GDB (layer by decade/ISO optionally) ----
def write_single_hazard_gdb(
    gdf: gpd.GeoDataFrame,
    countries_path: str,
    output_gdb: str,
    label: str,
    date_col: str | None = None,
    extra_name_part: str | None = None,
    verbose_geometry_logging: bool = False,
    do_spatial_iso: bool = False,          # toggle
    do_temporal_bucketing: bool = False,   # toggle
):
    """
    Write ONE hazard dataset (already tagged with event_type) into an OpenFileGDB,
    layering by decade and ISO only if toggles are enabled.
    """
    if gdf is None or gdf.empty:
        logging.info(f"{label}: nothing to write (empty).")
        return gpd.GeoDataFrame()

    # normalize date column
    gdf = gdf.copy()
    if date_col and date_col in gdf.columns and "date" not in gdf.columns:
        gdf = gdf.rename(columns={date_col: "date"})
    elif "date" not in gdf.columns:
        gdf["date"] = pd.NaT

    # load countries (only needed if do_spatial_iso)
    countries = None
    if do_spatial_iso:
        countries = gpd.read_file(countries_path).to_crs("EPSG:4326")
        if ISO_COL not in countries.columns:
            raise ValueError(f"Countries layer missing '{ISO_COL}'")

    # spatial ISO tag (optional)
    if do_spatial_iso:
        gdf_iso = spatial_join_iso_single(gdf, countries, ISO_COL, label)
    else:
        gdf_iso = gdf.copy()
        if "iso_a3" not in gdf_iso.columns:
            gdf_iso["iso_a3"] = "UNK"

    # temporal bucketing (optional)
    if do_temporal_bucketing:
        gdf_iso["decade"] = gdf_iso["date"].apply(assign_decade)
    else:
        if "decade" not in gdf_iso.columns:
            gdf_iso["decade"] = "unknown"

    # group into layers
    grouped = {}
    for _, row in gdf_iso.iterrows():
        decade = row.get("decade", "unknown")
        iso = row.get("iso_a3", "UNK")
        extra = (extra_name_part or "").strip().lower().replace(" ", "_")
        layer_name = f"{row.get('event_type','event')}_{decade}_{iso}"
        if extra:
            layer_name = f"{layer_name}_{extra}"
        grouped.setdefault(layer_name[:60], []).append(row)

    # clear existing GDB dir
    if os.path.exists(output_gdb):
        import shutil
        shutil.rmtree(output_gdb, ignore_errors=True)

    # write
    for lname, rows in grouped.items():
        lgdf = gpd.GeoDataFrame(rows, crs="EPSG:4326").copy()
        # standardize geometry
        lgdf["geometry"] = lgdf["geometry"].buffer(0)
        lgdf["geometry"] = lgdf["geometry"].apply(ensure_multipolygon)
        if verbose_geometry_logging:
            logging.info(f"{label}: layer '{lname}' geometry types:\n{lgdf.geometry.geom_type.value_counts()}")
        # rename date to event_date for GDB friendliness
        if "date" in lgdf.columns:
            lgdf = lgdf.rename(columns={"date": "event_date"})
        lgdf.to_file(output_gdb, driver="OpenFileGDB", layer=lname)

    logging.info(f"{label}: wrote {len(grouped)} layer(s) to {output_gdb}.")
    return gdf_iso

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
