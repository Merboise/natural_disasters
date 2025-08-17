# ----
# helpers.py (constants, logging, geometry helpers, GDB writer)
# ----
import os, sys, logging
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

# ---- small utils ----
def _bounds_ok(lat: float, lon: float) -> bool:
    return (-90 <= lat <= 90) and (-180 <= lon <= 180)

def _clean_radius(val):
    try:
        v = float(val)
        if np.isfinite(v) and v > 0:
            return v
    except Exception:
        pass
    return np.nan

def _first_positive(row, names):
    for n in names:
        if n in row:
            v = _clean_radius(row[n])
            if pd.notna(v):
                return v
    return np.nan

def build_circle(lat, lon, radius_nm, points=180) -> Optional[Polygon]:
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
    elif geom.geom_type in ['Polygon','MultiPolygon']:
        return geom
    return None

def ensure_multipolygon(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == 'Polygon':
        return MultiPolygon([geom])
    if geom.geom_type == 'MultiPolygon':
        return geom
    return None

# ---- geometry build logging ----
def log_polygon_failure(kind: str, sid: str, reason: str, output_folder: str):
    logging.error(f"{kind} polygon failed for {sid}: {reason}")
    os.makedirs(os.path.join(output_folder, "fallback_layers"), exist_ok=True)
    csv_path = os.path.join(output_folder, "fallback_layers", f"skipped_{kind.lower()}s.csv")
    import csv, os
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

# ---- writer: combine disasters to GDB (storms + quakes + tsunamis) ----
def _assign_decade(dt):
    if pd.isna(dt): return "unknown"
    y = dt.year if isinstance(dt, pd.Timestamp) else (int(dt) if pd.notna(dt) else None)
    if not y: return "unknown"
    return f"{y - (y % 10)}s"

def _spatial_join_iso(gdf, countries, iso_col, label):
    if gdf is None or gdf.empty:
        logging.warning(f"{label} GeoDataFrame is empty.")
        return gpd.GeoDataFrame()
    try:
        if gdf.crs != countries.crs:
            gdf = gdf.to_crs(countries.crs)
        joined = gpd.sjoin(gdf, countries[[iso_col, "geometry"]], how="inner", predicate="intersects")
        joined = joined.drop(columns="index_right").rename(columns={iso_col: "iso_a3"})
        logging.info(f"{label}: Spatial join assigned ISO codes to {len(joined)} features.")
        return joined
    except Exception as e:
        logging.error(f"{label}: Spatial join failed: {e}")
        gdf = gdf.copy()
        gdf["iso_a3"] = "UNK"
        return gdf

def combine_disasters_to_gdb(
    storm_gdf: gpd.GeoDataFrame,
    earthquake_gdf: gpd.GeoDataFrame,
    countries_path: str,
    output_gdb: str,
    tsunami_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(),
    output_folder: str = "disaster_output",
    fallback_path: str  = "disaster_output/fallback_layers",
    verbose_geometry_logging: bool = False,
):
    logging.info(f"Combining disasters and writing to {output_gdb} ...")
    iso_col = "SOV_A3"

    try:
        countries = gpd.read_file(countries_path).to_crs("EPSG:4326")
        if iso_col not in countries.columns:
            logging.error(f"Missing '{iso_col}' in countries shapefile.")
            return gpd.GeoDataFrame()
    except Exception as e:
        logging.error(f"Failed loading countries: {e}")
        return gpd.GeoDataFrame()

    # tag event types + date normalize
    if storm_gdf is not None and not storm_gdf.empty:
        storm_gdf = storm_gdf.copy()
        storm_gdf['event_type'] = 'storm'
        if 'storm_date' in storm_gdf.columns and 'date' not in storm_gdf.columns:
            storm_gdf = storm_gdf.rename(columns={'storm_date': 'date'})

    if earthquake_gdf is not None and not earthquake_gdf.empty:
        earthquake_gdf = earthquake_gdf.copy()
        earthquake_gdf['event_type'] = 'earthquake'
        if 'eq_date' in earthquake_gdf.columns and 'date' not in earthquake_gdf.columns:
            earthquake_gdf = earthquake_gdf.rename(columns={'eq_date': 'date'})

    if tsunami_gdf is not None and not tsunami_gdf.empty:
        tsunami_gdf = tsunami_gdf.copy()
        tsunami_gdf['event_type'] = 'tsunami'
        if 'date' not in tsunami_gdf.columns and 'year' in tsunami_gdf.columns:
            try:
                tsunami_gdf['date'] = pd.to_datetime(tsunami_gdf['year'].astype(int), format='%Y', errors='coerce')
            except Exception:
                tsunami_gdf['date'] = pd.NaT

    # iso joins
    storms_iso   = _spatial_join_iso(storm_gdf,     countries, iso_col, "Storms")
    quakes_iso   = _spatial_join_iso(earthquake_gdf,countries, iso_col, "Earthquakes")
    tsu_iso      = _spatial_join_iso(tsunami_gdf,   countries, iso_col, "Tsunamis")

    combined = pd.concat([df for df in [storms_iso, quakes_iso, tsu_iso] if df is not None and not df.empty], ignore_index=True)
    if combined.empty:
        logging.warning("Nothing to write: combined disasters GeoDataFrame is empty.")
        return gpd.GeoDataFrame()

    combined_gdf = gpd.GeoDataFrame(combined, crs="EPSG:4326")
    if 'date' in combined_gdf.columns:
        combined_gdf["decade"] = combined_gdf["date"].apply(_assign_decade)
    else:
        combined_gdf["decade"] = "unknown"

    # group into layers
    from collections import defaultdict
    grouped = defaultdict(list)
    for _, row in combined_gdf.iterrows():
        etype = row.get("event_type", "unknown")
        decade = row.get("decade", "unknown")
        iso = row.get("iso_a3", "UNK")
        band = (row.get("band") or "").lower() if 'band' in combined_gdf.columns else ""
        key = (etype, decade, iso, band)
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
