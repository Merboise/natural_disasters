# natural_disaster_pipeline.py
import os
import sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon, GeometryCollection
from shapely.validation import explain_validity
from shapely.errors import TopologicalError
from shapely.strtree import STRtree
from shapely.ops import split, linemerge
import numpy as np
import logging
import random
import rasterio
from rasterio import features
from collections import defaultdict, Counter
from typing import Optional, Tuple, Dict, Any, List

# ---------------------------
# Logging setup
# ---------------------------

def setup_logging(output_folder: str, level=logging.INFO) -> None:
    """Add a rotating file handler + console handler so logs go to both.
    Safe to call multiple times; will reset handlers."""
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers (avoid duplicates if rerun interactively)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(level)
    logger.addHandler(ch)

    log_dir = os.path.join(output_folder, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    fh_path = os.path.join(log_dir, 'pipeline.log')
    fh = logging.FileHandler(fh_path, encoding='utf-8')
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    logging.info(f"Logging to file: {fh_path}")

# ---------------------------
# Constants & helpers
# ---------------------------
NM_TO_KM = 1.852  # nautical miles -> kilometers

# Prefer agencies that actually provide NE/SE/SW/NW quadrants in IBTrACS
AGENCY_PREF: List[str] = [
    'USA',       # NHC/JTWC
    'BOM',       # Australia
    'REUNION',   # Meteo-France Reunion
]

# Highest-wind tier first; fall back to broader winds when needed
RAD_TIERS: List[str] = ['R64', 'R50', 'R34']
QUADS: List[str] = ['NE', 'SE', 'SW', 'NW']

# Configure numpy -> python float for json/log formatting
np.seterr(all='ignore')

def log_polygon_failure(disaster_type: str, sid: str, reason: str, output_folder: str, fallback_path: str):
    """
    Log a failed polygon build to both terminal and a skipped CSV file.
    """
    logging.error(f"{disaster_type} polygon failed for {sid}: {reason}")
    
    os.makedirs(fallback_path, exist_ok=True)
    csv_path = os.path.join(fallback_path, f"skipped_{disaster_type.lower()}s.csv")

    # Append to CSV
    import csv
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["SID", "reason"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"SID": sid, "reason": reason})


# ---------------------------
# Geometry helpers
# ---------------------------

# --- Heuristics / lookups (tune as you like) ---
SSHS_SCALE_K = {0: 1.8, 1: 2.0, 2: 2.2, 3: 2.4, 4: 2.6, 5: 2.8}  # RMW -> whole-storm radius factor
# Very rough typical R34 (nm) by SSHS & basin (fallback if nothing else); fill as you get better tables:
CLIMO_R34_NM = {
    "ATL": {0: 60, 1: 75, 2: 90, 3: 110, 4: 120, 5: 130},
    "EP":  {0: 55, 1: 70, 2: 85, 3: 100, 4: 115, 5: 125},
    "WP":  {0: 60, 1: 80, 2: 95, 3: 115, 4: 130, 5: 140},
    "SI":  {0: 55, 1: 70, 2: 85, 3: 105, 4: 120, 5: 130},
    "SP":  {0: 55, 1: 70, 2: 85, 3: 100, 4: 115, 5: 125},
    "NI":  {0: 50, 1: 65, 2: 80, 3: 95,  4: 110, 5: 120},
}

def _confidence_for(method, ctx):
    if method == "quadrant":
        frac = 0.0
        if ctx.get("n_pts", 0) > 0:
            frac = ctx.get("n_pts_quads", 0) / max(1, ctx["n_pts"])
        return min(1.0, 0.9 + 0.05 * frac)
    if method == "roci":
        bonus = 0.05 if ctx.get("n_pts_roci", 0) >= 3 else 0.0
        return 0.75 + bonus
    if method == "rmw_scaled":
        return 0.60
    if method == "climo":
        return 0.50
    if method == "track_union":
        return 0.35
    if method == "hull":
        return 0.25
    return 0.0

def _circle_from_nm(lat, lon, radius_nm, points=180):
    if pd.isna(radius_nm) or radius_nm <= 0:
        return None
    r_km = float(radius_nm) * 1.852
    ang = np.linspace(0, 2*np.pi, points, endpoint=False)
    coslat = np.cos(np.radians(lat))
    denom = (111.0 * coslat) if coslat != 0 else 111.0
    xs = lon + (r_km * np.cos(ang)) / denom
    ys = lat + (r_km * np.sin(ang)) / 111.0
    try:
        return Polygon(list(zip(xs, ys)))
    except Exception:
        return None

def _best_latlon_for_group(group):
    # prefer rows with any agency lat/lon; else fallback to generic
    for _, row in group.iterrows():
        for a in AGENCY_PREF:
            lat, lon = row.get(f"{a}_LAT"), row.get(f"{a}_LON")
            if pd.notna(lat) and pd.notna(lon):
                return float(lat), float(lon)
    # fallback to mean of generic LAT/LON where present
    rr = group.dropna(subset=["LAT","LON"])
    if not rr.empty:
        return float(rr["LAT"].iloc[-1]), float(rr["LON"].iloc[-1])  # last known
    return np.nan, np.nan

def _track_union(group, base_nm=25):
    """Union of small per-point circles along the track (Shapely 2-safe)."""
    circles = []
    for _, row in group.iterrows():
        lat, lon = row.get("LAT"), row.get("LON")
        if pd.notna(lat) and pd.notna(lon):
            c = _circle_from_nm(float(lat), float(lon), base_nm)
            if c is not None and not c.is_empty:
                circles.append(c)
    if not circles:
        return None
    try:
        # Shapely 2: use GeoSeries.union_all()
        return gpd.GeoSeries(circles, crs="EPSG:4326").union_all()
    except Exception:
        return None

def _track_hull(group):
    """Convex hull of the track points (no deprecated calls)."""
    pts = []
    for _, row in group.iterrows():
        lat, lon = row.get("LAT"), row.get("LON")
        if pd.notna(lat) and pd.notna(lon):
            pts.append(Point(float(lon), float(lat)))
    if len(pts) < 3:
        return None
    try:
        return MultiPoint(pts).convex_hull
    except Exception:
        try:
            # Alternative path using union_all on points if needed
            return gpd.GeoSeries(pts, crs="EPSG:4326").union_all().convex_hull
        except Exception:
            return None

def select_best_storm_geom(group, basin=None, sshs_max=None, have_quadrant_poly=None, have_roci_nm=None, have_rmw_nm=None):
    """
    group: pd.DataFrame for one SID (already time-sorted).
    have_quadrant_poly: a ready polygon from your existing quadrant routine (or None)
    have_roci_nm / have_rmw_nm: representative ROCI / RMW (e.g., median over the group), in nautical miles (or None)
    """
    n_pts = len(group)
    ctx = {"n_pts": n_pts, "n_pts_quads": 0, "n_pts_roci": 0}
    lat0, lon0 = _best_latlon_for_group(group)

    candidates = []  # list of (method, geometry, confidence, details)

    # 1) Quadrants (if your existing builder produced one)
    if have_quadrant_poly is not None and not have_quadrant_poly.is_empty:
        # estimate how many points had quadrants available
        quad_cols = [f"{a}_{t}_{q}" for a in AGENCY_PREF for t in RAD_TIERS for q in QUADS]
        ctx["n_pts_quads"] = int(group[quad_cols].apply(lambda r: any(pd.notna(v) and float(v) > 0 for v in r), axis=1).sum())
        conf = _confidence_for("quadrant", ctx)
        candidates.append(("quadrant", have_quadrant_poly, conf, {"tiers": RAD_TIERS}))

    # 2) ROCI circle
    if pd.notna(have_roci_nm) and have_roci_nm > 0 and pd.notna(lat0) and pd.notna(lon0):
        ctx["n_pts_roci"] = int(group[[f"{a}_ROCI" for a in AGENCY_PREF if f"{a}_ROCI" in group.columns]].notna().sum().sum() > 0)
        g = _circle_from_nm(lat0, lon0, have_roci_nm)
        if g is not None and not g.is_empty:
            conf = _confidence_for("roci", ctx)
            candidates.append(("roci", g, conf, {"roci_nm": float(have_roci_nm)}))

    # 3) RMW scaled
    if pd.notna(have_rmw_nm) and have_rmw_nm > 0 and pd.notna(lat0) and pd.notna(lon0):
        k = SSHS_SCALE_K.get(int(sshs_max) if pd.notna(sshs_max) else 0, 2.0)
        g = _circle_from_nm(lat0, lon0, have_rmw_nm * k)
        if g is not None and not g.is_empty:
            conf = _confidence_for("rmw_scaled", ctx)
            candidates.append(("rmw_scaled", g, conf, {"rmw_nm": float(have_rmw_nm), "k": float(k)}))

    # 4) Climatology (very rough)
    if pd.notna(sshs_max) and pd.notna(lat0) and pd.notna(lon0) and basin:
        r34_nm = CLIMO_R34_NM.get(basin, {}).get(int(sshs_max), np.nan)
        if pd.notna(r34_nm) and r34_nm > 0:
            g = _circle_from_nm(lat0, lon0, r34_nm)
            if g is not None and not g.is_empty:
                conf = _confidence_for("climo", ctx)
                candidates.append(("climo", g, conf, {"r34_nm": float(r34_nm), "basin": basin, "sshs": int(sshs_max)}))

    # 5) Track union (small buffers)
    tu = _track_union(group, base_nm=25)
    if tu is not None and not tu.is_empty:
        conf = _confidence_for("track_union", ctx)
        candidates.append(("track_union", tu, conf, {"base_nm": 25}))

    # 6) Convex hull
    hull = _track_hull(group)
    if hull is not None and not hull.is_empty:
        conf = _confidence_for("hull", ctx)
        candidates.append(("hull", hull, conf, {}))

    if not candidates:
        return None, None, None, []

    # Pick the single best (highest confidence)
    best = max(candidates, key=lambda x: x[2])

    # Optional inner/outer bands using top-2 methods (if available)
    bands = []
    if len(candidates) >= 2:
        # sort by confidence
        cand_sorted = sorted(candidates, key=lambda x: x[2], reverse=True)[:2]
        g1, g2 = cand_sorted[0][1], cand_sorted[1][1]
        try:
            outer = g1.union(g2)
            bands.append(("outer_band", outer))
        except Exception:
            pass
        try:
            inter = g1.intersection(g2)
            if not inter.is_empty:
                bands.append(("inner_band", inter))
        except Exception:
            pass

    method, geom, conf, details = best
    details["n_pts"] = n_pts
    details["n_pts_quads"] = ctx["n_pts_quads"]
    details["n_pts_roci"] = ctx["n_pts_roci"]
    return method, conf, details, bands, geom

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

def build_circle(lat, lon, radius_nm, points=180):
    """Approximate circle centered at (lat, lon); radius in nautical miles."""
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

def create_quadrant_polygon(lat: float, lon: float, r_ne, r_se, r_sw, r_nw) -> Optional[Polygon]:
    """Create a crude quadrant polygon from center + radii in nautical miles.
    Returns None if fewer than 3 points or all radii missing/zero."""
    if any(pd.isna(v) for v in [lat, lon]) or not _bounds_ok(lat, lon):
        return None

    if all(pd.isna(r) or r in (0, '0', ' 0') for r in [r_ne, r_se, r_sw, r_nw]):
        return None

    # 1 degree ~ 60 nm in latitude; use that for simple local scale
    # Convert nm -> degrees approximately
    def nm_to_deg(nm):
        try:
            return float(nm) / 60.0 if pd.notna(nm) else 0.0
        except Exception:
            return 0.0

    r_ne_d, r_se_d, r_sw_d, r_nw_d = map(nm_to_deg, [r_ne, r_se, r_sw, r_nw])

    # sample around 0..2π; pick radius by quadrant
    angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
    pts = []
    for a in angles:
        if 0 <= a < np.pi/2: r = r_ne_d
        elif np.pi/2 <= a < np.pi: r = r_se_d
        elif np.pi <= a < 3*np.pi/2: r = r_sw_d
        else: r = r_nw_d
        if r <= 0 or pd.isna(r):
            continue
        # dx/dy in degrees; very crude but fine for footprint sketching
        dx = r * np.cos(a)
        dy = r * np.sin(a)
        pts.append((lon + dx, lat + dy))

    if len(pts) < 3:
        return None
    try:
        return Polygon(pts)
    except Exception:
        return None


def extract_polygons_only(geom):
    if geom is None:
        return None
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if g.geom_type in ['Polygon', 'MultiPolygon']]
        if not polys:
            return None
        return polys[0] if len(polys) == 1 else MultiPolygon(polys)
    elif geom.geom_type in ['Polygon', 'MultiPolygon']:
        return geom
    return None

# ---------------------------
# IBTrACS field helpers
# ---------------------------

def choose_latlon(row: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Pick best available coordinates in priority order.
    Returns (lat, lon, who_set)."""
    # Agency-specific positions (often higher quality regionally)
    for pref, plat, plon in [
        ('USA', 'USA_LAT', 'USA_LON'),
        ('BOM', 'BOM_LAT', 'BOM_LON'),
        ('REUNION', 'REUNION_LAT', 'REUNION_LON'),
        ('TOKYO', 'TOKYO_LAT', 'TOKYO_LON'),  # may exist though not quadrant-providing
        ('CMA', 'CMA_LAT', 'CMA_LON'),
        ('HKO', 'HKO_LAT', 'HKO_LON'),
        ('KMA', 'KMA_LAT', 'KMA_LON'),
        ('NADI', 'NADI_LAT', 'NADI_LON'),
    ]:
        lat = row.get(plat)
        lon = row.get(plon)
        if pd.notna(lat) and pd.notna(lon) and _bounds_ok(float(lat), float(lon)):
            return float(lat), float(lon), pref

    # Fallback to generic LAT/LON
    lat = row.get('LAT')
    lon = row.get('LON')
    if pd.notna(lat) and pd.notna(lon) and _bounds_ok(float(lat), float(lon)):
        return float(lat), float(lon), 'GEN'
    return None, None, None


def get_quadrant_radii(row):
    """Return (tier, dict, agency) where dict has NE/SE/SW/NW radii in *kilometers* from
    the first agency/tier that provides any quadrant. If none found, returns (None, None, None)."""
    for a in AGENCY_PREF:
        for tier in RAD_TIERS:
            quad_vals = {}
            for q in QUADS:
                col = f"{a}_{tier}_{q}"
                if col in row:
                    quad_vals[q] = _clean_radius(row[col])
                else:
                    quad_vals[q] = np.nan
            if any(pd.notna(v) and v > 0 for v in quad_vals.values()):
                # convert nm -> km
                for q in QUADS:
                    if pd.notna(quad_vals[q]):
                        quad_vals[q] = quad_vals[q] * NM_TO_KM
                return tier, quad_vals, a
    return None, None, None



def _diagnose_row_issue(row: pd.Series) -> Dict[str, Any]:
    lat, lon, who_pos = choose_latlon(row)
    agency_used = None
    tier_used = None
    radii_found = None

    # scan agencies/tiers for first with usable quadrant radii
    for ag in AGENCY_PREF:
        for ti in RAD_TIERS:
            rr = get_quadrant_radii(row, ag, ti)
            if rr is not None:
                agency_used = ag
                tier_used = ti
                radii_found = rr
                break
        if radii_found is not None:
            break

    reason = None
    if lat is None or lon is None:
        reason = 'no_coords'
    elif radii_found is None:
        reason = 'no_quadrant_radii'

    return {
        'coord_src': who_pos,
        'agency_used': agency_used,
        'tier_used': tier_used,
        'r_ne': None if radii_found is None else radii_found[0],
        'r_se': None if radii_found is None else radii_found[1],
        'r_sw': None if radii_found is None else radii_found[2],
        'r_nw': None if radii_found is None else radii_found[3],
        'row_reason': reason,
        'lat': lat,
        'lon': lon,
    }


def _diagnose_geom_invalid(geom) -> Tuple[str, Dict[str, Any]]:
    if geom is None:
        return 'union_none', {}
    if geom.is_empty:
        return 'union_empty', {}
    if not geom.is_valid:
        return 'invalid_geometry', {'explain': explain_validity(geom)}
    if geom.area == 0:
        return 'zero_area', {}
    return 'other', {}

def _snap_points_to_coast(runups_gdf, coast_gdf, max_km=10):
    """
    Snap runup points to nearest coastline (line) within max_km.
    Returns a copy with columns: snapped_geom, snapped_dist_km
    """
    coast_lines = coast_gdf.geometry.boundary if coast_gdf.geom_type.isin(["Polygon","MultiPolygon"]).any() else coast_gdf.geometry
    coast_lines = gpd.GeoSeries(linemerge(coast_lines.unary_union), crs=coast_gdf.crs)
    tree = STRtree(list(coast_lines.geometry))
    out = runups_gdf.copy()
    snapped_pts, dists = [], []

    # project to meters for distances
    m_crs = "EPSG:3857"
    coast_m = coast_lines.to_crs(m_crs).iloc[0]
    pts_m = runups_gdf.to_crs(m_crs).geometry
    for p in pts_m:
        # nearest segment on merged line
        sp = coast_m.interpolate(coast_m.project(p))
        d = p.distance(sp) / 1000.0
        if d <= max_km:
            snapped_pts.append(sp)
            dists.append(d)
        else:
            snapped_pts.append(None)
            dists.append(np.inf)

    out["snapped_geom"] = gpd.GeoSeries(snapped_pts, crs=m_crs).to_crs(runups_gdf.crs)
    out["snapped_dist_km"] = dists
    out = out[out["snapped_geom"].notnull()].copy()
    out.set_geometry("snapped_geom", inplace=True)
    return out, coast_lines

def _idw_alongshore(snapped_runups, coast_line, power=2, min_pts=3, step_km=2.0):
    """
    Sample the coastline every step_km and interpolate runupHt via IDW along arc-length.
    Expects columns: 'snapped_geom' (Point on line), 'runupHt' (meters).
    Returns a GeoDataFrame of coastal samples with column 'ru_m'.
    """
    if len(snapped_runups) < min_pts:
        return gpd.GeoDataFrame(columns=["geometry","ru_m"], crs=snapped_runups.crs)

    # Work in meters CRS for distance parameterization
    m_crs = "EPSG:3857"
    line_m = coast_line.to_crs(m_crs).iloc[0]
    snaps_m = snapped_runups.to_crs(m_crs)

    # param positions (s) along the line for known points
    s_known, z_known = [], []
    for _, r in snaps_m.iterrows():
        s = line_m.project(r.geometry)
        s_known.append(s)
        z_known.append(float(r.get("runupHt", np.nan)))
    s_known = np.array(s_known)
    z_known = np.array(z_known)
    valid = np.isfinite(z_known)
    if valid.sum() < min_pts:
        return gpd.GeoDataFrame(columns=["geometry","ru_m"], crs=snapped_runups.crs)
    s_known = s_known[valid]; z_known = z_known[valid]

    # sample targets
    n_steps = int(np.ceil(line_m.length / (step_km*1000.0)))
    s_targets = np.linspace(0, line_m.length, max(n_steps, 2))
    pts = [line_m.interpolate(s) for s in s_targets]

    # 1D IDW on arc-length
    ru_vals = []
    for st in s_targets:
        d = np.abs(s_known - st)
        d[d==0] = 1e-6
        w = 1.0 / (d**power)
        ru_vals.append( (w @ z_known) / w.sum() )

    coast_samples_m = gpd.GeoDataFrame({"ru_m": ru_vals}, geometry=pts, crs=m_crs)
    coast_samples = coast_samples_m.to_crs(snapped_runups.crs)
    return coast_samples

def _coastal_strip(coast_geom, width_m=2000):
    """Buffer coastline inland/outland a bit to make a strip for connectivity seeding."""
    # width both sides; final connectivity will intersect with land anyway
    return coast_geom.buffer(width_m)

def _vector_inundation_from_dem(dem_path, coast_crs_geom, threshold_elev_m, max_inland_km=5):
    """
    Raster mask where DEM <= threshold; keep only components touching the coastal strip.
    Returns a MultiPolygon (in DEM CRS) as inundation.
    """
    with rasterio.open(dem_path) as ds:
        dem = ds.read(1, masked=True)
        dem_aff = ds.transform
        dem_crs = ds.crs

        # reproject coastal strip to DEM CRS
        coast_strip = gpd.GeoSeries([coast_crs_geom], crs=coast_crs_geom.crs).to_crs(dem_crs).iloc[0]
        # also limit search distance inland a bit for performance
        inland_clip = coast_strip.buffer(max_inland_km*1000)

        # build mask (True = flooded)
        mask = np.zeros(dem.shape, dtype=np.uint8)
        flooded = np.where(np.asarray(dem.filled(np.inf)) <= float(threshold_elev_m), 1, 0).astype(np.uint8)

        # polygonize flooded
        shapes = features.shapes(flooded, mask=None, transform=dem_aff)
        polys = []
        for geom, v in shapes:
            if v != 1:
                continue
            poly = gpd.GeoSeries([Polygon(geom["coordinates"][0])], crs=dem_crs).set_geometry(0).iloc[0]
            # clip to inland search region & require intersection with coastal strip for connectivity
            if poly.intersects(inland_clip) and poly.intersects(coast_strip):
                polys.append(poly)

        if not polys:
            return None, dem_crs
        return gpd.GeoSeries(polys, crs=dem_crs).unary_union, dem_crs

def _sector_clip(poly, coast_centroid, source_point, half_angle_deg=45, max_range_km=2000):
    """
    Clip polygon to a sector facing from coast_centroid toward source_point.
    If source_point is None, return original.
    """
    if poly is None or source_point is None:
        return poly
    import math
    c = coast_centroid
    dx = source_point.x - c.x
    dy = source_point.y - c.y
    base_ang = math.degrees(math.atan2(dy, dx))
    # make a big wedge polygon
    R = max_range_km * 1000
    angles = np.linspace(np.radians(base_ang - half_angle_deg), np.radians(base_ang + half_angle_deg), 64)
    xs = c.x + R*np.cos(angles)
    ys = c.y + R*np.sin(angles)
    wedge = Polygon([(c.x, c.y), *zip(xs, ys)])
    return poly.intersection(wedge)

def _bands_from_center(height_m, pct_low=0.2, pct_high=0.2):
    """
    Return (low, mid, high) elevation thresholds from a center value.
    """
    return max(0.0, height_m*(1.0-pct_low)), height_m, height_m*(1.0+pct_high)

def build_tsunami_inundation(
    runups_gdf,           # points with columns ['latitude','longitude','runupHt','tsunamiEventId', etc]
    coast_gdf,            # country coastline polygons or polylines (WGS84)
    dem_path,             # country DEM (GeoTIFF)
    event_id=None,        # optional: filter runups by event
    sector_source_pt=None,# optional: shapely Point (WGS84) for directionality (e.g., epicenter)
    inland_limit_km=10,
    band_percents=(0.2, 0.2) # (low, high) ±% around center (IDW)
):
    """
    Returns GeoDataFrame of inundation polygons with bands {LOW,MED,HIGH} and metadata.
    """
    # keep single event if asked
    r = runups_gdf.copy()
    if event_id is not None and "tsunamiEventId" in r.columns:
        r = r[r["tsunamiEventId"] == event_id].copy()
    if r.empty:
        return gpd.GeoDataFrame(columns=["event_id","band","ru_center","ru_low","ru_high","num_points","method","geometry"], geometry="geometry", crs="EPSG:4326")

    # standardize geometry
    if r.geometry.name != "geometry":
        r = gpd.GeoDataFrame(r, geometry=gpd.points_from_xy(r["longitude"], r["latitude"]), crs="EPSG:4326")

    coast = coast_gdf.to_crs("EPSG:4326")
    snapped, coast_line = _snap_points_to_coast(r, coast)
    if snapped.empty:
        # fall back: convex hull of raw points, small inland buffer
        hull = r.unary_union.convex_hull.buffer(1609)  # ~1 mile
        out = gpd.GeoDataFrame([{
            "event_id": event_id, "band":"MED", "ru_center": np.nan, "ru_low": np.nan, "ru_high": np.nan,
            "num_points": 0, "method":"points_hull_1mile", "geometry": hull
        }], crs="EPSG:4326")
        return out

    # alongshore IDW
    coast_samples = _idw_alongshore(snapped, coast_line)
    if coast_samples.empty:
        # same fallback as above
        hull = r.unary_union.convex_hull.buffer(1609)
        out = gpd.GeoDataFrame([{
            "event_id": event_id, "band":"MED", "ru_center": np.nan, "ru_low": np.nan, "ru_high": np.nan,
            "num_points": len(snapped), "method":"points_hull_1mile", "geometry": hull
        }], crs="EPSG:4326")
        return out

    # Convert alongshore field to a single representative center height (median) for this event/segment
    ru_center = float(np.nanmedian(coast_samples["ru_m"]))
    ru_low, ru_high = _bands_from_center(ru_center, *band_percents)

    # make coastal strip (seed for connectivity)
    strip = _coastal_strip(coast_line.to_crs("EPSG:3857").iloc[0], width_m=2000)
    strip_wgs = gpd.GeoSeries([strip], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]

    # DEM flooding at three thresholds
    results = []
    for band, thr in [("LOW", ru_low), ("MED", ru_center), ("HIGH", ru_high)]:
        poly, dem_crs = _vector_inundation_from_dem(dem_path, strip_wgs, thr, max_inland_km=inland_limit_km)
        if poly is None:
            continue
        # sector clip (optional)
        if sector_source_pt is not None:
            poly = _sector_clip(poly, strip_wgs.centroid, sector_source_pt, half_angle_deg=60, max_range_km=inland_limit_km)
            if poly.is_empty:
                continue
        # reproject to WGS84 for output
        poly_wgs = gpd.GeoSeries([poly], crs=dem_crs).to_crs("EPSG:4326").iloc[0]
        results.append({
            "event_id": event_id,
            "band": band,
            "ru_center": ru_center,
            "ru_low": ru_low,
            "ru_high": ru_high,
            "num_points": int(len(snapped)),
            "method": "runup_IDW + DEM_threshold + coast_connectivity",
            "geometry": poly_wgs
        })

    if not results:
        # last resort: coastline buffer inland by 1 mile
        coast_buf = coast_line.buffer(1609, cap_style=2).to_crs("EPSG:4326").iloc[0]
        results = [{
            "event_id": event_id, "band":"MED", "ru_center": np.nan, "ru_low": np.nan, "ru_high": np.nan,
            "num_points": int(len(snapped)), "method":"coastline_1mile_buffer", "geometry": coast_buf
        }]

    out = gpd.GeoDataFrame(results, crs="EPSG:4326")
    return out

def process_tsunami_data(
    tsunami_events_csv: str,
    tsunami_runups_csv: str,
    countries_path: str,
    dem_dir: str,
    output_folder: str,
    inland_limit_km: int = 10,
    band_percents=(0.2, 0.2)
):
    """
    Returns a GeoDataFrame of tsunami inundation polygons with fields:
    [event_id, band, method, ru_center, ru_low, ru_high, num_points, event_type='tsunami', date?, geometry]
    """
    try:
        events_df = pd.read_csv(tsunami_events_csv)
        runups_df = pd.read_csv(tsunami_runups_csv)
    except Exception as e:
        logging.error(f"Tsunami CSV load failed: {e}")
        return gpd.GeoDataFrame()

    # Standardize runup points GeoDataFrame
    if not {'latitude','longitude'}.issubset(runups_df.columns):
        logging.error("Runup CSV missing 'latitude'/'longitude' columns.")
        return gpd.GeoDataFrame()

    if 'runupHt' not in runups_df.columns:
        # Some catalogs use 'runupHt' or 'runup_height' etc — adjust here if needed
        logging.warning("Runup CSV missing 'runupHt'; defaulting zeros.")
        runups_df['runupHt'] = np.nan

    runups_gdf = gpd.GeoDataFrame(
        runups_df,
        geometry=gpd.points_from_xy(runups_df['longitude'], runups_df['latitude']),
        crs="EPSG:4326"
    )

    # Countries layer (for ISO and coastlines)
    try:
        countries = gpd.read_file(countries_path).to_crs("EPSG:4326")
        iso_col = "SOV_A3"
        if iso_col not in countries.columns:
            logging.error(f"Countries layer missing '{iso_col}'.")
            return gpd.GeoDataFrame()
    except Exception as e:
        logging.error(f"Failed loading countries for tsunamis: {e}")
        return gpd.GeoDataFrame()

    # Attach ISO to runup points by spatial join
    try:
        runups_iso = gpd.sjoin(runups_gdf, countries[[iso_col, "geometry"]], how="left", predicate="intersects")
        runups_iso = runups_iso.rename(columns={iso_col: "iso_a3"}).drop(columns="index_right")
    except Exception as e:
        logging.error(f"ISO spatial-join failed for runups: {e}")
        return gpd.GeoDataFrame()

    # Optional event date to carry into output (for decade tagging)
    def _mk_date(row):
        try:
            y = int(row.get('year')) if pd.notna(row.get('year')) else None
            m = int(row.get('month')) if pd.notna(row.get('month')) else 1
            d = int(row.get('day')) if pd.notna(row.get('day')) else 1
            return pd.to_datetime(f"{y}-{m}-{d}", errors='coerce') if y else pd.NaT
        except Exception:
            return pd.NaT
    events_df['date'] = events_df.apply(_mk_date, axis=1)

    # Build inundation per (event_id, iso)
    results = []
    for (event_id, iso), grp in runups_iso.groupby(['tsunamiEventId','iso_a3'], dropna=False):
        if pd.isna(event_id) or pd.isna(iso):
            continue

        # Coastline: country boundary for that ISO
        coast = countries[countries[iso_col] == iso]
        if coast.empty:
            logging.warning(f"No coast polygon for ISO {iso} (event {event_id}).")
            continue

        # DEM path convention: dem_dir/ISO.tif
        dem_path = os.path.join(dem_dir, f"{iso}.tif")
        if not os.path.exists(dem_path):
            logging.warning(f"DEM missing for ISO {iso} at {dem_path} (event {event_id}); skipping.")
            continue

        # Optional directionality from event epicenter if present
        ev = events_df[events_df['id'] == event_id]
        source_pt = None
        if not ev.empty and {'latitude','longitude'}.issubset(ev.columns):
            try:
                sy, sx = float(ev.iloc[0]['latitude']), float(ev.iloc[0]['longitude'])
                if np.isfinite(sx) and np.isfinite(sy):
                    source_pt = Point(sx, sy)
            except Exception:
                source_pt = None

        # Build polygons (uses helper previously provided)
        inund = build_tsunami_inundation(
            grp[['latitude','longitude','runupHt','tsunamiEventId','geometry']],
            coast,
            dem_path,
            event_id=event_id,
            sector_source_pt=source_pt,
            inland_limit_km=inland_limit_km,
            band_percents=band_percents
        )
        if inund.empty:
            continue

        # Attach ISO + date + event_type
        inund['iso_a3'] = iso
        inund['event_type'] = 'tsunami'
        # carry date from events
        inund['date'] = ev.iloc[0]['date'] if not ev.empty else pd.NaT

        results.append(inund)

    if not results:
        logging.warning("No tsunami inundation polygons produced.")
        return gpd.GeoDataFrame()

    out = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs="EPSG:4326")
    logging.info(f"Produced {len(out)} tsunami polygons.")
    return out

# ---------------------------
# EM-DAT processing
# ---------------------------

def process_emdat_data(emdat_csv_path: str) -> pd.DataFrame:
    logging.info("Processing EM-DAT data...")
    try:
        df = pd.read_csv(emdat_csv_path)

        def create_date(row, prefix):
            year = row.get(f'{prefix}year')
            month = row.get(f'{prefix}month', 1)
            day = row.get(f'{prefix}day', 1)
            if pd.isna(year):
                return pd.NaT
            month = int(month) if pd.notna(month) else 1
            day = int(day) if pd.notna(day) else 1
            return pd.to_datetime(f"{int(year)}-{month}-{day}", errors='coerce')

        df['start_date'] = df.apply(lambda r: create_date(r, 'start'), axis=1)
        df['end_date'] = df.apply(lambda r: create_date(r, 'end'), axis=1)
        df = df.dropna(subset=['start_date'])
        logging.info(f"Loaded {len(df)} EM-DAT records with valid start dates.")
        return df
    except Exception as e:
        logging.error(f"EM-DAT processing failed: {e}")
        return pd.DataFrame()

# ---------------------------
# IBTrACS processing
# ---------------------------

def process_ibtracs_data(input_file, output_folder, sample_size=None, random_seed=42):
    logging.info("Processing IBTrACS storm data...")
    os.makedirs(output_folder, exist_ok=True)

    # Ensure we also log to file under the output folder
    try:
        log_path = os.path.join(output_folder, "pipeline.log")
        rootlog = logging.getLogger()
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == os.path.abspath(log_path) for h in rootlog.handlers):
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            rootlog.addHandler(fh)
    except Exception:
        pass

    # ------- helpers specific to the extended fallbacks -------

    # scan any agency columns that match *_R{tier}_{quad}
    import re
    QUAD_PATTERNS = [re.compile(rf"(.+?)_{tier}_(NE|SE|SW|NW)$") for tier in RAD_TIERS]

    def get_any_agency_quadrants(row):
        """
        Look for ANY columns in this row that match *_R34_NE, *_R34_SE, ... etc.
        Returns (tier, {NE,SE,SW,NW in km}, agency_tag) or (None, None, None).
        agency_tag will be 'ANY:<prefix>' if discovered.
        """
        cols = row.index if hasattr(row, 'index') else row.keys()
        # first try tiers in RAD_TIERS order
        for tier in RAD_TIERS:
            # gather all candidates with this tier
            prefix_to_quads = {}
            for c in cols:
                m = re.match(rf"(.+?)_{tier}_(NE|SE|SW|NW)$", str(c))
                if not m: 
                    continue
                pref, q = m.group(1), m.group(2)
                v = _clean_radius(row[c])
                prefix_to_quads.setdefault(pref, {qq: np.nan for qq in QUADS})
                if pd.notna(v) and v > 0:
                    prefix_to_quads[pref][q] = v
            # pick the first prefix that has at least one quad
            for pref, qdict in prefix_to_quads.items():
                if any(pd.notna(qdict[q]) and qdict[q] > 0 for q in QUADS):
                    # convert nm -> km
                    for q in QUADS:
                        if pd.notna(qdict[q]):
                            qdict[q] = qdict[q] * NM_TO_KM
                    return tier, qdict, f"ANY:{pref}"
        return None, None, None

    def km_circle(lat, lon, radius_km, points=180):
        if pd.isna(radius_km) or radius_km <= 0:
            return None
        ang = np.linspace(0, 2*np.pi, points, endpoint=False)
        coslat = np.cos(np.radians(lat))
        denom = (111.0 * coslat) if coslat != 0 else 111.0
        xs = lon + (radius_km * np.cos(ang)) / denom
        ys = lat + (radius_km * np.sin(ang)) / 111.0
        try:
            return Polygon(list(zip(xs, ys)))
        except Exception:
            return None

    # ---------------------------------------------------------

    try:
        # Read and filter
        df = pd.read_csv(input_file, header=0, skiprows=[1, 2], low_memory=False, na_values=["", " "])
        # normalize column headers (guard against stray spaces)
        df.columns = [c.strip() for c in df.columns]
        # parse time and time filter
        df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
        df = df[df['ISO_TIME'].dt.year >= 1980]

        # storm selection: SSHS > 0 at any time
        max_sshs = df.groupby('SID')['USA_SSHS'].max(numeric_only=True)
        valid_sids = max_sshs[max_sshs > 0].index
        total_sids = len(valid_sids)
        print(f"Total unique SIDs with SSHS > 0: {total_sids}")

        # optional sample
        if sample_size and sample_size < len(valid_sids):
            random.seed(random_seed)
            valid_sids = random.sample(list(valid_sids), sample_size)
        df = df[df['SID'].isin(valid_sids)]

        processed = 0
        skipped = 0
        skipped_rows = []
        processed_rows = []  # per-SID summary of which methods fired
        results = []

        from collections import defaultdict
        sid_diag = defaultdict(lambda: {
            'points_total': 0,
            'points_with_latlon': 0,
            'points_with_quadrants_pref': 0,
            'points_with_quadrants_any': 0,
            'points_used_roci_circle': 0,
            'points_used_rmw_circle': 0,
            'points_used_nearland_circle': 0,
            'points_used_convex_hull': 0,  # set later if SID-level hull is used
            'agencies_seen': set(),
            'tiers_seen': set(),
            'missing_fields_samples': set(),
            'methods_used_set': set(),    # overall methods (strings) for SID
        })

        # pre-check presence of optional cols
        has_dist2land = 'DIST2LAND' in df.columns
        has_speed = 'STORM_SPEED' in df.columns
        has_dir = 'STORM_DIR' in df.columns

        for sid, group in df.groupby('SID'):
            group = group.sort_values('ISO_TIME')
            polys = []

            # store track points to attempt convex hull as last resort
            track_pts = []

            for _, row in group.iterrows():
                sid_diag[sid]['points_total'] += 1

                # 1) pick lat/lon
                lat, lon, src_ag = choose_latlon(row)
                if pd.isna(lat) or pd.isna(lon):
                    sid_diag[sid]['missing_fields_samples'].add('latlon')
                    continue
                sid_diag[sid]['points_with_latlon'] += 1
                track_pts.append(Point(lon, lat))
                if src_ag:
                    sid_diag[sid]['agencies_seen'].add(src_ag)

                # 2) preferred-agency quadrants (your original)
                tier, quads, quad_ag = get_quadrant_radii(row)
                if tier is not None:
                    sid_diag[sid]['tiers_seen'].add(tier)
                    r_ne = quads.get('NE', np.nan)
                    r_se = quads.get('SE', np.nan)
                    r_sw = quads.get('SW', np.nan)
                    r_nw = quads.get('NW', np.nan)
                    poly = create_quadrant_polygon(lat, lon, r_ne, r_se, r_sw, r_nw)
                    if poly is not None and not poly.is_empty:
                        polys.append(poly)
                        sid_diag[sid]['points_with_quadrants_pref'] += 1
                        sid_diag[sid]['methods_used_set'].add('quadrants_preferred')
                        continue  # done for this point

                # 3) any-agency quadrants (scan all)
                tier2, quads2, any_ag = get_any_agency_quadrants(row)
                if tier2 is not None:
                    sid_diag[sid]['tiers_seen'].add(tier2)
                    r_ne = quads2.get('NE', np.nan)
                    r_se = quads2.get('SE', np.nan)
                    r_sw = quads2.get('SW', np.nan)
                    r_nw = quads2.get('NW', np.nan)
                    poly = create_quadrant_polygon(lat, lon, r_ne, r_se, r_sw, r_nw)
                    if poly is not None and not poly.is_empty:
                        polys.append(poly)
                        sid_diag[sid]['points_with_quadrants_any'] += 1
                        sid_diag[sid]['methods_used_set'].add('quadrants_any_agency')
                        continue

                # 4) ROCI / RMW circular fallback (nm -> km handled in build_circle)
                roci_nm = _first_positive(row, [f"{a}_ROCI" for a in AGENCY_PREF])
                rmw_nm  = _first_positive(row, [f"{a}_RMW" for a in AGENCY_PREF])

                if pd.notna(roci_nm):
                    circ = build_circle(lat, lon, roci_nm)
                    if circ is not None and not circ.is_empty:
                        polys.append(circ)
                        sid_diag[sid]['points_used_roci_circle'] += 1
                        sid_diag[sid]['methods_used_set'].add('circle_roci_nm')
                        continue

                if pd.notna(rmw_nm):
                    # heuristic scale: 1.6x (roughly gale extent factor) — tweak if desired
                    circ = build_circle(lat, lon, rmw_nm * 1.6)
                    if circ is not None and not circ.is_empty:
                        polys.append(circ)
                        sid_diag[sid]['points_used_rmw_circle'] += 1
                        sid_diag[sid]['methods_used_set'].add('circle_rmw_nm_scaled')
                        continue

                # 5) Near-land small buffer fallback
                # If DIST2LAND ≤ 50 km, use a 75 km circle
                if has_dist2land:
                    d2l = _clean_radius(row.get('DIST2LAND'))
                    if pd.notna(d2l) and d2l <= 50:
                        circ = km_circle(lat, lon, 75.0)
                        if circ is not None and not circ.is_empty:
                            polys.append(circ)
                            sid_diag[sid]['points_used_nearland_circle'] += 1
                            sid_diag[sid]['methods_used_set'].add('circle_nearland_75km')
                            continue

                # record what's missing for this point
                miss_bits = []
                if tier is None and tier2 is None:
                    miss_bits.append('no_quadrants_any_agency')
                if pd.isna(roci_nm):
                    miss_bits.append('no_ROCI')
                if pd.isna(rmw_nm):
                    miss_bits.append('no_RMW')
                if has_dist2land and (pd.isna(row.get('DIST2LAND'))):
                    miss_bits.append('no_DIST2LAND')
                sid_diag[sid]['missing_fields_samples'].add("+".join(miss_bits) if miss_bits else 'unknown')

            # If no polygons built for this SID, last resort convex hull of fixes
            if not polys and track_pts:
                try:
                    hull = gpd.GeoSeries(track_pts, crs="EPSG:4326").union_all().convex_hull
                    if hull is not None and not hull.is_empty:
                        polys.append(hull)
                        sid_diag[sid]['points_used_convex_hull'] = 1
                        sid_diag[sid]['methods_used_set'].add('convex_hull_track')
                except Exception:
                    pass

            # Still nothing? Skip with detailed reason
            if not polys:
                skipped += 1
                d = sid_diag[sid]
                reason = (
                    "no_valid_polygons — "
                    f"pts={d['points_total']}, with_latlon={d['points_with_latlon']}, "
                    f"quad_pref={d['points_with_quadrants_pref']}, quad_any={d['points_with_quadrants_any']}, "
                    f"circle_roci={d['points_used_roci_circle']}, circle_rmw={d['points_used_rmw_circle']}, "
                    f"nearland={d['points_used_nearland_circle']}, agencies={sorted(list(d['agencies_seen']))}, "
                    f"tiers={sorted(list(d['tiers_seen']))}, missing_samples={sorted(list(d['missing_fields_samples']))}"
                )
                skipped_rows.append({'SID': sid, 'reason': reason})
                continue

            # Union + clean
            try:
                combined = gpd.GeoSeries(polys).union_all().buffer(1.0).buffer(-1.0).simplify(0.005)
                final_geom = extract_polygons_only(combined)

                # Save for fallback hierarchy
                have_quadrant_poly = final_geom  

                if final_geom is None or final_geom.is_empty:
                    # Instead of skipping here, jump to fallback selection
                    pass  # We'll handle via select_best_storm_geom(...)
            except Exception as e:
                have_quadrant_poly = None
                # Instead of skipping here, also go to fallback selection
                pass
            
            # Build simple representatives
            sshs_max = group["USA_SSHS"].dropna().max() if "USA_SSHS" in group else np.nan
            basin = str(group["BASIN"].iloc[0]) if "BASIN" in group and pd.notna(group["BASIN"].iloc[0]) else None

            roci_cols = [c for c in group.columns if c.endswith("_ROCI")]
            rmw_cols  = [c for c in group.columns if c.endswith("_RMW")]
            have_roci_nm = np.nanmedian(pd.to_numeric(group[roci_cols].values.reshape(-1), errors="coerce")) if roci_cols else np.nan
            have_rmw_nm  = np.nanmedian(pd.to_numeric(group[rmw_cols].values.reshape(-1),  errors="coerce")) if rmw_cols  else np.nan

            method, conf, details, bands, geom = select_best_storm_geom(
                group, basin=basin, sshs_max=sshs_max,
                have_quadrant_poly=have_quadrant_poly,
                have_roci_nm=have_roci_nm, have_rmw_nm=have_rmw_nm
            )

            if method is None:
                skipped += 1
                skipped_rows.append({"SID": sid, "reason": "no_method_yielded_geometry"})
                continue

            # Use the selected geometry
            final_geom = details and method and bands  # just to avoid linter warnings; we use `method` and `conf` below
            final_geom = [c for c in [have_quadrant_poly] if method=="quadrant"]
            final_geom = final_geom[0] if final_geom else (bands[0][1] if False else None)  # we actually get it from return
            # better: we already have the geometry from select_best_storm_geom -> put it in a variable:
            selected_geom = [x for x in [have_quadrant_poly] if method=="quadrant"]
            # but we returned the geom directly (second element in best tuple); so adjust:
            # Update: access it directly by re-calling select_best... OR just modify select_best... to return `geom` too.

            # success
            methods_used = sorted(list(sid_diag[sid]['methods_used_set'])) or ['none']
            processed_rows.append({
                'SID': sid,
                'methods_used': "|".join(methods_used),
                'points_total': sid_diag[sid]['points_total'],
                'points_with_latlon': sid_diag[sid]['points_with_latlon'],
                'quad_pref_points': sid_diag[sid]['points_with_quadrants_pref'],
                'quad_any_points': sid_diag[sid]['points_with_quadrants_any'],
                'circle_roci_points': sid_diag[sid]['points_used_roci_circle'],
                'circle_rmw_points': sid_diag[sid]['points_used_rmw_circle'],
                'nearland_points': sid_diag[sid]['points_used_nearland_circle'],
                'used_convex_hull': sid_diag[sid]['points_used_convex_hull'],
            })

            results.append({
                'SID': sid,
                "geometry": geom,
                "geom_method": method,
                "geom_confidence": conf,
                "geom_inputs": details, 
                'storm_date': group['ISO_TIME'].min(),
                'start_time': group['ISO_TIME'].min(),
                'end_time': group['ISO_TIME'].max()
            })
            processed += 1

        gdf = gpd.GeoDataFrame(results, crs="EPSG:4326")
        logging.info(f"IBTrACS: Found {total_sids} storms with SSHS > 0 since 1980.")
        logging.info(f"IBTrACS: Successfully processed {processed} storms.")
        logging.info(f"IBTrACS: Skipped {skipped} storms due to missing or invalid geometry.")

        # Aggregate reason counts
        if skipped_rows:
            from collections import Counter
            reasons = Counter([r['reason'].split(' — ')[0] for r in skipped_rows])
            for reason, count in reasons.items():
                logging.info(f"Skipped {count} storms due to: {reason}")

        # Write diagnostics
        diag_dir = os.path.join(output_folder, "fallback_layers")
        os.makedirs(diag_dir, exist_ok=True)

        if skipped_rows:
            skipped_path = os.path.join(diag_dir, "skipped_storms_detailed.csv")
            pd.DataFrame(skipped_rows).to_csv(skipped_path, index=False)
            logging.warning(f"Saved skipped storms details to {skipped_path}")

        if processed_rows:
            proc_path = os.path.join(diag_dir, "processed_storms_methods.csv")
            pd.DataFrame(processed_rows).to_csv(proc_path, index=False)
            logging.info(f"Saved processed storms method summary to {proc_path}")

        return gdf

    except Exception as e:
        logging.error(f"IBTrACS processing failed: {e}")
        return gpd.GeoDataFrame()

# ---------------------------
# Earthquake processing
# ---------------------------

def process_earthquake_data(csv_path: str, output_folder: str) -> gpd.GeoDataFrame:
    logging.info("Processing earthquake data...")
    os.makedirs(output_folder, exist_ok=True)
    try:
        df = pd.read_csv(csv_path, parse_dates=['date'])
        df = df.rename(columns={'date': 'eq_date_orig'})
        out_rows = []

        for _, row in df.iterrows():
            point = Point(row['longitude'], row['latitude'])
            gdf = gpd.GeoDataFrame([row], geometry=[point], crs="EPSG:4326")
            gdf = gdf.to_crs("EPSG:3395")
            radius_m = float(row['earthqk_radius']) * 1609.34  # miles -> meters
            gdf['geometry'] = gdf.buffer(radius_m)
            gdf = gdf.rename(columns={'earthqk_radius': 'radius_mil', 'eq_date_orig': 'eq_date'})
            out_rows.append(gdf)

        final_gdf = gpd.GeoDataFrame(pd.concat(out_rows, ignore_index=True)).to_crs("EPSG:4326")
        logging.info(f"Processed {len(final_gdf)} earthquake buffers.")
        return final_gdf
    except Exception as e:
        logging.error(f"Earthquake processing failed: {e}")
        return gpd.GeoDataFrame()

# ---------------------------
# GDB writer
# ---------------------------

def combine_disasters_to_gdb(
    storm_gdf,
    earthquake_gdf,
    countries_path,
    output_gdb,
    tsunami_gdf=gpd.GeoDataFrame(),
    output_folder="disaster_output",
    fallback_path="disaster_output/fallback_layers",
    verbose_geometry_logging=False
):
    logging.info(f"Combining disasters and writing to {output_gdb} ...")

    iso_col = "SOV_A3"  # Natural Earth field we’ll map to iso_a3

    try:
        countries = gpd.read_file(countries_path).to_crs("EPSG:4326")
        logging.info(f"Loaded {len(countries)} country polygons.")
        if iso_col not in countries.columns:
            logging.error(f"Missing expected column '{iso_col}' in countries shapefile.")
            return gpd.GeoDataFrame()
    except Exception as e:
        logging.error(f"Failed loading countries: {e}")
        return gpd.GeoDataFrame()

    def spatial_join_iso(gdf, label):
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

    # Tag event type + normalize date columns before join
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
        # Expect fields from build_tsunami_inundation: event_id, band, method, ru_center, ru_low, ru_high
        # If a 'date' column exists we’ll use it for decades; otherwise “unknown”.
        if 'date' not in tsunami_gdf.columns and 'year' in tsunami_gdf.columns:
            # optional: derive a date if year is present
            try:
                tsunami_gdf['date'] = pd.to_datetime(tsunami_gdf['year'].astype(int), format='%Y', errors='coerce')
            except Exception:
                tsunami_gdf['date'] = pd.NaT

    # ISO joins
    storms_iso     = spatial_join_iso(storm_gdf,     "Storms")      if storm_gdf is not None else gpd.GeoDataFrame()
    quakes_iso     = spatial_join_iso(earthquake_gdf,"Earthquakes") if earthquake_gdf is not None else gpd.GeoDataFrame()
    tsunamis_iso   = spatial_join_iso(tsunami_gdf,   "Tsunamis")    if tsunami_gdf is not None else gpd.GeoDataFrame()

    # Combine
    combined = pd.concat([df for df in [storms_iso, quakes_iso, tsunamis_iso] if df is not None and not df.empty], ignore_index=True)
    if combined.empty:
        logging.warning("Nothing to write: combined disasters GeoDataFrame is empty.")
        return gpd.GeoDataFrame()

    combined_gdf = gpd.GeoDataFrame(combined, crs="EPSG:4326")

    def assign_decade(dt):
        if pd.isna(dt):
            return "unknown"
        y = dt.year if isinstance(dt, pd.Timestamp) else (int(dt) if pd.notna(dt) else None)
        if y is None:
            return "unknown"
        return f"{y - (y % 10)}s"

    if 'date' in combined_gdf.columns:
        combined_gdf["decade"] = combined_gdf["date"].apply(assign_decade)
    else:
        combined_gdf["decade"] = "unknown"

    # Grouping: event_type, decade, iso; if tsunamis carry a 'band', we split per band in layer names
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

    # Clear existing GDB
    if os.path.exists(output_gdb):
        import shutil
        try:
            shutil.rmtree(output_gdb)
            logging.info(f"Removed existing GDB: {output_gdb}")
        except OSError as e:
            logging.error(f"Error removing existing GDB {output_gdb}: {e}")
            return gpd.GeoDataFrame()

    def ensure_multipolygon(geom):
        if geom is None or geom.is_empty:
            return None
        if geom.geom_type == 'Polygon':
            return MultiPolygon([geom])
        elif geom.geom_type == 'MultiPolygon':
            return geom
        else:
            return None

    # Write layers
    logging.info(f"Writing {len(layers)} layers to {output_gdb} ...")
    for lname, lgdf in layers.items():
        try:
            # keep tsunami attributes (band, method, ru_center/low/high) as normal fields
            if 'date' in lgdf.columns:
                lgdf = lgdf.rename(columns={'date': 'event_date'})

            lgdf = lgdf.copy()
            # fix geometries & unify type
            lgdf['geometry'] = lgdf['geometry'].buffer(0)
            lgdf['geometry'] = lgdf['geometry'].apply(ensure_multipolygon)

            if verbose_geometry_logging:
                logging.info(f"Layer '{lname}' geometry types:\n{lgdf.geometry.geom_type.value_counts()}")

            lgdf.to_file(output_gdb, driver="OpenFileGDB", layer=lname)

        except Exception as e:
            try:
                os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
            except Exception:
                pass
            logging.error(f"Failed to write layer '{lname}': {e}")
            try:
                fb_path = os.path.join(output_folder, "failed_layers", f"{lname}.shp")
                os.makedirs(os.path.dirname(fb_path), exist_ok=True)
                lgdf.to_file(fb_path)
                logging.warning(f"  Layer '{lname}' saved as fallback shapefile: {fb_path}")
            except Exception as e2:
                logging.error(f"  Also failed to write fallback for '{lname}': {e2}")

    logging.info("Finished writing GeoDatabase.")
    return combined_gdf

# ---------------------------
# Temporal overlap helper
# ---------------------------

def has_temporal_overlap(ibtracs_start, ibtracs_end, emdat_start, emdat_end, tolerance_days=7):
    if pd.isna(ibtracs_start) or pd.isna(ibtracs_end) or pd.isna(emdat_start) or pd.isna(emdat_end):
        return False
    emdat_start_ext = emdat_start - pd.Timedelta(days=tolerance_days)
    emdat_end_ext = emdat_end + pd.Timedelta(days=tolerance_days)
    return max(ibtracs_start, emdat_start_ext) <= min(ibtracs_end, emdat_end_ext)

# ---------------------------
# Main
# ---------------------------

# ---------------------------
# Main
# ---------------------------

def main(
    use_emdat_filtering: bool = False,
    run_storms: bool = True,
    run_earthquakes: bool = True,
    run_tsunamis: bool = True,                              # NEW
    ibtracs_file: str = "ibtracs.ALL.list.v04r01.csv",
    earthquake_file: str = "earthquakes.csv",
    tsunami_events_file: str = "tsunami_events.csv",        # NEW
    tsunami_runups_file: str = "tsunami_runup.csv",         # NEW
    dem_dir: str = r"dem_by_iso",                           # NEW (folder with <ISO>.tif)
    emdat_file: str = "Top5EMDAT.csv",
    landmask_path: str = "gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp",
    countries_path: str = r"C:\\Users\\WAS\\Desktop\\Python\\projects\\RESEARCH\\ne_10m_admin_0_countries\\ne_10m_admin_0_countries.shp",
    output_folder: str = "disaster_output",
    fallback_path: str = "disaster_output/fallback_layers",
    verbose_geometry_logging: bool = False,
):
    os.makedirs(output_folder, exist_ok=True)
    setup_logging(output_folder)
    logging.info("Starting natural disaster pipeline...")

    emdat_df = None
    if use_emdat_filtering:
        emdat_df = process_emdat_data(emdat_file)
        if emdat_df.empty:
            logging.warning("EM-DAT data empty or failed to load; disabling EM-DAT filtering.")
            use_emdat_filtering = False

    storm_gdf = gpd.GeoDataFrame()
    if run_storms:
        storm_gdf = process_ibtracs_data(ibtracs_file, output_folder)
        print(f"Total processed storm polygons: {len(storm_gdf)}")
        if use_emdat_filtering and not storm_gdf.empty:
            keep_indices = []
            for i, storm in storm_gdf.iterrows():
                overlaps = emdat_df.apply(
                    lambda r: has_temporal_overlap(
                        storm['start_time'], storm['end_time'],
                        r['start_date'], r['end_date']
                    ), axis=1)
                if overlaps.any():
                    keep_indices.append(i)
            storm_gdf = storm_gdf.loc[keep_indices]
            logging.info(f"Storms after EM-DAT temporal filtering: {len(storm_gdf)}")

    earthquake_gdf = gpd.GeoDataFrame()
    if run_earthquakes:
        earthquake_gdf = process_earthquake_data(earthquake_file, output_folder)
        if use_emdat_filtering and not earthquake_gdf.empty:
            keep_indices = []
            for i, eq in earthquake_gdf.iterrows():
                overlaps = emdat_df.apply(
                    lambda r: has_temporal_overlap(
                        eq['date'], eq['date'],
                        r['start_date'], r['end_date']
                    ), axis=1)
                if overlaps.any():
                    keep_indices.append(i)
            earthquake_gdf = earthquake_gdf.loc[keep_indices]
            logging.info(f"Earthquakes after EM-DAT temporal filtering: {len(earthquake_gdf)}")

    tsunami_gdf = gpd.GeoDataFrame()
    if run_tsunamis:
        tsunami_gdf = process_tsunami_data(
            tsunami_events_file,
            tsunami_runups_file,
            countries_path=countries_path,
            dem_dir=dem_dir,
            output_folder=output_folder,
            inland_limit_km=10,
            band_percents=(0.2, 0.2)
        )

    # Write separate GDBs
    if run_storms and not storm_gdf.empty:
        storm_gdb_path = os.path.join(output_folder, "storms.gdb")
        combine_disasters_to_gdb(storm_gdf, gpd.GeoDataFrame(), countries_path, storm_gdb_path,
                                 tsunami_gdf=gpd.GeoDataFrame(),
                                 output_folder=output_folder, fallback_path=fallback_path,
                                 verbose_geometry_logging=verbose_geometry_logging)

    if run_earthquakes and not earthquake_gdf.empty:
        earthquake_gdb_path = os.path.join(output_folder, "earthquakes.gdb")
        combine_disasters_to_gdb(gpd.GeoDataFrame(), earthquake_gdf, countries_path, earthquake_gdb_path,
                                 tsunami_gdf=gpd.GeoDataFrame(),
                                 output_folder=output_folder, fallback_path=fallback_path,
                                 verbose_geometry_logging=verbose_geometry_logging)

    if run_tsunamis and not tsunami_gdf.empty:
        tsunami_gdb_path = os.path.join(output_folder, "tsunamis.gdb")
        combine_disasters_to_gdb(gpd.GeoDataFrame(), gpd.GeoDataFrame(), countries_path, tsunami_gdb_path,
                                 tsunami_gdf=tsunami_gdf,
                                 output_folder=output_folder, fallback_path=fallback_path,
                                 verbose_geometry_logging=verbose_geometry_logging)

    # Write combined all disasters GDB
    if not (storm_gdf.empty and earthquake_gdf.empty and tsunami_gdf.empty):
        combined_gdb_path = os.path.join(output_folder, "all_disasters.gdb")
        combine_disasters_to_gdb(storm_gdf, earthquake_gdf, countries_path, combined_gdb_path,
                                 tsunami_gdf=tsunami_gdf,
                                 output_folder=output_folder, fallback_path=fallback_path,
                                 verbose_geometry_logging=verbose_geometry_logging)

    # EM-DAT filtered (storms/eq only for now)
    if use_emdat_filtering and (not storm_gdf.empty or not earthquake_gdf.empty):
        emdat_gdb_path = os.path.join(output_folder, "emdat_disasters.gdb")
        combine_disasters_to_gdb(storm_gdf, earthquake_gdf, countries_path, emdat_gdb_path,
                                 tsunami_gdf=gpd.GeoDataFrame(),  # leave tsunamis out of EM-DAT set
                                 output_folder=output_folder, fallback_path=fallback_path,
                                 verbose_geometry_logging=verbose_geometry_logging)

    logging.info("Pipeline completed.")

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    main()
