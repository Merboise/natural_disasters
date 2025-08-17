# ----
# storms.py (IBTrACS processing + fallback hierarchy + geometry logging)
# ----
import os, logging, random, numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point, MultiPoint, Polygon
from .helpers import (
    AGENCY_PREF, RAD_TIERS, QUADS, NM_TO_KM,
    CLIMO_R34_NM, SSHS_SCALE_K,
    _clean_radius, _first_positive, build_circle, extract_polygons_only,
    diagnose_geom, log_polygon_failure,
)

def choose_latlon(row: pd.Series):
    for pref, plat, plon in [
        ('USA', 'USA_LAT', 'USA_LON'),
        ('BOM', 'BOM_LAT', 'BOM_LON'),
        ('REUNION','REUNION_LAT','REUNION_LON'),
        ('TOKYO','TOKYO_LAT','TOKYO_LON'),
        ('CMA','CMA_LAT','CMA_LON'),
        ('HKO','HKO_LAT','HKO_LON'),
        ('KMA','KMA_LAT','KMA_LON'),
        ('NADI','NADI_LAT','NADI_LON'),
    ]:
        lat, lon = row.get(plat), row.get(plon)
        if pd.notna(lat) and pd.notna(lon):
            return float(lat), float(lon), pref
    lat, lon = row.get('LAT'), row.get('LON')
    if pd.notna(lat) and pd.notna(lon):
        return float(lat), float(lon), 'GEN'
    return np.nan, np.nan, None

def create_quadrant_polygon(lat, lon, r_ne_km, r_se_km, r_sw_km, r_nw_km):
    if all(pd.isna(r) or r<=0 for r in [r_ne_km, r_se_km, r_sw_km, r_nw_km]):
        return None
    ang = np.linspace(0, 2*np.pi, 360, endpoint=False)
    pts=[]
    for a in ang:
        if 0<=a<np.pi/2: r=r_ne_km
        elif np.pi/2<=a<np.pi: r=r_se_km
        elif np.pi<=a<3*np.pi/2: r=r_sw_km
        else: r=r_nw_km
        if pd.isna(r) or r<=0: 
            continue
        coslat = np.cos(np.radians(lat))
        denom = (111.0 * coslat) if coslat != 0 else 111.0
        x = lon + (r*np.cos(a))/denom
        y = lat + (r*np.sin(a))/111.0
        pts.append((x,y))
    if len(pts)<3: return None
    try:
        return Polygon(pts)
    except Exception:
        return None

def get_quadrant_radii(row):
    for a in AGENCY_PREF:
        for tier in RAD_TIERS:
            vals={}
            for q in QUADS:
                col=f"{a}_{tier}_{q}"
                v = _clean_radius(row.get(col))
                vals[q]= v*NM_TO_KM if pd.notna(v) else np.nan
            if any(pd.notna(vals[q]) and vals[q]>0 for q in QUADS):
                return tier, vals, a
    return None, None, None

def _circle_from_nm(lat, lon, radius_nm, points=180):
    # wrapper to avoid importing in this module
    return build_circle(lat, lon, radius_nm, points)

def _track_union(group, base_nm=25):
    circles=[]
    for _, r in group.iterrows():
        lat, lon = r.get("LAT"), r.get("LON")
        if pd.notna(lat) and pd.notna(lon):
            c=_circle_from_nm(float(lat), float(lon), base_nm)
            if c is not None and not c.is_empty:
                circles.append(c)
    if not circles: return None
    try:
        return gpd.GeoSeries(circles, crs="EPSG:4326").union_all()
    except Exception:
        return None

def _track_hull(group):
    pts=[]
    for _, r in group.iterrows():
        lat, lon = r.get("LAT"), r.get("LON")
        if pd.notna(lat) and pd.notna(lon):
            pts.append(Point(float(lon), float(lat)))
    if len(pts)<3: return None
    try:
        return MultiPoint(pts).convex_hull
    except Exception:
        try:
            return gpd.GeoSeries(pts, crs="EPSG:4326").union_all().convex_hull
        except Exception:
            return None

def select_best_storm_geom(group, basin=None, sshs_max=None, have_quadrant_poly=None, have_roci_nm=None, have_rmw_nm=None):
    n_pts = len(group)
    candidates=[]

    if have_quadrant_poly is not None and not have_quadrant_poly.is_empty:
        candidates.append(("quadrant", 0.9, have_quadrant_poly))

    if pd.notna(have_roci_nm) and have_roci_nm>0:
        lat0, lon0, _ = choose_latlon(group.iloc[-1])
        circ=_circle_from_nm(lat0, lon0, have_roci_nm)
        if circ is not None and not circ.is_empty:
            candidates.append(("roci", 0.8, circ))

    if pd.notna(have_rmw_nm) and have_rmw_nm>0:
        lat0, lon0, _ = choose_latlon(group.iloc[-1])
        k = SSHS_SCALE_K.get(int(sshs_max) if pd.notna(sshs_max) else 0, 2.0)
        circ=_circle_from_nm(lat0, lon0, have_rmw_nm*k)
        if circ is not None and not circ.is_empty:
            candidates.append(("rmw_scaled", 0.6, circ))

    if pd.notna(sshs_max) and basin:
        lat0, lon0, _ = choose_latlon(group.iloc[-1])
        r34 = CLIMO_R34_NM.get(basin,{}).get(int(sshs_max), np.nan)
        if pd.notna(r34) and r34>0:
            circ=_circle_from_nm(lat0, lon0, r34)
            if circ is not None and not circ.is_empty:
                candidates.append(("climo", 0.5, circ))

    tu=_track_union(group, base_nm=25)
    if tu is not None and not tu.is_empty:
        candidates.append(("track_union", 0.35, tu))

    hull=_track_hull(group)
    if hull is not None and not hull.is_empty:
        candidates.append(("hull", 0.25, hull))

    if not candidates: return None, None, None
    method, conf, geom = max(candidates, key=lambda x: x[1])
    return method, conf, geom

def process_ibtracs_data(input_file: str, output_folder: str, sample_size=None, random_seed=42):
    logging.info("Processing IBTrACS storm data...")
    os.makedirs(output_folder, exist_ok=True)

    try:
        df = pd.read_csv(input_file, header=0, skiprows=[1,2], low_memory=False, na_values=[""," "])
        df.columns = [c.strip() for c in df.columns]
        df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
        df = df[df['ISO_TIME'].dt.year >= 1980]

        max_sshs = df.groupby('SID')['USA_SSHS'].max(numeric_only=True)
        valid_sids = max_sshs[max_sshs>0].index
        if sample_size and sample_size < len(valid_sids):
            random.seed(random_seed)
            valid_sids = random.sample(list(valid_sids), sample_size)
        df = df[df['SID'].isin(valid_sids)]

        results=[]; skipped_rows=[]; processed=0; skipped=0

        for sid, group in df.groupby('SID'):
            group = group.sort_values('ISO_TIME')
            polys=[]
            for _, row in group.iterrows():
                lat, lon, _ = choose_latlon(row)
                if pd.isna(lat) or pd.isna(lon):
                    continue
                tier, quads, _ag = get_quadrant_radii(row)
                if tier is not None:
                    poly = create_quadrant_polygon(lat, lon, quads.get('NE'), quads.get('SE'), quads.get('SW'), quads.get('NW'))
                    if poly is not None and not poly.is_empty:
                        polys.append(poly)
                        continue
                # fallback circles
                roci_nm = _first_positive(row, [f"{a}_ROCI" for a in AGENCY_PREF])
                rmw_nm  = _first_positive(row, [f"{a}_RMW"  for a in AGENCY_PREF])
                if pd.notna(roci_nm):
                    c = build_circle(lat, lon, roci_nm)
                    if c is not None and not c.is_empty:
                        polys.append(c); continue
                if pd.notna(rmw_nm):
                    c = build_circle(lat, lon, rmw_nm*1.6)
                    if c is not None and not c.is_empty:
                        polys.append(c); continue

            # First attempt union of whatever we got from the track
            have_quadrant_poly=None
            if polys:
                try:
                    combined = gpd.GeoSeries(polys, crs="EPSG:4326").union_all().buffer(1.0).buffer(-1.0).simplify(0.005)
                    have_quadrant_poly = extract_polygons_only(combined)
                except Exception:
                    have_quadrant_poly=None

            # Representative values for selection
            sshs_max = group["USA_SSHS"].dropna().max() if "USA_SSHS" in group else np.nan
            basin = str(group["BASIN"].iloc[0]) if "BASIN" in group and pd.notna(group["BASIN"].iloc[0]) else None
            roci_cols = [c for c in group.columns if c.endswith("_ROCI")]
            rmw_cols  = [c for c in group.columns if c.endswith("_RMW")]
            have_roci_nm = np.nanmedian(pd.to_numeric(group[roci_cols].values.reshape(-1), errors="coerce")) if roci_cols else np.nan
            have_rmw_nm  = np.nanmedian(pd.to_numeric(group[rmw_cols].values.reshape(-1),  errors="coerce")) if rmw_cols  else np.nan

            method, conf, geom = select_best_storm_geom(
                group, basin=basin, sshs_max=sshs_max,
                have_quadrant_poly=have_quadrant_poly,
                have_roci_nm=have_roci_nm, have_rmw_nm=have_rmw_nm
            )

            if geom is None or geom.is_empty:
                skipped += 1
                reason = diagnose_geom(geom)
                log_polygon_failure("storm", sid, f"{reason} (method={method})", output_folder)
                continue

            results.append({
                'SID': sid,
                'geometry': geom,
                'geom_method': method,
                'geom_confidence': conf,
                'storm_date': group['ISO_TIME'].min(),
                'start_time': group['ISO_TIME'].min(),
                'end_time': group['ISO_TIME'].max()
            })
            processed += 1

        gdf = gpd.GeoDataFrame(results, crs="EPSG:4326")
        logging.info(f"IBTrACS: processed={processed}, skipped={skipped}")
        return gdf
    except Exception as e:
        logging.error(f"IBTrACS processing failed: {e}")
        return gpd.GeoDataFrame()
