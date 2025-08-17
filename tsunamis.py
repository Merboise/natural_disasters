# ----
# tsunamis.py
# ----
import os, logging, numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.strtree import STRtree
from shapely.ops import linemerge
import rasterio
from rasterio import features
from .helpers import diagnose_geom

def _snap_points_to_coast(runups_gdf, coast_gdf, max_km=10):
    coast_lines = coast_gdf.geometry.boundary if coast_gdf.geom_type.isin(["Polygon","MultiPolygon"]).any() else coast_gdf.geometry
    coast_lines = gpd.GeoSeries(linemerge(coast_lines.unary_union), crs=coast_gdf.crs)
    out = runups_gdf.copy()

    m_crs = "EPSG:3857"
    coast_m = coast_lines.to_crs(m_crs).iloc[0]
    pts_m = runups_gdf.to_crs(m_crs).geometry
    snapped_pts=[]; dists=[]
    for p in pts_m:
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
    if len(snapped_runups)<min_pts:
        return gpd.GeoDataFrame(columns=["geometry","ru_m"], crs=snapped_runups.crs)
    m_crs="EPSG:3857"
    line_m = coast_line.to_crs(m_crs).iloc[0]
    snaps_m = snapped_runups.to_crs(m_crs)

    s_known=[]; z_known=[]
    for _, r in snaps_m.iterrows():
        s = line_m.project(r.geometry)
        s_known.append(s); z_known.append(float(r.get("runupHt", np.nan)))
    s_known=np.array(s_known); z_known=np.array(z_known)
    valid = np.isfinite(z_known)
    if valid.sum()<min_pts:
        return gpd.GeoDataFrame(columns=["geometry","ru_m"], crs=snapped_runups.crs)
    s_known=s_known[valid]; z_known=z_known[valid]

    n_steps = int(np.ceil(line_m.length/(step_km*1000.0)))
    s_targets = np.linspace(0, line_m.length, max(n_steps, 2))
    pts=[line_m.interpolate(s) for s in s_targets]

    ru_vals=[]
    for st in s_targets:
        d = np.abs(s_known - st); d[d==0]=1e-6
        w = 1.0/(d**power)
        ru_vals.append((w@z_known)/w.sum())

    coast_samples_m = gpd.GeoDataFrame({"ru_m":ru_vals}, geometry=pts, crs=m_crs)
    return coast_samples_m.to_crs(snapped_runups.crs)

def _coastal_strip(coast_geom, width_m=2000):
    return coast_geom.buffer(width_m)

def _vector_inundation_from_dem(dem_path, coast_wgs, threshold_elev_m, max_inland_km=10):
    with rasterio.open(dem_path) as ds:
        dem = ds.read(1, masked=True)
        dem_aff = ds.transform
        dem_crs = ds.crs
        coast_strip = gpd.GeoSeries([coast_wgs], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
        inland_clip = coast_strip.buffer(max_inland_km*1000)

        flooded = np.where(np.asarray(dem.filled(np.inf)) <= float(threshold_elev_m), 1, 0).astype(np.uint8)
        polys=[]
        for geom, v in features.shapes(flooded, transform=dem_aff):
            if v != 1: continue
            try:
                poly = Polygon(geom["coordinates"][0])
            except Exception:
                continue
            if Polygon(poly).intersects(inland_clip) and Polygon(poly).intersects(coast_strip):
                polys.append(Polygon(poly))
        if not polys:
            return None, dem_crs
        return gpd.GeoSeries(polys, crs=dem_crs).unary_union, dem_crs

def _bands_from_center(height_m, pct_low=0.2, pct_high=0.2):
    return max(0.0, height_m*(1.0-pct_low)), height_m, height_m*(1.0+pct_high)

def build_tsunami_inundation(runups_gdf, coast_gdf, dem_path, event_id=None, inland_limit_km=10, band_percents=(0.2,0.2)):
    r = runups_gdf.copy()
    if event_id is not None and "tsunamiEventId" in r.columns:
        r = r[r["tsunamiEventId"]==event_id].copy()
    if r.empty:
        return gpd.GeoDataFrame(columns=["event_id","band","ru_center","ru_low","ru_high","num_points","method","geometry"], crs="EPSG:4326", geometry="geometry")

    if r.geometry.name != "geometry":
        r = gpd.GeoDataFrame(r, geometry=gpd.points_from_xy(r["longitude"], r["latitude"]), crs="EPSG:4326")

    coast = coast_gdf.to_crs("EPSG:4326")
    snapped, coast_line = _snap_points_to_coast(r, coast)
    if snapped.empty:
        hull = r.unary_union.convex_hull.buffer(1609)
        return gpd.GeoDataFrame([{
            "event_id": event_id, "band":"MED","ru_center":np.nan,"ru_low":np.nan,"ru_high":np.nan,
            "num_points":0,"method":"points_hull_1mile","geometry":hull
        }], crs="EPSG:4326")

    coast_samples = _idw_alongshore(snapped, coast_line)
    if coast_samples.empty:
        hull = r.unary_union.convex_hull.buffer(1609)
        return gpd.GeoDataFrame([{
            "event_id": event_id, "band":"MED","ru_center":np.nan,"ru_low":np.nan,"ru_high":np.nan,
            "num_points":len(snapped),"method":"points_hull_1mile","geometry":hull
        }], crs="EPSG:4326")

    ru_center = float(np.nanmedian(coast_samples["ru_m"]))
    ru_low, ru_high = _bands_from_center(ru_center, *band_percents)

    strip = _coastal_strip(coast_line.to_crs("EPSG:3857").iloc[0], width_m=2000)
    strip_wgs = gpd.GeoSeries([strip], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]

    results=[]
    for band, thr in [("LOW", ru_low), ("MED", ru_center), ("HIGH", ru_high)]:
        poly, dem_crs = _vector_inundation_from_dem(dem_path, strip_wgs, thr, max_inland_km=inland_limit_km)
        if poly is None: 
            continue
        poly_wgs = gpd.GeoSeries([poly], crs=dem_crs).to_crs("EPSG:4326").iloc[0]
        results.append({
            "event_id": event_id, "band": band,
            "ru_center":ru_center,"ru_low":ru_low,"ru_high":ru_high,
            "num_points":int(len(snapped)),"method":"runup_IDW+DEM_threshold",
            "geometry": poly_wgs
        })
    if not results:
        coast_buf = coast_line.buffer(1609, cap_style=2).to_crs("EPSG:4326").iloc[0]
        results=[{
            "event_id": event_id,"band":"MED","ru_center":np.nan,"ru_low":np.nan,"ru_high":np.nan,
            "num_points":int(len(snapped)),"method":"coastline_1mile_buffer","geometry":coast_buf
        }]
    return gpd.GeoDataFrame(results, crs="EPSG:4326")

def process_tsunami_data(tsunami_events_csv, tsunami_runups_csv, countries_path, dem_dir, output_folder, inland_limit_km=10, band_percents=(0.2,0.2)):
    try:
        events_df = pd.read_csv(tsunami_events_csv)
        runups_df = pd.read_csv(tsunami_runups_csv)
    except Exception as e:
        logging.error(f"Tsunami CSV load failed: {e}")
        return gpd.GeoDataFrame()

    if not {'latitude','longitude'}.issubset(runups_df.columns):
        logging.error("Runup CSV missing 'latitude'/'longitude'.")
        return gpd.GeoDataFrame()
    if 'runupHt' not in runups_df.columns:
        runups_df['runupHt'] = np.nan

    runups_gdf = gpd.GeoDataFrame(runups_df, geometry=gpd.points_from_xy(runups_df['longitude'], runups_df['latitude']), crs="EPSG:4326")

    try:
        countries = gpd.read_file(countries_path).to_crs("EPSG:4326")
        iso_col = "SOV_A3"
        if iso_col not in countries.columns:
            logging.error(f"Countries layer missing '{iso_col}'.")
            return gpd.GeoDataFrame()
    except Exception as e:
        logging.error(f"Failed loading countries for tsunamis: {e}")
        return gpd.GeoDataFrame()

    try:
        runups_iso = gpd.sjoin(runups_gdf, countries[[iso_col,"geometry"]], how="left", predicate="intersects")
        runups_iso = runups_iso.rename(columns={iso_col:"iso_a3"}).drop(columns="index_right")
    except Exception as e:
        logging.error(f"ISO spatial-join failed for runups: {e}")
        return gpd.GeoDataFrame()

    def _mk_date(row):
        try:
            y = int(row.get('year')) if pd.notna(row.get('year')) else None
            m = int(row.get('month')) if pd.notna(row.get('month')) else 1
            d = int(row.get('day')) if pd.notna(row.get('day')) else 1
            return pd.to_datetime(f"{y}-{m}-{d}", errors='coerce') if y else pd.NaT
        except Exception:
            return pd.NaT
    events_df['date'] = events_df.apply(_mk_date, axis=1)

    results=[]
    for (event_id, iso), grp in runups_iso.groupby(['tsunamiEventId','iso_a3'], dropna=False):
        if pd.isna(event_id) or pd.isna(iso):
            continue
        coast = countries[countries[iso_col]==iso]
        if coast.empty:
            logging.warning(f"No coast polygon for ISO {iso} (event {event_id}).")
            continue
        dem_path = os.path.join(dem_dir, f"{iso}.tif")
        if not os.path.exists(dem_path):
            logging.warning(f"DEM missing for ISO {iso} at {dem_path} (event {event_id}); skipping.")
            continue

        inund = build_tsunami_inundation(
            grp[['latitude','longitude','runupHt','tsunamiEventId','geometry']],
            coast, dem_path, event_id=event_id,
            inland_limit_km=inland_limit_km, band_percents=band_percents
        )
        if inund.empty:
            continue
        ev = events_df[events_df['id']==event_id]
        inund['iso_a3']=iso
        inund['event_type']='tsunami'
        inund['date'] = ev.iloc[0]['date'] if not ev.empty else pd.NaT
        results.append(inund)

    if not results:
        logging.warning("No tsunami inundation polygons produced.")
        return gpd.GeoDataFrame()
    out = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs="EPSG:4326")
    logging.info(f"Produced {len(out)} tsunami polygons.")
    return out
