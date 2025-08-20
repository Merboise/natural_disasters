# ----
# tsunamis.py
# ----

import os, logging, math, tempfile
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

import fiona
from memory_profiler import profile

# env (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from .helpers import diagnose_geom, ISO_COL
from .dem_manager_local import build_local_dem_index, select_tiles_for_geom

DEFAULT_DEM_LOCAL_ROOT = os.getenv("DEM_LOCAL_ROOT")


def _snap_points_to_coast(runups_gdf, coast_gdf, max_km=10):
    """
    Snap runup points to the nearest coastline within max_km.
    Returns (snapped_points_gdf, coast_lines_geoseries)
    """
    # 1) Build a unified lineal coastline
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

    # IMPORTANT: reset index to keep alignment simple
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
        return gpd.GeoSeries(polys, crs=dem_crs).union_all(), dem_crs

@profile
@profile
def stream_flood_polygons_from_dem(
    dem_path: str,
    strip_wgs,                  # WGS84 geometry describing the coastal strip
    min_elev_m: float,          # Emin (e.g., -1.0)
    max_elev_m: float,          # Hmax (e.g., ceil(ru_high + 1.0))
    inland_limit_km: float,     # inland cap for clip
    tmp_vector_path: str,       # on-disk target (GeoPackage path)
    layer_name: str = "flood",
    simplify_m: float = 0.0,    # optional geometry simplify in meters
    min_area_m2: float = 0.0,   # drop tiny slivers (0 = keep all)
    mode: str = "w",            # "w" first tile, then "a" for subsequent tiles
):
    """
    Streamed threshold→polygon pipeline for MERIT-Hydro DEMs (float32 EGM96, nodata=-9999).
    Reads each raster block, builds a 0/1 flood mask in RAM (no full-array loads),
    polygonizes positives, clips to coastal strip & inland cap, and appends to a GeoPackage.
    """
    if mode == "w" and os.path.exists(tmp_vector_path):
        try:
            os.remove(tmp_vector_path)
        except OSError:
            pass

    schema = {"geometry": "Polygon", "properties": {"src": "str:8"}}
    crs_wgs = "EPSG:4326"

    with rasterio.open(dem_path) as ds, \
         fiona.open(tmp_vector_path, mode, driver="GPKG",
                    layer=layer_name, schema=schema, crs=crs_wgs) as sink:

        nodata = ds.nodata if ds.nodata is not None else -9999.0
        Hmax_f32 = np.float32(max_elev_m)
        Emin_f32 = np.float32(min_elev_m)

        # Precompute clips in DEM CRS + fast bbox for culling blocks
        strip_dem = gpd.GeoSeries([strip_wgs], crs="EPSG:4326").to_crs(ds.crs).iloc[0]
        inland_clip_dem = gpd.GeoSeries([strip_wgs], crs="EPSG:4326") \
                              .to_crs("EPSG:3857").buffer(inland_limit_km * 1000.0) \
                              .to_crs(ds.crs).iloc[0]
        ic_minx, ic_miny, ic_maxx, ic_maxy = inland_clip_dem.bounds

        for (j, i), window in ds.block_windows(1):
            # Fast bbox reject
            left, bottom, right, top = rasterio.windows.bounds(window, ds.transform)
            if (right < ic_minx) or (left > ic_maxx) or (top < ic_miny) or (bottom > ic_maxy):
                continue

            # Tiny preview to skip obvious non-hits
            thumb = ds.read(
                1, window=window, out_shape=(1, 64, 64),
                resampling=Resampling.nearest, masked=False
            ).astype(np.float32, copy=False)
            valid_thumb = (thumb != nodata)
            if not valid_thumb.any():
                continue
            tmin = np.min(thumb[valid_thumb]); tmax = np.max(thumb[valid_thumb])
            if (tmin > Hmax_f32) or (tmax < Emin_f32):
                continue

            # Read the real block and build mask
            a = ds.read(1, window=window, masked=False).astype(np.float32, copy=False)
            valid = (a != nodata)
            np.logical_and(valid, a >= Emin_f32, out=valid)
            np.logical_and(valid, a <= Hmax_f32, out=valid)
            if not valid.any():
                continue
            mask = valid.astype(np.uint8, copy=False)

            # Polygonize positives (mask==1); note window transform!
            for geom, v in features.shapes(mask, transform=rasterio.windows.transform(window, ds.transform)):
                if v != 1:
                    continue
                try:
                    poly_dem = shape(geom)
                except Exception:
                    continue
                # Clip tests
                if not (poly_dem.intersects(inland_clip_dem) and poly_dem.intersects(strip_dem)):
                    continue

                # Optional min-area filter to cut tiny slivers early
                if min_area_m2 > 0.0:
                    area_m2 = gpd.GeoSeries([poly_dem], crs=ds.crs).to_crs("EPSG:3857").area.iloc[0]
                    if area_m2 < float(min_area_m2):
                        continue

                # To WGS84 (+ optional simplify)
                poly_wgs = gpd.GeoSeries([poly_dem], crs=ds.crs).to_crs(crs_wgs).iloc[0]
                if simplify_m and simplify_m > 0:
                    poly_wgs = gpd.GeoSeries([poly_wgs], crs=crs_wgs) \
                                  .to_crs("EPSG:3857").buffer(0).simplify(simplify_m) \
                                  .to_crs(crs_wgs).iloc[0]
                    if poly_wgs.is_empty:
                        continue

                sink.write({"geometry": mapping(poly_wgs), "properties": {"src": "dem"}})

    return tmp_vector_path

def dissolve_vector_to_geom(path, layer_name="flood", dissolve_batch=5000):
    """
    Read polygons in batches from path:layer and return a single dissolved Shapely geometry.
    Keeps memory bounded by dissolving incrementally.
    """
    unions = []
    batch = []
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

    if not unions:
        return None
    # Final small union
    return unary_union(unions)

def _bands_from_center(height_m, pct_low=0.2, pct_high=0.2):
    return max(0.0, height_m*(1.0-pct_low)), height_m, height_m*(1.0+pct_high)

@profile
def build_tsunami_inundation(
    runups_gdf,
    coast_gdf,
    event_id=None,
    sector_source_pt=None,          # shapely Point (WGS84) or None
    inland_limit_km=10,
    band_percents=(0.2, 0.2),
    dem_path=None,                  # legacy single-DEM path (unused when dem_local_root is set)
    use_dem: bool = True,
    dem_local_root: str | None = None,
    dem_tile_size_deg: int = 5,
    tmp_dir: str | None = None,
):
    """
    Returns GeoDataFrame with bands {LOW, MED, HIGH}.
    If use_dem and local tiles available => streamed DEM threshold polygonization with MERIT-aware logic.
    Otherwise => slope-based coastline buffers (no DEM).
    Sector clipping applied when sector_source_pt is provided.
    """
    tmp_dir = tmp_dir or tempfile.gettempdir()

    r = runups_gdf.copy()
    if event_id is not None and "tsunamiEventId" in r.columns:
        r = r[r["tsunamiEventId"] == event_id].copy()
    if r.empty:
        return gpd.GeoDataFrame(
            columns=["event_id","band","ru_center","ru_low","ru_high","num_points","method","geometry"],
            crs="EPSG:4326", geometry="geometry"
        )

    if r.geometry.name != "geometry":
        r = gpd.GeoDataFrame(r, geometry=gpd.points_from_xy(r["longitude"], r["latitude"]), crs="EPSG:4326")

    coast = coast_gdf.to_crs("EPSG:4326")

    # Snap & IDW alongshore
    snapped, coast_line = _snap_points_to_coast(r, coast)
    if snapped.empty:
        hull = r.unary_union.convex_hull.buffer(1609)  # ~1 mile
        if sector_source_pt is not None:
            hull = sector_clip(hull, hull.centroid, sector_source_pt, half_angle_deg=60, max_range_km=inland_limit_km)
        return gpd.GeoDataFrame([{
            "event_id": event_id, "band":"MED","ru_center":np.nan,"ru_low":np.nan,"ru_high":np.nan,
            "num_points":0,"method":"points_hull_1mile","geometry":hull
        }], crs="EPSG:4326")

    coast_samples = _idw_alongshore(snapped, coast_line)
    if coast_samples.empty:
        hull = r.unary_union.convex_hull.buffer(1609)
        if sector_source_pt is not None:
            hull = sector_clip(hull, hull.centroid, sector_source_pt, half_angle_deg=60, max_range_km=inland_limit_km)
        return gpd.GeoDataFrame([{
            "event_id": event_id, "band":"MED","ru_center":np.nan,"ru_low":np.nan,"ru_high":np.nan,
            "num_points":len(snapped),"method":"points_hull_1mile","geometry":hull
        }], crs="EPSG:4326")

    # Representative runup height (meters a.s.l., matches MERIT’s EGM96)
    ru_center = float(np.nanmedian(coast_samples["ru_m"]))
    if not np.isfinite(ru_center):
        fallback_center = float(np.nanmedian(snapped.get("runupHt"))) if "runupHt" in snapped.columns else np.nan
        ru_center = fallback_center if np.isfinite(fallback_center) else 1.0

    # Bands
    low_pct, high_pct = band_percents
    ru_low  = max(0.0, ru_center * (1.0 - low_pct))
    ru_high = ru_center * (1.0 + high_pct)

    # MERIT-aware thresholds
    Emin = -1.0
    Hmax_top = float(np.ceil(ru_high + 1.0))  # safety margin

    # Coastal strip (compute BEFORE tile selection)
    strip_m  = _coastal_strip(coast_line.to_crs("EPSG:3857").iloc[0], width_m=2000)
    strip_wgs = gpd.GeoSeries([strip_m], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]

    # Find intersecting tiles (no in-memory mosaic)
    tiles = gpd.GeoDataFrame()
    if use_dem and dem_local_root:
        try:
            index_gdf = build_local_dem_index(dem_local_root, tile_size_deg=dem_tile_size_deg, suffix='_elv.tif')
            tiles = select_tiles_for_geom(index_gdf, strip_wgs)
            if tiles.empty:
                logging.warning(f"No local DEM tiles intersect coastal strip (event {event_id}).")
        except Exception as e:
            logging.warning(f"Local DEM tile search failed (event {event_id}): {e}")

    results = []

    # --- DEM streaming branch ---
    if use_dem and not tiles.empty:
        for band, thr in [("LOW", ru_low), ("MED", ru_center), ("HIGH", ru_high)]:
            # A small per-band margin around the target threshold
            # Use the same Emin (allows ~0 to slight negative ocean values)
            Hmax = float(thr) + 1.5
            tmp_gpkg = os.path.join(tmp_dir, f"_tmp_flood_{event_id}_{band}.gpkg")

            first = True
            for _, t in tiles.iterrows():
                stream_flood_polygons_from_dem(
                    dem_path=t["path"],
                    strip_wgs=strip_wgs,
                    min_elev_m=Emin,
                    max_elev_m=Hmax,
                    inland_limit_km=inland_limit_km,
                    tmp_vector_path=tmp_gpkg,
                    layer_name="flood",
                    simplify_m=0.0,
                    min_area_m2=200.0,   # small sliver filter; adjust as needed
                    mode=("w" if first else "a")
                )
                first = False

            dissolved = dissolve_vector_to_geom(tmp_gpkg, layer_name="flood", dissolve_batch=4000)
            try:
                os.remove(tmp_gpkg)
            except OSError:
                pass

            poly_wgs = dissolved if (dissolved and not dissolved.is_empty) \
                       else coast_line.buffer(1609, cap_style=2).to_crs("EPSG:4326").iloc[0]

            if sector_source_pt is not None and not poly_wgs.is_empty:
                poly_wgs = sector_clip(poly_wgs, strip_wgs.centroid, sector_source_pt,
                                       half_angle_deg=60, max_range_km=inland_limit_km)

            results.append({
                "event_id": event_id, "band": band,
                "ru_center": ru_center, "ru_low": ru_low, "ru_high": ru_high,
                "num_points": int(len(snapped)),
                "method": "runup_IDW + DEM_threshold(binary)",
                "geometry": poly_wgs
            })

    # --- Fallback: no DEM (or no tiles) → slope-based buffers ---
    if not results:
        slope = 0.015  # 1.5% nominal coastal slope
        D_med_km  = float(np.clip(ru_center / max(slope, 1e-3) / 1000.0, 0.5, inland_limit_km))
        D_low_km  = max(0.25, 0.8 * D_med_km)
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

@profile
def process_tsunami_data(
    tsunami_events_csv,
    tsunami_runups_csv,
    countries_path,
    dem_dir,
    inland_limit_km=10,
    band_percents=(0.2, 0.2),
    use_dem: bool = True,
    dem_local_root: str | None = None,
    dem_tile_size_deg: int = 5,
    output_folder: str | None = None,
    tmp_dir: str | None = None,
):
    dem_local_root = dem_local_root or DEFAULT_DEM_LOCAL_ROOT
    tmp_dir = tmp_dir or output_folder or tempfile.gettempdir()
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
        if ISO_COL not in countries.columns:
            logging.error(f"Countries layer missing '{ISO_COL}'.")
            return gpd.GeoDataFrame()
    except Exception as e:
        logging.error(f"Failed loading countries for tsunamis: {e}")
        return gpd.GeoDataFrame()

    try:
        runups_iso = gpd.sjoin(runups_gdf, countries[[ISO_COL,"geometry"]], how="left", predicate="intersects")
        runups_iso = runups_iso.rename(columns={ISO_COL:"iso_a3"}).drop(columns="index_right")
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
        coast = countries[countries[ISO_COL]==iso]
        if coast.empty:
            logging.warning(f"No coast polygon for ISO {iso} (event {event_id}).")
            continue

        # legacy per-ISO DEM path (only if not using local tiles)
        legacy_dem_path = None
        if not dem_local_root and dem_dir:
            legacy_dem_path = os.path.join(dem_dir, f"{iso}.tif")

        inund = build_tsunami_inundation(
        runups_gdf=grp[['latitude','longitude','runupHt','tsunamiEventId','geometry']],
        coast_gdf=coast,
        event_id=event_id,
        inland_limit_km=inland_limit_km,
        band_percents=band_percents,
        use_dem=True,
        dem_local_root=dem_local_root,       # <— your MERIT root folder (from .env)
        dem_tile_size_deg=dem_tile_size_deg, # 5 or 10 depending on your naming scheme
        tmp_dir=tmp_dir                      # e.g., output_folder to keep temp near outputs
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
