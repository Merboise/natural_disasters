# tsunami.py
# Orchestrates the tsunami pipeline to produce 12 layers per spec.
from __future__ import annotations

from .bootstrap_gdal import verify_gdal_ready
verify_gdal_ready()

import os, logging, time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Dict

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

from .runups import (
    ensure_wgs84,
    normalize_runups,
    snap_runups_to_coast,
    build_runup_segments,
    infill_runup_segments,
    runup_segments_to_inland_poly,
    buffer_on_land,
)
from .rays import (
    pick_origin,
    cast_rays_hits,
    hits_to_segments,
    ray_density_to_inland_poly,
)

WGS84 = 4326

logger = logging.getLogger("tsunami")

@contextmanager
def _timer(tag: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        logger.debug("[%s] %.1f ms", tag, dt)

def _log_gdf(tag, gdf):
    try:
        n = 0 if gdf is None else len(gdf)
        crs = getattr(gdf, "crs", None)
        gtypes = [] if gdf is None or gdf.empty else list(gdf.geom_type.value_counts().to_dict().items())
        logger.info("[gdf:%s] n=%s crs=%s types=%s", tag, n, crs, gtypes)
    except Exception as e:
        logger.debug("[gdf:%s] summary failed: %r", tag, e)

def read_any(path: str, layer: str | None = None) -> gpd.GeoDataFrame:
    if path is None: return None
    if str(path).lower().endswith(".csv"):
        df = pd.read_csv(path)
        lon_col = next((c for c in df.columns if c.lower() in ("lon","longitude","x")), None)
        lat_col = next((c for c in df.columns if c.lower() in ("lat","latitude","y")), None)
        if lon_col and lat_col:
            return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=WGS84)
        return gpd.GeoDataFrame(df, geometry=[], crs=WGS84)
    else:
        try:
            return gpd.read_file(path, layer=layer, engine="pyogrio")
        except Exception:
            return gpd.read_file(path, layer=layer)

def write_layer_safe(gdf: gpd.GeoDataFrame, gpkg_path: str, layer: str, overwrite: bool = False) -> None:
    if gdf is None: return
    gdf = ensure_wgs84(gdf)
    os.makedirs(os.path.dirname(gpkg_path), exist_ok=True)
    if overwrite and os.path.exists(gpkg_path):
        os.remove(gpkg_path)
    try:
        gdf.to_file(gpkg_path, layer=layer, driver="GPKG", engine="pyogrio", mode="a")
    except Exception:
        gdf.to_file(gpkg_path, layer=layer, driver="GPKG")

@dataclass
class Config:
    runups_csv: str
    events_csv: str
    events_id_col: str
    event_id: int | str
    coast_lines_path: str
    landmask_path: Optional[str]
    height_col: str = "runupHt"
    runups_id_col: str = "tsunamiEventId"
    # snapping
    max_snap_km: float = 15.0
    # segments
    min_ht_m: float = 0.5
    L_min_km: float = 4.0
    beta_km_per_m: float = 8.0
    exp: float = 1.0
    L_max_km: float = 60.0
    merge_tol_m: float = 1000.0
    # infill
    ht_lo_m: float = 0.2
    ht_hi_m: float = 0.5
    eps_km: float = 15.0
    min_samples: int = 2
    majority_threshold: float = 0.5
    same_parent_only: bool = True
    max_bridge_gap_km: float = 20.0
    min_combined_coverage: float = 0.6
    # runups inland
    D_min_km: float = 0.1
    alpha_km_per_m: float = 1.0
    gamma: float = 1.0
    D_max_km: float = 12.0
    sample_step_m: float = 250.0
    # rays
    step_deg: float = 1.0
    max_range_km: float = 3000.0
    simplify_m: float = 10.0
    hit_buffer_m: float = 750.0
    expand_km: float = 125.0
    gap_base_km: float = 20.0
    gap_per_1000km: float = 40.0
    # ray density
    window_km: float = 1000.0
    k_coeff: float = 0.02
    realistic_cap_km: float = 1.0
    hard_cap_km: float = 12.0
    out_gpkg: str = "tsunami_output.gpkg"

def load_origin(events_gdf: gpd.GeoDataFrame, *, events_id_col: str, event_id) -> Point:
    events = ensure_wgs84(events_gdf)

    # 1) resolve the ID column case-insensitively
    cols_lc = {c.lower(): c for c in events.columns}
    if events_id_col.lower() not in cols_lc:
        raise ValueError(f"events-id-col '{events_id_col}' not found. Available: {list(events.columns)}")
    id_col = cols_lc[events_id_col.lower()]

    # 2) normalize IDs on both sides (string, strip, lower)
    want = str(event_id).strip().lower()
    ids_norm = events[id_col].astype(str).str.strip().str.lower()

    hit = events.loc[ids_norm == want]
    if hit.empty:
        # helpful diagnostics
        sample = events[[id_col]].head(10).to_dict(orient="records")
        raise ValueError(f"event_id '{event_id}' not found in '{id_col}'. Sample values: {sample}")

    # 3) prefer explicit longitude/latitude, then lon/lat, then geometry
    row = hit.iloc[0]
    cols_lc_row = {c.lower(): c for c in hit.columns}

    def pick(colnames):
        for k in colnames:
            if k in cols_lc_row:
                return cols_lc_row[k]
        return None

    lon_c = pick(("longitude", "lon", "x"))
    lat_c = pick(("latitude", "lat", "y"))
    if lon_c and lat_c:
        return Point(float(row[lon_c]), float(row[lat_c]))

    if "geometry" in hit.columns and getattr(row, "geometry", None) is not None and hasattr(row.geometry, "x"):
        return row.geometry

    raise ValueError("Cannot derive origin point: no [longitude|lon|x] and [latitude|lat|y] or valid geometry.")


def build_layers(cfg: Config) -> Dict[str, gpd.GeoDataFrame]:
    logger.info("build_layers start event_id=%s", cfg.event_id)
    with _timer("normalize_runups"):
        runups = normalize_runups(cfg.runups_csv, id_col=cfg.runups_id_col, height_col=cfg.height_col,
                                lon_col=getattr(cfg, "lon_col", None), lat_col=getattr(cfg, "lat_col", None))
        _log_gdf("runups_raw", runups)

    with _timer("read_coast_land_events"):
        coast = read_any(cfg.coast_lines_path); _log_gdf("coast_raw", coast)
        landmask = read_any(cfg.landmask_path) if cfg.landmask_path else None; _log_gdf("landmask_raw", landmask)
        events = read_any(cfg.events_csv); _log_gdf("events_raw", events)

    with _timer("origin"):
        origin_pt = load_origin(events, events_id_col=cfg.events_id_col, event_id=cfg.event_id)
        origin = gpd.GeoDataFrame({"event_id":[cfg.event_id]}, geometry=[origin_pt], crs=WGS84)
        _log_gdf("origin", origin)

    with _timer("snap_runups_to_coast"):
        snapped, rejected = snap_runups_to_coast(runups, coast.geometry if isinstance(coast, gpd.GeoDataFrame) else coast,
                                                max_snap_km=cfg.max_snap_km)
        _log_gdf("runups_snapped", snapped); _log_gdf("runups_rejected", rejected)

    with _timer("build_runup_segments"):
        runups_segments = build_runup_segments(snapped, coast.geometry if isinstance(coast, gpd.GeoDataFrame) else coast,
                                            height_col=cfg.height_col, min_ht_m=cfg.min_ht_m,
                                            L_min_km=cfg.L_min_km, beta_km_per_m=cfg.beta_km_per_m,
                                            exp=cfg.exp, L_max_km=cfg.L_max_km, merge_tol_m=cfg.merge_tol_m)
        _log_gdf("runups_segments", runups_segments)

    with _timer("infill_runup_segments"):
        runups_infill = infill_runup_segments(snapped, runups_segments, coast.geometry if isinstance(coast, gpd.GeoDataFrame) else coast,
                                            ht_lo_m=cfg.ht_lo_m, ht_hi_m=cfg.ht_hi_m, eps_km=cfg.eps_km,
                                            min_samples=cfg.min_samples, majority_threshold=cfg.majority_threshold,
                                            same_parent_only=cfg.same_parent_only, max_bridge_gap_km=cfg.max_bridge_gap_km,
                                            min_combined_coverage=cfg.min_combined_coverage)
        _log_gdf("runups_infill", runups_infill)

    with _timer("runups_impact_poly"):
        segments_union = (pd.concat([runups_segments, runups_infill], ignore_index=True)
                        if not runups_infill.empty else runups_segments)
        runups_impact = runup_segments_to_inland_poly(segments_union, snapped, landmask,
                                                    height_col=cfg.height_col, D_min_km=cfg.D_min_km,
                                                    alpha_km_per_m=cfg.alpha_km_per_m, gamma=cfg.gamma,
                                                    D_max_km=cfg.D_max_km, sample_step_m=cfg.sample_step_m)
        _log_gdf("runups_impact_poly", runups_impact)

    with _timer("rays_hits"):
        ray_hits = cast_rays_hits(coast.geometry if isinstance(coast, gpd.GeoDataFrame) else coast, origin.geometry.iloc[0],
                                step_deg=cfg.step_deg, max_range_km=cfg.max_range_km,
                                simplify_m=cfg.simplify_m, hit_buffer_m=cfg.hit_buffer_m, engine="aeqd")
        _log_gdf("rays_hits", ray_hits)

    with _timer("rays_segments_linestring"):
        rays_segments = hits_to_segments(ray_hits, coast.geometry if isinstance(coast, gpd.GeoDataFrame) else coast,
                                        expand_km=cfg.expand_km, gap_base_km=cfg.gap_base_km, gap_per_1000km=cfg.gap_per_1000km)
        _log_gdf("rays_segments_linestring", rays_segments)

    with _timer("rays_impact_poly"):
        rays_impact = ray_density_to_inland_poly(ray_hits, coast.geometry if isinstance(coast, gpd.GeoDataFrame) else coast, landmask,
                                                window_km=cfg.window_km, k_coeff=cfg.k_coeff, D_min_km=cfg.D_min_km,
                                                realistic_cap_km=cfg.realistic_cap_km, hard_cap_km=cfg.hard_cap_km,
                                                sample_step_m=cfg.sample_step_m)
        _log_gdf("rays_impact_poly", rays_impact)

    with _timer("merge_impacts"):
        both = []
        if runups_impact is not None and not runups_impact.empty: both.append(runups_impact.unary_union)
        if rays_impact is not None and not rays_impact.empty: both.append(rays_impact.unary_union)
        merged_impact = gpd.GeoDataFrame(geometry=[unary_union(both)] if both else [], crs=WGS84)

        seg_union = []
        if not runups_segments.empty: seg_union.append(runups_segments.unary_union)
        if not rays_segments.empty: seg_union.append(rays_segments.unary_union)
        base_geom = unary_union(seg_union) if seg_union else (merged_impact.unary_union if not merged_impact.empty else None)
        if base_geom is not None:
            conservative = gpd.GeoDataFrame(geometry=[gpd.GeoSeries([base_geom], crs=WGS84).to_crs(3857).buffer(12000.0).to_crs(WGS84).unary_union], crs=WGS84)
        else:
            conservative = gpd.GeoDataFrame(geometry=[], crs=WGS84)
        if landmask is not None and not conservative.empty:
            conservative = gpd.overlay(conservative, landmask, how="intersection", keep_geom_type=True)
        _log_gdf("merged_impact_poly", merged_impact)
        _log_gdf("merged_conservative_poly", conservative)
    logger.info("build_layers done")

    return {
        "origin": origin,
        "runups_snapped": snapped,
        "runups_rejected": rejected,
        "runups_segments_linestring": runups_segments,
        "runups_infill": runups_infill,
        "runups_impact_poly": runups_impact,
        "rays_hits": ray_hits,
        "rays_segments_linestring": rays_segments,
        "rays_impact_poly": rays_impact,
        "merged_impact_poly": merged_impact,
        "merged_conservative_poly": conservative,
    }

def run_cli(
    runups: str,
    coastlines: str,
    landmask: Optional[str],
    events: str,
    events_id_col: str,
    event_id,
    runups_id_col: str,
    height_col: str,
    out_gpkg: str,
):
    cfg = Config(
        runups_csv=runups,
        events_csv=events,
        events_id_col=events_id_col,
        event_id=event_id,
        coast_lines_path=coastlines,
        landmask_path=landmask,
        height_col=height_col,
        runups_id_col=runups_id_col,
        out_gpkg=out_gpkg,
    )
    layers = build_layers(cfg)
    first = True
    for name, gdf in layers.items():
        write_layer_safe(gdf, out_gpkg, layer=name, overwrite=first)
        first = False

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--runups", required=True)
    p.add_argument("--coastlines", required=True)
    p.add_argument("--landmask", required=False)
    p.add_argument("--events", required=True)
    p.add_argument("--events-id-col", required=True)
    p.add_argument("--event-id", required=True)
    p.add_argument("--runups-id-col", default="tsunamiEventId")
    p.add_argument("--height-col", default="runupHt")
    p.add_argument("--out-gpkg", required=True)
    p.add_argument("--log", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    args = p.parse_args()

    level = getattr(logging, args.log, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    run_cli(
        runups=args.runups,
        coastlines=args.coastlines,
        landmask=args.landmask,
        events=args.events,
        events_id_col=args.events_id_col,
        event_id=args.event_id,
        runups_id_col=args.runups_id_col,
        height_col=args.height_col,
        out_gpkg=args.out_gpkg,
    )
