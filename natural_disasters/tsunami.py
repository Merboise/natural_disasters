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
from shapely.geometry import Point, box
from shapely.ops import unary_union

from .runups import (
    ensure_wgs84,
    normalize_runups,
    snap_runups_to_coast,
    build_runup_segments,
    infill_runup_segments,
    infill_bridge_rulebased,
    runup_segments_to_inland_poly,
    buffer_on_land,
)
from .rays import (
    pick_origin,
    cast_rays_hits,
    hits_to_segments,
    ray_density_to_inland_poly,
)
from .infill_config import InfillConfig

WGS84 = 4326
_INFILL_OPTS: InfillConfig | None = None
logger = logging.getLogger("tsunami")

from .tsunami_helpers import _timer, _log_gdf
from .tsunami_helpers import derive_r_decay

# --- Internal helpers for performant unions and overlay ---
def _union_3857_simplify(geoms_wgs, *, simplify_m: float = 10.0, tag: str = "union"):
    import time as _time
    t0 = _time.perf_counter()
    if not geoms_wgs:
        return None
    gs = gpd.GeoSeries(geoms_wgs, crs=WGS84).to_crs(3857)
    # repair invalids
    try:
        from shapely.validation import make_valid as _make_valid
        gs = gs.apply(_make_valid)
    except Exception:
        gs = gs.buffer(0)
    # simplify lightly to speed up unions; keep topology
    if simplify_m and simplify_m > 0:
        gs = gs.simplify(float(simplify_m), preserve_topology=True)
    # union (prefer union_all)
    try:
        from shapely import union_all as _union_all
        merged = _union_all(gs.values.tolist())
    except Exception:
        merged = unary_union(gs.values.tolist())
    out = gpd.GeoSeries([merged], crs=3857).to_crs(WGS84).iloc[0]
    dt = (_time.perf_counter() - t0) * 1000.0
    logger.debug("[%s] %.1f ms (n=%d)", tag, dt, len(geoms_wgs))
    return out

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
    max_snap_km: float = 25.0
    # segments
    min_ht_m: float = 0.5
    L_min_km: float = 4.0
    beta_km_per_m: float = 8.0
    exp: float = 1.0
    L_max_km: float = 60.0
    merge_tol_m: float = 1000.0
    # segments length multiplier (scale alongshore extents)
    segments_length_mult: float = 1.0
    # infill
    ht_lo_m: float = 0.2
    ht_hi_m: float = 0.5
    eps_km: float = 15.0
    min_samples: int = 2
    majority_threshold: float = 0.5
    same_parent_only: bool = False
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
    # conservative merge buffer (km)
    conservative_buffer_km: float = 12.0
    # controls whether to run merge_impacts (unions/overlay)
    do_merge_impacts: bool = True

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
        runups = normalize_runups(
            cfg.runups_csv,
            id_col=cfg.runups_id_col,
            height_col=cfg.height_col,
            lon_col=getattr(cfg, "lon_col", None),
            lat_col=getattr(cfg, "lat_col", None),
            event_id_filter=cfg.event_id,
        )
        _log_gdf("runups_raw", runups)

    with _timer("read_coast_land_events"):
        coast = read_any(cfg.coast_lines_path)
        _log_gdf("coast_raw", coast)

        landmask = read_any(cfg.landmask_path) if cfg.landmask_path else None
        _log_gdf("landmask_raw", landmask)

        events = read_any(cfg.events_csv)
        _log_gdf("events_raw", events)

    with _timer("origin"):
        origin_pt = load_origin(
            events,
            events_id_col=cfg.events_id_col,
            event_id=cfg.event_id,
        )
        origin = gpd.GeoDataFrame(
            {"event_id": [cfg.event_id]},
            geometry=[origin_pt],
            crs=WGS84,
        )
        _log_gdf("origin", origin)

    with _timer("snap_runups_to_coast"):
        snapped, rejected = snap_runups_to_coast(
            runups,
            coast.geometry if isinstance(coast, gpd.GeoDataFrame) else coast,
            max_snap_km=cfg.max_snap_km,
        )
        _log_gdf("runups_snapped", snapped)
        _log_gdf("runups_rejected", rejected)

    # Derive data-driven ray decay and search range from runups
    with _timer("derive_r_decay"):
        R0_km, R_search_km, rdiag = derive_r_decay(snapped, origin.geometry.iloc[0])
        logger.info(
            "rdecay: bins_kept=%s bins_dropped=%s R0_km=%.0f R_search_km=%.0f fallback=%s",
            rdiag.get("bins_kept"), rdiag.get("bins_dropped"), R0_km, R_search_km, rdiag.get("fallback")
        )
    with _timer("build_runup_segments"):
        runups_segments = build_runup_segments(
            snapped,
            coast.geometry if isinstance(coast, gpd.GeoDataFrame) else coast,
            height_col=cfg.height_col,
            height_meters_min=cfg.min_ht_m,
            alongshore_km_min=cfg.L_min_km,
            alongshore_km_max=cfg.L_max_km,
            km_per_meter_factor=cfg.beta_km_per_m,
            height_exponent=cfg.exp,
            merge_tol_meters=cfg.merge_tol_m,
            length_multiplier=cfg.segments_length_mult,
        )
        _log_gdf("runups_segments", runups_segments)

    with _timer("infill_runup_segments"):
        # Compute rays first so we can use hits as amplifiers in rule-based infill
        ray_hits = cast_rays_hits(
            coast.geometry if isinstance(coast, gpd.GeoDataFrame) else coast,
            origin.geometry.iloc[0],
            step_deg=cfg.step_deg,
            max_range_km=R_search_km if R_search_km else cfg.max_range_km,
            simplify_m=cfg.simplify_m,
            hit_buffer_m=cfg.hit_buffer_m,
            engine="aeqd",
        )
        _log_gdf("rays_hits", ray_hits)

        runups_infill = infill_bridge_rulebased(
            snapped,
            runups_segments,
            coast.geometry if isinstance(coast, gpd.GeoDataFrame) else coast,
            ray_hits_gdf=ray_hits,
            delta_max_km=cfg.infill_delta_max_km,
            eps_edge_km=cfg.infill_eps_edge_km,
            ht_lo_m=cfg.ht_lo_m, ht_hi_m=cfg.ht_hi_m,
            same_parent_only=not args.infill_allow_cross_parent,
            log_decisions=args.infill_log_decisions,
            log_rejects_limit=args.infill_log_rejects_limit,
        )
        _log_gdf("runups_infill", runups_infill)

    with _timer("runups_impact_poly"):
        segments_union = (
            pd.concat([runups_segments, runups_infill], ignore_index=True)
            if not runups_infill.empty
            else runups_segments
        )
        runups_impact = runup_segments_to_inland_poly(
            segments_union,
            snapped,
            landmask,
            height_col=cfg.height_col,
            D_min_km=cfg.D_min_km,
            alpha_km_per_m=cfg.alpha_km_per_m,
            gamma=cfg.gamma,
            D_max_km=cfg.D_max_km,
            sample_step_m=cfg.sample_step_m,
        )
        _log_gdf("runups_impact_poly", runups_impact)

    with _timer("rays_segments_linestring"):
        rays_segments = hits_to_segments(
            ray_hits,
            coast.geometry if isinstance(coast, gpd.GeoDataFrame) else coast,
            expand_km=cfg.expand_km,
            gap_base_km=cfg.gap_base_km,
            gap_per_1000km=cfg.gap_per_1000km,
        )
        _log_gdf("rays_segments_linestring", rays_segments)

    with _timer("rays_impact_poly"):
        rays_impact = ray_density_to_inland_poly(
            ray_hits,
            coast.geometry if isinstance(coast, gpd.GeoDataFrame) else coast,
            landmask,
            window_km=cfg.window_km,
            k_coeff=cfg.k_coeff,
            D_min_km=cfg.D_min_km,
            realistic_cap_km=cfg.realistic_cap_km,
            hard_cap_km=cfg.hard_cap_km,
            sample_step_m=cfg.sample_step_m,
            origin_pt_wgs=origin.geometry.iloc[0],
            r0_km=R0_km,
        )
        _log_gdf("rays_impact_poly", rays_impact)

    if cfg.do_merge_impacts:
        with _timer("merge_impacts"):
            both = []
            if runups_impact is not None and not runups_impact.empty:
                both.append(runups_impact.geometry.values.tolist())
            if rays_impact is not None and not rays_impact.empty:
                both.append(rays_impact.geometry.values.tolist())
            merged_geom = None
            if both:
                flat = [g for lst in both for g in lst]
                merged_geom = _union_3857_simplify(flat, simplify_m=10.0, tag="impacts_union_3857")
            merged_impact = gpd.GeoDataFrame(
                geometry=[merged_geom] if merged_geom else [],
                crs=WGS84,
            )

            seg_union_geoms = []
            if not runups_segments.empty:
                seg_union_geoms.extend(runups_segments.geometry.values.tolist())
            if not rays_segments.empty:
                seg_union_geoms.extend(rays_segments.geometry.values.tolist())
            if seg_union_geoms:
                base_geom = _union_3857_simplify(seg_union_geoms, simplify_m=10.0, tag="segments_union_3857")
            else:
                base_geom = (merged_impact.unary_union if not merged_impact.empty else None)
            if base_geom is not None:
                import time as _time
                t0b = _time.perf_counter()
                cons_3857 = gpd.GeoSeries([base_geom], crs=WGS84).to_crs(3857)
                try:
                    from shapely.validation import make_valid as _make_valid
                    cons_3857 = cons_3857.apply(_make_valid)
                except Exception:
                    cons_3857 = cons_3857.buffer(0)
                cons_3857 = cons_3857.buffer(float(cfg.conservative_buffer_km) * 1000.0)
                conservative = gpd.GeoDataFrame(geometry=[cons_3857.unary_union], crs=3857).to_crs(WGS84)
                logger.debug("[conservative_buffer] %.1f ms", (_time.perf_counter() - t0b) * 1000.0)
            else:
                conservative = gpd.GeoDataFrame(geometry=[], crs=WGS84)

            if landmask is not None and not conservative.empty:
                # Prefilter landmask by conservative bbox and intersect with a unioned subset for speed
                b = conservative.total_bounds
                try:
                    sidx = landmask.sindex
                    cand = list(sidx.intersection((b[0], b[1], b[2], b[3])))
                    land_sub = landmask.iloc[cand]
                except Exception:
                    land_sub = landmask
                lm_union = None
                if land_sub is not None and not land_sub.empty:
                    lm_union = _union_3857_simplify(land_sub.geometry.values.tolist(), simplify_m=10.0, tag="landmask_union_3857")
                if lm_union is not None:
                    inter = conservative.geometry.iloc[0].intersection(lm_union)
                    conservative = gpd.GeoDataFrame(geometry=[inter] if (inter and not inter.is_empty) else [], crs=WGS84)

            _log_gdf("merged_impact_poly", merged_impact)
            _log_gdf("merged_conservative_poly", conservative)
    else:
        merged_impact = gpd.GeoDataFrame(geometry=[], crs=WGS84)
        conservative = gpd.GeoDataFrame(geometry=[], crs=WGS84)
        logger.info("merge_impacts skipped by configuration (testing mode)")

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
    # Optional segment tuning
    segments_min_ht_m: float | None = None,
    segments_length_mult: float | None = None,
    # Caching
    use_cache: bool = False,
    # Skip heavy merge_impacts (testing speedup)
    skip_merge_impacts: bool = False,
    # Optional infill tuning overrides
    infill_eps_km: float | None = None,
    infill_min_samples: int | None = None,
    infill_majority: float | None = None,
    infill_min_combined_coverage: float | None = None,
    infill_allow_zero_cover: bool = False,
    infill_one_side: bool = False,
    infill_accept_singleton: bool = False,
    infill_log_rejects_limit: int = 50,
    infill_seed_without_base: bool = False,
    infill_seed_min_km: float = 1.0,
    bridge_unconditional_under_km: float | None = None,
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
    # Apply optional CLI overrides onto cfg
    if infill_eps_km is not None:
        cfg.eps_km = float(infill_eps_km)
    if infill_min_samples is not None:
        cfg.min_samples = int(infill_min_samples)
    if infill_majority is not None:
        cfg.majority_threshold = float(infill_majority)
    if infill_min_combined_coverage is not None:
        cfg.min_combined_coverage = float(infill_min_combined_coverage)

    # Build infill options object
    global _INFILL_OPTS
    _INFILL_OPTS = InfillConfig(
        bridge_allow_zero_cover=bool(infill_allow_zero_cover),
        bridge_one_side=bool(infill_one_side),
        accept_singleton=bool(infill_accept_singleton),
        log_rejects_limit=int(infill_log_rejects_limit or 50),
        seed_without_base=bool(infill_seed_without_base),
        seed_min_km=float(infill_seed_min_km or 1.0),
        bridge_unconditional_under_km=bridge_unconditional_under_km,
    )
    # Apply optional segment threshold
    if segments_min_ht_m is not None:
        cfg.min_ht_m = float(segments_min_ht_m)
    if segments_length_mult is not None:
        cfg.segments_length_mult = float(segments_length_mult)

    # Set cache flag for downstream modules that opt-in via env
    if use_cache:
        os.environ["USE_CACHE"] = "1"

    # Testing toggle: skip heavy merge_impacts
    cfg.do_merge_impacts = (not skip_merge_impacts)

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
    # Optional segment tuning
    p.add_argument("--segments-min-ht-m", type=float)
    p.add_argument("--segments-length-mult", type=float, help="Scale factor for alongshore runup segments (e.g., 2.0 to double)")
    p.add_argument("--skip-merge-impacts", action="store_true", help="Skip unions/overlay to speed up testing")
    # Always use in-memory cache for coastlines; flag retained for CLI compatibility
    p.add_argument("--use-cache", action="store_true")
    # Optional infill tuning (CLI testing)
    p.add_argument("--infill-eps-km", type=float)
    p.add_argument("--infill-min-samples", type=int)
    p.add_argument("--infill-majority", type=float)
    p.add_argument("--infill-min-combined-coverage", type=float)
    p.add_argument("--infill-allow-zero-cover", action="store_true")
    p.add_argument("--infill-one-side", action="store_true")
    p.add_argument("--infill-accept-singleton", action="store_true")
    p.add_argument("--infill-log-rejects-limit", type=int, default=50)
    p.add_argument("--infill-seed-without-base", action="store_true")
    p.add_argument("--infill-seed-min-km", type=float, default=1.0)
    p.add_argument("--bridge-unconditional-under-km", type=float, help="Accept any base→base gap <= this length (km) without amplifiers or scoring")
    p.add_argument("--infill-allow-cross-parent", action="store_true")
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
        segments_min_ht_m=args.segments_min_ht_m,
        segments_length_mult=args.segments_length_mult,
        use_cache=args.use_cache,
        skip_merge_impacts=args.skip_merge_impacts,
        infill_eps_km=args.infill_eps_km,
        infill_min_samples=args.infill_min_samples,
        infill_majority=args.infill_majority,
        infill_min_combined_coverage=args.infill_min_combined_coverage,
        infill_allow_zero_cover=args.infill_allow_zero_cover,
        infill_one_side=args.infill_one_side,
        infill_accept_singleton=args.infill_accept_singleton,
        infill_log_rejects_limit=args.infill_log_rejects_limit,
        infill_seed_without_base=args.infill_seed_without_base,
        infill_seed_min_km=args.infill_seed_min_km,
        bridge_unconditional_under_km=args.bridge_unconditional_under_km,
    )
