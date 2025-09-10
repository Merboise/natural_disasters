# runups.py
# Full implementation: snapping, segments, infill, inland polygons.
from __future__ import annotations
import math, logging, time
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict
from contextlib import contextmanager

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
from shapely import STRtree

WGS84 = 4326
WEBM = 3857

logger = logging.getLogger("tsunami.runups")

@contextmanager
def _timer(tag: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.debug("[%s] %.1f ms", tag, (time.perf_counter()-t0)*1000)

def _log_gdf(tag, gdf):
    n = 0 if gdf is None else len(gdf)
    crs = getattr(gdf, "crs", None)
    logger.info("[gdf:%s] n=%s crs=%s", tag, n, crs)

# ---------- helpers ----------

def ensure_wgs84(g: gpd.GeoSeries | gpd.GeoDataFrame) -> gpd.GeoSeries | gpd.GeoDataFrame:
    if g is None:
        return g
    if g.crs is None:
        g = g.set_crs(WGS84)
    elif int(g.crs.to_epsg() or 0) != WGS84:
        g = g.to_crs(WGS84)
    return g

def to_metric(g): return g.to_crs(WEBM)
def to_wgs(g): return g.to_crs(WGS84)


def _coast_parts_3857(coast_lines: gpd.GeoSeries):
    coast_lines = ensure_wgs84(coast_lines)
    gdf = gpd.GeoDataFrame(geometry=coast_lines, crs=WGS84).explode(index_parts=False, ignore_index=True)
    gdf_m = gdf.to_crs(WEBM)
    parts = []
    parents = []
    for pid, geom in enumerate(gdf_m.geometry.values):
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            parts.append(geom); parents.append(pid)
        elif geom.geom_type == "MultiLineString":
            for sub in geom.geoms:
                parts.append(sub); parents.append(pid)
    return parts, parents

def _substring(line_m: LineString, s0: float, s1: float) -> Optional[LineString]:
    if s1 <= s0: return None
    try:
        from shapely.ops import substring
        return substring(line_m, s0, s1, normalized=False)
    except Exception:
        n = max(2, int((s1 - s0) / 50.0))
        pts = [line_m.interpolate(t) for t in np.linspace(s0, s1, n)]
        return LineString(pts)

def _project_to_line_s(line_m: LineString, pt_m: Point) -> float:
    return float(line_m.project(pt_m))

def _tree_query_pairs(tree: STRtree, geoms_list, target_geom, max_m):
    """Return list[(idx, geom)] robustly across Shapely 1.x and 2.x."""
    # Preferred: query with a buffered target for proximity prefilter
    try:
        cand = tree.query(target_geom.buffer(max_m))
    except Exception:
        cand = tree.query(target_geom)
    # Shapely 2.x: numpy array of indices
    if len(cand) and isinstance(cand[0], (int, np.integer)):
        return [(int(i), geoms_list[int(i)]) for i in cand]
    # Shapely 1.x: list of geometries
    return [(j, g) for j, g in enumerate(geoms_list) if g in cand]

# ---------- normalize ----------

def _detect_lon_lat(df) -> tuple[str, str]:
    cols = {c.strip().lower(): c for c in df.columns}
    lon_key = next((k for k in ("longitude","lon","x") if k in cols), None)
    lat_key = next((k for k in ("latitude","lat","y") if k in cols), None)
    if lon_key and lat_key:
        return cols[lon_key], cols[lat_key]
    raise ValueError(
        f"Could not find longitude/latitude columns. "
        f"Available: {list(df.columns)}. Expected one of "
        f"[longitude|lon|x] and [latitude|lat|y]."
    )

def normalize_runups(csv_path: str, id_col="tsunamiEventId", height_col="runupHt",
                     lon_col=None, lat_col=None) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)
    # prefer explicit longitude/latitude if present
    if lon_col is None or lat_col is None:
        lon_col, lat_col = _detect_lon_lat(df)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=WGS84,
    )
    # standardize output column names
    out = gdf.rename(columns={id_col: "tsunamiEventId", height_col: "runupHt"})
    return out[["tsunamiEventId", "runupHt", "geometry"]]


# ---------- snapping ----------

def snap_runups_to_coast(
    runups_gdf: gpd.GeoDataFrame,
    coast_lines_gs: gpd.GeoSeries,
    *,
    max_snap_km: float = 15.0,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    
    logger.info("snap_runups_to_coast max_snap_km=%.2f", max_snap_km)
    logger.debug("runups CRS=%s, coast CRS=%s", runups.crs, coast_lines.crs)

    runups = ensure_wgs84(runups_gdf).copy()
    coast_lines = ensure_wgs84(coast_lines_gs)

    valid = runups.geometry.apply(lambda p: isinstance(p, Point) and -90 <= p.y <= 90 and -180 <= p.x <= 180)
    invalid = runups.loc[~valid].copy()
    invalid["reason"] = "invalid_coord"
    runups = runups.loc[valid].copy()

    runups_m = runups.to_crs(WEBM)
    parts_m, parent_ids = _coast_parts_3857(coast_lines)
    if not parts_m:
        empty = gpd.GeoDataFrame(columns=["tsunamiEventId","runupHt","parent_id","s_m","geometry"], geometry="geometry", crs=WGS84)
        return empty, invalid

    tree = STRtree(parts_m)
    max_m = max_snap_km * 1000.0

    snapped_rows = []
    rejected_rows = []

    for idx, row in runups_m.iterrows():
        pt = row.geometry
        pairs = _tree_query_pairs(tree, parts_m, pt, max_m)
        # distance filter within max_m
        pairs = [(j, g) for j, g in pairs if g.distance(pt) <= max_m]
        if not pairs:
            rr = runups.loc[[idx]].copy()
            rr["reason"] = "beyond_max_snap"
            rejected_rows.append(rr)
            continue

        j, geom_m = min(pairs, key=lambda t: t[1].distance(pt))
        pid = int(parent_ids[j])

        s_m = float(geom_m.project(pt))
        snapped_rows.append({
            "tsunamiEventId": row.get("tsunamiEventId", None),
            "runupHt": row.get("runupHt", None),
            "parent_id": pid,
            "s_m": s_m,
            "geometry": pt,
        })

    snapped_m = gpd.GeoDataFrame(snapped_rows, geometry="geometry", crs=WEBM) if snapped_rows else gpd.GeoDataFrame(columns=["tsunamiEventId","runupHt","parent_id","s_m","geometry"], geometry="geometry", crs=WEBM)
    snapped = snapped_m.to_crs(WGS84)
    rejected = pd.concat(rejected_rows, ignore_index=True) if rejected_rows else pd.DataFrame(columns=list(runups.columns)+["reason"])
    rejected = gpd.GeoDataFrame(rejected, geometry="geometry", crs=WGS84)
    if not invalid.empty:
        rejected = pd.concat([rejected, invalid], ignore_index=True)
        rejected = gpd.GeoDataFrame(rejected, geometry="geometry", crs=WGS84)
    return snapped, rejected

# ---------- segments from runups ----------

def _span_km_from_height(H: float, L_min_km=4.0, beta_km_per_m=8.0, exp=1.0, L_max_km=60.0) -> float:
    if H is None or np.isnan(H):
        return L_min_km
    L = beta_km_per_m * (max(0.0, float(H)) ** exp)
    return float(np.clip(L, L_min_km, L_max_km))

def _merge_intervals(intervals: List[Tuple[float,float]], gap_tol_m: float) -> List[Tuple[float,float]]:
    if not intervals: return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for a,b in intervals[1:]:
        ca, cb = out[-1]
        if a - cb <= gap_tol_m:
            out[-1][1] = max(cb, b)
        else:
            out.append([a,b])
    return [(a,b) for a,b in out]

def build_runup_segments(
    snapped_gdf: gpd.GeoDataFrame,
    coast_lines_gs: gpd.GeoSeries,
    *,
    height_col: str = "runupHt",
    min_ht_m: float = 0.5,
    L_min_km: float = 4.0,
    beta_km_per_m: float = 8.0,
    exp: float = 1.0,
    L_max_km: float = 60.0,
    merge_tol_m: float = 1000.0,
) -> gpd.GeoDataFrame:
    """Create coastal segments around runups with H>=min_ht_m, merged by tolerance."""
    snapped = ensure_wgs84(snapped_gdf)
    coast_lines = ensure_wgs84(coast_lines_gs)
    parts_m, parent_ids = _coast_parts_3857(coast_lines)

    grouped: Dict[int, List[Tuple[float,float]]] = {}
    for _, r in snapped.iterrows():
        H = r.get(height_col, None)
        if H is None or float(H) < min_ht_m:
            continue
        pid = int(r["parent_id"])
        s_m = float(r["s_m"])
        L_km = _span_km_from_height(H, L_min_km, beta_km_per_m, exp, L_max_km)
        half = 0.5 * L_km * 1000.0
        grouped.setdefault(pid, []).append((s_m - half, s_m + half))

    seg_geoms = []
    seg_pids = []

    for pid, ivals in grouped.items():
        ivals = _merge_intervals(ivals, merge_tol_m)
        # find first occurrence of pid in parent_ids
        if pid not in parent_ids:
            continue
        j = parent_ids.index(pid)
        line_m = parts_m[j]
        for s0, s1 in ivals:
            s0c, s1c = max(0.0, s0), min(line_m.length, s1)
            if s1c <= s0c: 
                continue
            seg = _substring(line_m, s0c, s1c)
            if seg and not seg.is_empty:
                seg_geoms.append(seg)
                seg_pids.append(pid)

    segs_m = gpd.GeoDataFrame({"parent_id": seg_pids, "geometry": seg_geoms}, geometry="geometry", crs=WEBM)
    segs = to_wgs(segs_m)
    return segs

# ---------- infill ----------

def infill_runup_segments(
    snapped_gdf: gpd.GeoDataFrame,
    base_segs_gdf: gpd.GeoDataFrame,
    coast_lines_gs: gpd.GeoSeries,
    *,
    ht_lo_m: float = 0.2,
    ht_hi_m: float = 0.5,
    eps_km: float = 15.0,
    min_samples: int = 2,
    majority_threshold: float = 0.5,
    same_parent_only: bool = True,
    max_bridge_gap_km: float = 20.0,
    min_combined_coverage: float = 0.6,
) -> gpd.GeoDataFrame:
    """Fill gaps using lower-height evidence. Returns extra segments to union with base."""
    snapped = ensure_wgs84(snapped_gdf)
    base = ensure_wgs84(base_segs_gdf)
    coast_lines = ensure_wgs84(coast_lines_gs)

    eps_m = eps_km * 1000.0
    bridge_m = max_bridge_gap_km * 1000.0

    parts_m, parent_ids = _coast_parts_3857(coast_lines)

    # Base intervals per parent for coverage tests
    base_intervals: Dict[int, List[Tuple[float,float]]] = {pid:[] for pid in set(parent_ids)}
    if not base.empty:
        base_m = to_metric(base)
        for _, row in base_m.iterrows():
            # choose nearest parent line
            dists = [row.geometry.distance(ln) for ln in parts_m]
            j = int(np.argmin(dists))
            pid = parent_ids[j]
            line = parts_m[j]
            coords = list(row.geometry.coords)
            svals = [line.project(Point(*c)) for c in coords]
            s0, s1 = min(svals), max(svals)
            base_intervals.setdefault(pid, []).append((s0,s1))
        for pid in list(base_intervals.keys()):
            base_intervals[pid] = _merge_intervals(base_intervals[pid], gap_tol_m=1.0)

    extra_intervals: Dict[int, List[Tuple[float,float]]] = {}

    for pid in set(snapped["parent_id"].astype(int).tolist()):
        sub = snapped[(snapped["parent_id"].astype(int) == pid) & (snapped["runupHt"] >= ht_lo_m) & (snapped["runupHt"] < ht_hi_m)]
        if sub.empty: continue
        s_vals = np.array(sub["s_m"].astype(float).tolist())
        s_vals.sort()
        # form clusters by consecutive gaps ≤ eps
        clusters = []
        start = 0
        for i in range(1, len(s_vals)):
            if (s_vals[i] - s_vals[i-1]) > eps_m:
                clusters.append(s_vals[start:i])
                start = i
        clusters.append(s_vals[start:])

        for cl in clusters:
            if len(cl) < int(min_samples): continue
            s0, s1 = float(cl[0]), float(cl[-1])

            # coverage by base
            cover = 0.0
            if base_intervals.get(pid):
                covered = []
                for a,b in base_intervals[pid]:
                    lo, hi = max(a, s0), min(b, s1)
                    if hi > lo: covered.append((lo,hi))
                if covered:
                    merged = _merge_intervals(covered, 1.0)
                    cover_len = sum(b-a for a,b in merged)
                    cover = cover_len / max(1.0, (s1 - s0))

            accept = cover >= float(majority_threshold)

            if not accept and same_parent_only and base_intervals.get(pid):
                left = [b for b in base_intervals[pid] if b[1] <= s0]
                right = [b for b in base_intervals[pid] if b[0] >= s1]
                if left and right:
                    L = max(b[0] for b in left), max(b[1] for b in left)
                    R = min(b[0] for b in right), min(b[1] for b in right)
                    gap = max(0.0, R[0] - L[1])
                    if gap <= bridge_m and (cover >= float(min_combined_coverage)):
                        accept = True

            if accept:
                extra_intervals.setdefault(pid, []).append((s0, s1))

    geoms = []
    pids = []
    for pid, ivals in extra_intervals.items():
        ivals = _merge_intervals(ivals, gap_tol_m=1.0)
        if pid in parent_ids:
            j = parent_ids.index(pid)
            line = parts_m[j]
            for a,b in ivals:
                seg = _substring(line, max(0.0,a), min(line.length,b))
                if seg and not seg.is_empty:
                    geoms.append(seg)
                    pids.append(pid)

    gdf = gpd.GeoDataFrame({"parent_id": pids, "geometry": geoms}, geometry="geometry", crs=WEBM).to_crs(WGS84)
    return gdf

# ---------- inland polygons from runup segments ----------

def runup_segments_to_inland_poly(
    segs_gdf: gpd.GeoDataFrame,
    snapped_gdf: gpd.GeoDataFrame,
    landmask_gdf: Optional[gpd.GeoDataFrame] = None,
    *,
    height_col: str = "runupHt",
    D_min_km: float = 0.1,
    alpha_km_per_m: float = 1.0,
    gamma: float = 1.0,
    D_max_km: float = 12.0,
    sample_step_m: float = 250.0,
) -> gpd.GeoDataFrame:
    """
    Approximate inland impact polygon by sampling along segments and unioning local buffers
    with distance scaled from nearest runup height on the same parent line.
    """
    segs = ensure_wgs84(segs_gdf)
    snapped = ensure_wgs84(snapped_gdf)
    if segs.empty:
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)

    segs_m = to_metric(segs)

    # Build per-parent snapped points in metric CRS
    pts_by_parent: Dict[int, np.ndarray] = {}
    H_by_parent: Dict[int, np.ndarray] = {}
    for pid, group in snapped.groupby("parent_id"):
        arr = np.array([[float(r["s_m"]), float(r.get(height_col, 0.0))] for _, r in group.iterrows()], dtype=float)
        if arr.size == 0:
            continue
        arr = arr[arr[:,0].argsort()]
        pts_by_parent[int(pid)] = arr[:,0]
        H_by_parent[int(pid)] = arr[:,1]

    discs = []
    for _, row in segs_m.iterrows():
        pid = int(row.get("parent_id", -1))
        line = row.geometry
        L = line.length
        if L <= 0:
            continue
        n = max(2, int(math.ceil(L / float(sample_step_m))))
        stations = np.linspace(0.0, L, n)
        for s in stations:
            p = line.interpolate(s)
            # nearest snapped height on same parent
            H = 0.0
            if pid in pts_by_parent:
                S = pts_by_parent[pid]
                hvals = H_by_parent[pid]
                j = int(np.searchsorted(S, s))
                candidates = []
                if j < len(S): candidates.append((abs(S[j]-s), hvals[j]))
                if j > 0: candidates.append((abs(S[j-1]-s), hvals[j-1]))
                if candidates:
                    H = max(0.0, float(min(candidates, key=lambda t: t[0])[1]))
            D_km = max(D_min_km, min(D_max_km, alpha_km_per_m * (H ** gamma)))
            discs.append(p.buffer(D_km * 1000.0))

    poly = unary_union(discs) if discs else None
    gdf = gpd.GeoDataFrame(geometry=[poly] if poly else [], crs=WEBM).to_crs(WGS84)
    if landmask_gdf is not None and not gdf.empty:
        landmask = ensure_wgs84(landmask_gdf)
        gdf = gpd.overlay(gdf, landmask, how="intersection", keep_geom_type=True)
    return gdf

# ---------- general land buffer helper used by conservative ----------

def buffer_on_land(
    lines_ll: gpd.GeoDataFrame | gpd.GeoSeries,
    landmask_ll: Optional[gpd.GeoDataFrame] = None,
    *,
    width_m: float = 1000.0,
    coast_ll: Optional[gpd.GeoSeries] = None,
    land_from_coast_buffer_km: Optional[float] = None,
) -> gpd.GeoDataFrame:
    lines = ensure_wgs84(gpd.GeoDataFrame(geometry=gpd.GeoSeries(lines_ll), crs=WGS84))
    land = None
    if landmask_ll is not None and not gpd.GeoDataFrame(landmask_ll).empty:
        land = ensure_wgs84(gpd.GeoDataFrame(landmask_ll))
    elif coast_ll is not None and land_from_coast_buffer_km is not None:
        coast = ensure_wgs84(coast_ll)
        land = gpd.GeoDataFrame(geometry=coast.buffer(land_from_coast_buffer_km/111.32), crs=WGS84)
    if land is None:
        return lines.to_crs(WEBM).buffer(width_m).to_crs(WGS84)

    polys = lines.to_crs(WEBM).buffer(width_m).to_crs(WGS84)
    gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys), crs=WGS84)
    return gpd.overlay(gdf, land, how="intersection", keep_geom_type=True)
