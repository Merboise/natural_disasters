# rays.py
# Full implementation: origin, ray hits, segments from hits, inland by ray density.
from __future__ import annotations
import math, time
import logging
from typing import Tuple, List, Optional, Dict

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
from shapely import STRtree
from pyproj import CRS, Transformer
from contextlib import contextmanager

from .tsunami_helpers import (
    ensure_wgs84 as _h_ensure_wgs84,
)
from .tsunami_mem_cache import TsunamiMemCache
_MC = TsunamiMemCache()
try:
    # Optional cache-backed coast access (opt-in)
    from .tsunami_disk_cache import get_parts_wgs_exploded as _cache_get_parts_wgs_exploded
except Exception:
    _cache_get_parts_wgs_exploded = None  # type: ignore

logger = logging.getLogger("tsunami.rays")

WGS84 = 4326
WEBM = 3857

@contextmanager
def _timer(tag: str):
    # keep local behavior (no dependency); mirrors helpers._timer signature
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.debug("[%s] %.1f ms", tag, (time.perf_counter()-t0)*1000)

# ---------- helpers ----------

def ensure_wgs84(g: gpd.GeoSeries | gpd.GeoDataFrame) -> gpd.GeoSeries | gpd.GeoDataFrame:
    # delegate to shared helper to keep consistent behavior
    return _h_ensure_wgs84(g)

def _aeqd_for(pt):
    lat0, lon0 = float(pt.y), float(pt.x)
    try:
        return CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +ellps=WGS84 +units=m +type=crs")
    except Exception:
        return CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +R=6371000 +units=m +type=crs")

def _prep_aeqd(origin_pt):
    aeqd = _aeqd_for(origin_pt)
    wgs84 = CRS.from_epsg(4326)
    fwd = Transformer.from_crs(wgs84, aeqd, always_xy=True).transform
    inv = Transformer.from_crs(aeqd, wgs84, always_xy=True).transform
    return aeqd, fwd, inv

def _ray(m, ang_rad):
    R = m
    return LineString([(0.0, 0.0), (R * math.cos(ang_rad), R * math.sin(ang_rad))])

def _substring(line_m, s0, s1):
    if s1 <= s0: return None
    try:
        from shapely.ops import substring
        return substring(line_m, s0, s1, normalized=False)
    except Exception:
        n = max(2, int((s1 - s0) / 50.0))
        pts = [line_m.interpolate(t) for t in np.linspace(s0, s1, n)]
        return LineString(pts)

# ---------- origin ----------

def pick_origin(runup_evidence: Optional[gpd.GeoDataFrame], coast_lines: gpd.GeoSeries) -> Point:
    coast = ensure_wgs84(coast_lines)
    if runup_evidence is not None and not runup_evidence.empty:
        return runup_evidence.unary_union.centroid
    return gpd.GeoSeries(coast).unary_union.centroid

# ---------- ray casting ----------
def cast_rays_hits(
    coast_lines_gs: gpd.GeoSeries,
    origin_pt_wgs: Point,
    *,
    step_deg: float = 1.0,
    max_range_km: float = 3000.0,
    simplify_m: float = 10.0,
    hit_buffer_m: float = 750.0,
    engine: str = "aeqd",
) -> gpd.GeoDataFrame:
    assert engine in ("aeqd", "raster")
    logger.info(
        "cast_rays_hits step_deg=%.2f max_range_km=%.1f simplify_m=%.1f hit_buffer_m=%.1f",
        step_deg, max_range_km, simplify_m, hit_buffer_m
    )
    coast = ensure_wgs84(coast_lines_gs)
    O = origin_pt_wgs
    # In-memory cache for exploded parts is always used

    with _timer("prep_aeqd"):
        aeqd = _aeqd_for(O)
        to_aeqd = Transformer.from_crs(WGS84, aeqd, always_xy=True).transform
        to_wgs = Transformer.from_crs(aeqd, WGS84, always_xy=True).transform

        lines_m: list[LineString] = []
        id_map: list[int] = []
        parts_wgs, _parent_ids, _key = _MC.get_parts_wgs_exploded(gpd.GeoDataFrame(geometry=gpd.GeoSeries(coast), crs=WGS84))
        if len(parts_wgs) > 0:
            parts_aeqd = parts_wgs.to_crs(aeqd)
            for i, g in enumerate(parts_aeqd.values):
                if g is None or g.is_empty:
                    continue
                gm = g.simplify(simplify_m, preserve_topology=False)
                if gm.geom_type == "LineString":
                    lines_m.append(gm); id_map.append(i)
                elif gm.geom_type == "MultiLineString":
                    for sub in gm.geoms:
                        lines_m.append(sub); id_map.append(i)

        if not lines_m:
            return gpd.GeoDataFrame(
                columns=["bearing_deg", "range_m", "parent_id", "s_m", "geometry"],
                geometry="geometry",
                crs=WGS84,
            )
        logger.info("aeqd lines=%d", len(lines_m))

    # STRtree and global index maps
    tree = STRtree(lines_m)
    idx_by_id = {id(g): i for i, g in enumerate(lines_m)}
    try:
        idx_by_wkb = {g.wkb: i for i, g in enumerate(lines_m)}
    except Exception:
        idx_by_wkb = {}

    O_m = Point(*to_aeqd(O.x, O.y))
    max_r_m = float(max_range_km) * 1000.0

    hits = []
    with _timer("raycast"):
        processed = 0
        last_log = time.time()
        for b in np.arange(0.0, 360.0, step_deg):
            # Exponential radial search. Stop at first hit.
            rad = math.radians(b)
            d = min(50_000.0, max_r_m)  # start at 50 km
            best = None  # (rng_m, j_global, best_pt)
            while d <= max_r_m:
                end = Point(O_m.x + d * math.cos(rad), O_m.y + d * math.sin(rad))
                seg = LineString([O_m, end])
                env = seg.buffer(hit_buffer_m).envelope
                cand = tree.query(env)

                # Normalize to (geom, j_global)
                if len(cand) and isinstance(cand[0], (int, np.integer)):
                    cand_pairs = [(lines_m[int(i)], int(i)) for i in cand]
                else:
                    tmp = []
                    for g in cand:
                        j = idx_by_id.get(id(g))
                        if j is None:
                            j = idx_by_wkb.get(getattr(g, "wkb", b""), None)
                        if j is None:
                            continue
                        tmp.append((g, j))
                    cand_pairs = tmp

                if cand_pairs:
                    seg_buf = seg.buffer(hit_buffer_m)
                    for geom, j in cand_pairs:
                        inter = geom.intersection(seg)
                        if inter.is_empty:
                            inter = geom.buffer(hit_buffer_m).intersection(seg_buf)
                            if inter.is_empty:
                                continue
                        # Gather candidate points
                        pts = []
                        if inter.geom_type == "Point":
                            pts = [inter]
                        elif inter.geom_type == "LineString":
                            pts = [Point(inter.coords[0]), Point(inter.coords[-1])]
                        elif inter.geom_type in ("MultiPoint", "GeometryCollection", "MultiLineString"):
                            pts = [g for g in getattr(inter, "geoms", []) if g.geom_type == "Point"]
                        for p in pts:
                            s = seg.project(p)
                            if s < 0.0:
                                continue
                            rng_m = float(p.distance(O_m))
                            if best is None or rng_m < best[0]:
                                best = (rng_m, j, p)
                    if best is not None:
                        break  # first hit found within current radius

                d *= 2.0  # expand search radius

            if best is not None:
                rng_m, j_global, p = best
                parent = id_map[j_global]
                line = lines_m[j_global]
                s_m = float(line.project(p))
                x, y = to_wgs(p.x, p.y)
                hits.append(
                    {
                        "bearing_deg": float(b),
                        "range_m": rng_m,
                        "parent_id": int(parent),
                        "s_m": s_m,
                        "geometry": Point(x, y),
                    }
                )

            processed += 1
            if time.time() - last_log >= 30.0:
                logger.debug("cast_rays_hits progress: %d rays processed", processed)
                last_log = time.time()

        logger.info("hits=%d", len(hits))

    gdf = gpd.GeoDataFrame(hits, geometry="geometry", crs=WGS84)
    return gdf


# ---------- segments from hits ----------

def hits_to_segments(
    hits_gdf: gpd.GeoDataFrame,
    coast_lines_gs: gpd.GeoSeries,
    *,
    expand_km: float = 125.0,
    gap_base_km: float = 20.0,
    gap_per_1000km: float = 40.0,
) -> gpd.GeoDataFrame:
    """Expand each hit to an along-coast interval and merge with variable gap rule."""
    if hits_gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)
    hits = hits_gdf.copy()
    coast = ensure_wgs84(coast_lines_gs)

    # Use in-memory cache for exploded WGS84 parts
    parts_wgs, _pids, _key = _MC.get_parts_wgs_exploded(
        gpd.GeoDataFrame(geometry=gpd.GeoSeries(coast), crs=WGS84)
    )
    parts_metric = parts_wgs.to_crs(WEBM)
    parent_of_part = list(range(len(parts_metric)))

    expand_metric = expand_km * 1000.0
    ivals_by_parent: Dict[int, List[tuple]] = {}
    for _, r in hits.iterrows():
        pid = int(r["parent_id"])
        s = float(r["s_m"])
        rng_km = (float(r.get("range_m", 0.0)) / 1000.0)
        ivals_by_parent.setdefault(pid, []).append((s - 0.5*expand_metric, s + 0.5*expand_metric, rng_km))

    def merge_variable(ivals: List[tuple]) -> List[Tuple[float,float]]:
        if not ivals: return []
        ivals = sorted(ivals, key=lambda t: (t[0], t[1]))
        out = [list(ivals[0][:2]) + [ivals[0][2]]]
        for a,b,range_km in ivals[1:]:
            ca, cb, rprev = out[-1]
            gap = a - cb
            allow = (gap_base_km + gap_per_1000km * ((range_km + rprev) / 2.0 / 1000.0)) * 1000.0
            if gap <= allow:
                out[-1][1] = max(cb, b)
                out[-1][2] = (rprev + range_km) / 2.0
            else:
                out.append([a,b,range_km])
        return [(a,b) for a,b,_ in out]

    seg_geoms = []
    seg_pids = []
    for pid, ivals in ivals_by_parent.items():
        merged = merge_variable(ivals)
        try:
            j = parent_of_part.index(pid)
            line = parts_metric.iloc[j]
        except ValueError:
            j = int(np.argmin([parts_metric.iloc[k].distance(parts_metric.iloc[0].interpolate(0)) for k in range(len(parts_metric))]))
            line = parts_metric.iloc[j]
        for a,b in merged:
            a = max(0.0, a); b = max(a, b)
            seg = _substring(line, a, b)
            if seg and not seg.is_empty:
                seg_geoms.append(seg)
                seg_pids.append(pid)

    logger.info("hits_to_segments expand_km=%.1f gap_base_km=%.1f gap_per_1000km=%.1f", expand_km, gap_base_km, gap_per_1000km)
    segs = gpd.GeoDataFrame({"parent_id": seg_pids, "geometry": seg_geoms}, geometry="geometry", crs=WEBM).to_crs(WGS84)
    return segs

# ---------- inland by ray density ----------

def ray_density_to_inland_poly(
    hits_gdf: gpd.GeoDataFrame,
    coast_lines_gs: gpd.GeoSeries,
    landmask_gdf: Optional[gpd.GeoDataFrame] = None,
    *,
    window_km: float = 1000.0,
    k_coeff: float = 0.02,
    D_min_km: float = 0.1,
    realistic_cap_km: float = 1.0,
    hard_cap_km: float = 12.0,
    sample_step_m: float = 250.0,
    origin_pt_wgs: Optional[Point] = None,
    r0_km: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """Convert ray hit density into inland polygon by sliding window density mapping."""
    if hits_gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)

    coast = ensure_wgs84(coast_lines_gs)
    parts_metric = gpd.GeoSeries(coast, crs=WGS84).to_crs(WEBM)
    parts_list = list(parts_metric.geometry.values)
    window_m = window_km * 1000.0

    discs = []
    fallback_used = 0
    # Precompute per-hit weights by distance to origin if provided
    weights = None
    if origin_pt_wgs is not None and r0_km is not None and r0_km > 0:
        ox, oy = float(origin_pt_wgs.x), float(origin_pt_wgs.y)
        def _w(pt: Point):
            from .tsunami_helpers import haversine_km
            d = haversine_km(ox, oy, float(pt.x), float(pt.y))
            w = math.exp(-float(d)/float(r0_km))
            return max(0.0, min(1.0, w))
        weights = hits_gdf.geometry.apply(_w).astype(float).values

    for pid, group in hits_gdf.groupby("parent_id"):
        if not parts_list:
            continue
        # Map parent_id to the corresponding coastline part in metric CRS
        try:
            j = int(pid)
            if j < 0 or j >= len(parts_metric):
                raise IndexError
            line = parts_metric.iloc[j]
        except Exception:
            # Fallback: pick the longest line as a crude proxy to avoid empty output
            try:
                line = max(parts_list, key=lambda L: float(getattr(L, 'length', 0.0)))
            except Exception:
                line = parts_list[0]
            fallback_used += 1
        S = np.array(group["s_m"].astype(float).tolist())
        W = None
        if weights is not None:
            idxs = group.index.values
            # Map to weights by original index order
            W = np.array([weights[hits_gdf.index.get_loc(i)] for i in idxs], dtype=float)
        S.sort()
        if W is not None:
            # keep W aligned with S by sorting keys
            order = np.argsort(S)
            W = W[order]
            S = S[order]
        L = float(line.length)
        n = max(2, int(L / sample_step_m))
        stations = np.linspace(0.0, L, n)
        for s in stations:
            left = s - 0.5*window_m
            right = s + 0.5*window_m
            if W is None:
                cnt = int(((S >= left) & (S <= right)).sum())
                rho = cnt / max(1.0, window_km)
            else:
                mask = (S >= left) & (S <= right)
                sum_w = float(W[mask].sum())
                rho = sum_w / max(1.0, window_km)
            D_km = max(D_min_km, min(hard_cap_km, min(realistic_cap_km, k_coeff * rho)))
            p = line.interpolate(s)
            discs.append(p.buffer(D_km * 1000.0))

    poly = unary_union(discs) if discs else None
    gdf = gpd.GeoDataFrame(geometry=[poly] if poly else [], crs=WEBM).to_crs(WGS84)
    if landmask_gdf is not None and not gdf.empty:
        land = ensure_wgs84(landmask_gdf)
        gdf = gpd.overlay(gdf, land, how="intersection", keep_geom_type=True)

    logger.info("ray_density_to_inland_poly window_km=%.1f k_coeff=%.3f caps=(%.1f realistic, %.1f hard) fallback_used=%d", window_km, k_coeff, realistic_cap_km, hard_cap_km, fallback_used)
    return gdf
