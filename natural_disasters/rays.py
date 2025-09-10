# rays.py
# Full implementation: origin, ray hits, segments from hits, inland by ray density.
from __future__ import annotations
import math
from typing import Tuple, List, Optional, Dict

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
from shapely import STRtree
from pyproj import CRS, Transformer

WGS84 = 4326
WEBM = 3857

# ---------- helpers ----------

def ensure_wgs84(g: gpd.GeoSeries | gpd.GeoDataFrame) -> gpd.GeoSeries | gpd.GeoDataFrame:
    if g is None:
        return g
    if g.crs is None:
        g = g.set_crs(WGS84)
    elif int(g.crs.to_epsg() or 0) != WGS84:
        g = g.to_crs(WGS84)
    return g

def _aeqd_for(pt: Point) -> CRS:
    return CRS.from_proj4(f"+proj=aeqd +lat_0={pt.y} +lon_0={pt.x} +x_0=0 +y_0=0 +units=m +no_defs")

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
    """Cast rays radially and find first-hit intersections; return hits with parent and station."""
    assert engine in ("aeqd", "raster")
    coast = ensure_wgs84(coast_lines_gs)
    O = origin_pt_wgs
    aeqd = _aeqd_for(O)
    to_aeqd = Transformer.from_crs(WGS84, aeqd, always_xy=True).transform
    to_wgs = Transformer.from_crs(aeqd, WGS84, always_xy=True).transform

    # Prepare coast in AEQD
    coast_m = coast.to_crs(aeqd)
    lines_m = []
    id_map = []
    for i, g in enumerate(coast_m.values):
        if g is None or g.is_empty: continue
        gm = g.simplify(simplify_m)
        if gm.geom_type == "LineString":
            lines_m.append(gm); id_map.append(i)
        elif gm.geom_type == "MultiLineString":
            for sub in gm.geoms:
                lines_m.append(sub); id_map.append(i)

    if not lines_m:
        return gpd.GeoDataFrame(columns=["bearing_deg","range_m","parent_id","s_m","geometry"], geometry="geometry", crs=WGS84)

    tree = STRtree(lines_m)
    O_m = Point(*to_aeqd(O.x, O.y))
    max_r_m = max_range_km * 1000.0

    hits = []
    for b in np.arange(0.0, 360.0, step_deg):
        rad = math.radians(b)
        end = Point(O_m.x + max_r_m * math.cos(rad), O_m.y + max_r_m * math.sin(rad))
        ray = LineString([O_m, end])
        cand = tree.query(ray)
        best_pt = None
        best_d = 1e30
        best_idx = None
        for geom in cand:
            inter = geom.intersection(ray)
            if inter.is_empty:
                inter = geom.buffer(hit_buffer_m).intersection(ray)
                if inter.is_empty:
                    continue
            pts = []
            if inter.geom_type == "Point":
                pts = [inter]
            elif inter.geom_type in ("MultiPoint","GeometryCollection","LineString","MultiLineString"):
                pts = [g for g in (list(inter.geoms) if hasattr(inter, "geoms") else [inter]) if g.geom_type == "Point"]
            for p in pts:
                d = p.distance(O_m)
                if d < best_d:
                    best_d = d
                    best_pt = p
                    try:
                        j = lines_m.index(geom)
                    except ValueError:
                        j = None
                    best_idx = j

        if best_pt is None or best_idx is None:
            continue
        parent = id_map[best_idx]
        line = lines_m[best_idx]
        s_m = float(line.project(best_pt))
        x,y = to_wgs(best_pt.x, best_pt.y)
        hits.append({"bearing_deg": float(b), "range_m": float(best_d), "parent_id": int(parent), "s_m": s_m, "geometry": Point(x,y)})

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

    # Explode to parts in metric CRS
    parts_wgs = gpd.GeoSeries(coast, crs=WGS84).explode(index_parts=False, ignore_index=True)
    parts_m = parts_wgs.to_crs(WEBM)

    # Build intervals by parent
    expand_m = expand_km * 1000.0
    ivals_by_parent: Dict[int, List[tuple]] = {}
    for _, r in hits.iterrows():
        pid = int(r["parent_id"])
        s = float(r["s_m"])
        rng_km = (float(r.get("range_m", 0.0)) / 1000.0)
        ivals_by_parent.setdefault(pid, []).append((s - 0.5*expand_m, s + 0.5*expand_m, rng_km))

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
        # map pid to nearest part (simple assumption)
        if len(parts_m) == 0: continue
        line = parts_m.iloc[0]
        for a,b in merged:
            a = max(0.0, a); b = max(a, b)
            seg = _substring(line, a, b)
            if seg and not seg.is_empty:
                seg_geoms.append(seg)
                seg_pids.append(pid)

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
) -> gpd.GeoDataFrame:
    """Convert ray hit density into inland polygon by sliding window density mapping."""
    if hits_gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)

    coast = ensure_wgs84(coast_lines_gs)
    parts_m = gpd.GeoSeries(coast, crs=WGS84).to_crs(WEBM)
    parts_list = list(parts_m.geometry.values)
    window_m = window_km * 1000.0

    discs = []
    for pid, group in hits_gdf.groupby("parent_id"):
        if not parts_list:
            continue
        line = parts_list[0]
        S = np.array(group["s_m"].astype(float).tolist())
        S.sort()
        L = float(line.length)
        n = max(2, int(L / sample_step_m))
        stations = np.linspace(0.0, L, n)
        for s in stations:
            left = s - 0.5*window_m
            right = s + 0.5*window_m
            cnt = int(((S >= left) & (S <= right)).sum())
            rho = cnt / max(1.0, window_km)  # hits per km
            D_km = max(D_min_km, min(hard_cap_km, min(realistic_cap_km, k_coeff * rho)))
            p = line.interpolate(s)
            discs.append(p.buffer(D_km * 1000.0))

    poly = unary_union(discs) if discs else None
    gdf = gpd.GeoDataFrame(geometry=[poly] if poly else [], crs=WEBM).to_crs(WGS84)
    if landmask_gdf is not None and not gdf.empty:
        land = ensure_wgs84(landmask_gdf)
        gdf = gpd.overlay(gdf, land, how="intersection", keep_geom_type=True)
    return gdf
