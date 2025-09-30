from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import substring as shp_substring
from shapely.strtree import STRtree
from shapely.geometry import LineString
from pyproj import Transformer, CRS
import math
import numpy as np
import pandas as pd

WGS84 = 4326
WEBM = 3857

logger = logging.getLogger("tsunami.helpers")

# ------------------------- CRS helpers -------------------------

def ensure_wgs84(g: gpd.GeoSeries | gpd.GeoDataFrame) -> gpd.GeoSeries | gpd.GeoDataFrame:
    if g is None:
        return g
    if getattr(g, "crs", None) is None:
        return g.set_crs(WGS84)
    epsg = int(g.crs.to_epsg() or 0)
    return g if epsg == WGS84 else g.to_crs(WGS84)

def to_metric(g):
    return g.to_crs(WEBM)

def to_wgs(g):
    return g.to_crs(WGS84)

# ------------------------- CSV detection helpers -------------------------

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

def _detect_event_id_col(df, preferred: str | None):
    if preferred and preferred in df.columns:
        return preferred
    lc = {c.lower(): c for c in df.columns}
    for key in ("tsunamieventid","eventid","event_id","id"):
        if key in lc:
            return lc[key]
    for c in df.columns:
        if "eventid" in c.strip().lower():
            return c
    raise ValueError(f"Could not detect event id column. Have {list(df.columns)}")

def _detect_height_col(df, preferred: str | None):
    if preferred and preferred in df.columns:
        return preferred
    lc = {c.lower(): c for c in df.columns}
    for key in ("runupht","runup_height","height","h"):
        if key in lc:
            return lc[key]
    for c in df.columns:
        if "runup" in c.strip().lower() and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError(f"Could not detect height column. Have {list(df.columns)}")

# ------------------------- Geometry helpers -------------------------

def _merge_intervals(intervals: list[tuple[float,float]], *, gap_merge_m: float = 0.0) -> list[tuple[float,float]]:
    if not intervals:
        return []
    iv = sorted((float(a), float(b)) for a,b in intervals)
    out = [list(iv[0])]
    for a,b in iv[1:]:
        A,B = out[-1]
        if a <= B + float(gap_merge_m):
            out[-1][1] = max(B, b)
        else:
            out.append([a,b])
    return [(a,b) for a,b in out]

def _cut_parent_substrings(parent_line_m: LineString, ivals_m: list[tuple[float,float]]) -> list[LineString]:
    out = []
    for a,b in ivals_m:
        a = max(0.0, min(parent_line_m.length, float(a)))
        b = max(0.0, min(parent_line_m.length, float(b)))
        if b <= a:
            continue
        try:
            seg = shp_substring(parent_line_m, a, b)
            if seg and not seg.is_empty and seg.geom_type == "LineString":
                out.append(seg)
        except Exception:
            pass
    return out

def _tree_query_pairs(tree: STRtree, geoms_list, target_geom, max_snap_meters):
    try:
        cand = tree.query(target_geom.buffer(max_snap_meters))
    except Exception:
        cand = tree.query(target_geom)
    if len(cand) and isinstance(cand[0], (int, np.integer)):
        return [(int(i), geoms_list[int(i)]) for i in cand]
    return [(j, g) for j, g in enumerate(geoms_list) if g in cand]

# ------------------------- Logging helpers -------------------------

@contextmanager
def _timer(tag: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.debug("[%s] %.1f ms", tag, (time.perf_counter()-t0)*1000)

def _log_gdf(tag, gdf):
    try:
        n = 0 if gdf is None else len(gdf)
        crs = getattr(gdf, "crs", None)
        gtypes = [] if gdf is None or gdf.empty else list(getattr(gdf, "geom_type", pd.Series(dtype=str)).value_counts().to_dict().items())
        logger.info("[gdf:%s] n=%s crs=%s types=%s", tag, n, crs, gtypes)
    except Exception as e:
        logger.debug("[gdf:%s] summary failed: %r", tag, e)

# ------------------------- Batched ops + vectorized reprojection -------------------------

def union_all_geoms(geoms: list):
    try:
        from shapely import union_all as _union_all
        return _union_all([g for g in geoms if g is not None])
    except Exception:
        from shapely.ops import unary_union as _uun
        return _uun([g for g in geoms if g is not None])

def project_lines_vectorized(lines: list[LineString], from_crs: int | str, to_crs: int | str) -> list[LineString]:
    if not lines:
        return []
    tf = Transformer.from_crs(CRS.from_user_input(from_crs), CRS.from_user_input(to_crs), always_xy=True)
    out: list[LineString] = []
    for L in lines:
        if L is None or L.is_empty:
            out.append(L)
            continue
        coords = list(L.coords)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        x2, y2 = tf.transform(xs, ys)
        out.append(LineString(zip(x2, y2)))
    return out

# ------------------------- Distance + decay fitting -------------------------

def haversine_km(lon1, lat1, lon2, lat2) -> float:
    R = 6371.0088
    phi1 = math.radians(float(lat1)); phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0-a)))
    return R * c

def derive_r_decay(
    snapped_gdf: gpd.GeoDataFrame,
    origin_pt,
    *,
    bin_km: float = 250.0,
    min_bin_n: int = 10,
    envelope_q: float = 0.90,
    r0_min_km: float = 500.0,
    r0_max_km: float = 5000.0,
    search_min_km: float = 2000.0,
    search_max_km: float = 10000.0,
    search_mult: float = 3.0,
    fallback_r0_km: float = 1500.0,
    fallback_search_km: float = 3000.0,
    h_min_m: float = 0.05,
) -> tuple[float, float, dict]:
    """Fit log(q90) = a - b*d on binned runups to derive R0 and search radius.
    Returns (R0_km, R_search_km, diagnostics).
    """
    diag = {"fallback": False}
    if snapped_gdf is None or snapped_gdf.empty:
        diag.update({"reason": "no_runups"})
        return float(fallback_r0_km), float(fallback_search_km), diag

    # Distances from origin to each runup
    ox, oy = float(origin_pt.x), float(origin_pt.y)
    dists = []
    for p in snapped_gdf.geometry.values:
        if p is None or p.is_empty:
            continue
        dists.append(haversine_km(ox, oy, float(p.x), float(p.y)))
    if not dists:
        diag.update({"reason": "no_valid_points"})
        return float(fallback_r0_km), float(fallback_search_km), diag

    H = snapped_gdf["runupHt"].astype(float).values
    D = np.asarray(dists, dtype=float)
    if len(D) != len(H):
        n = min(len(D), len(H))
        D = D[:n]; H = H[:n]

    # Bin by distance
    max_d = float(np.nanmax(D)) if len(D) else 0.0
    nbins = max(1, int(math.ceil((max_d + 1e-6) / float(bin_km))))
    bins = np.floor(D / float(bin_km)).astype(int)

    # Aggregate per bin with minimal merging heuristic
    records = {}
    for b in range(nbins):
        mask = (bins == b)
        if not mask.any():
            continue
        h = H[mask]
        d_center = (b + 0.5) * float(bin_km)
        records[b] = {"n": int(len(h)), "d": d_center, "q": float(np.quantile(h, envelope_q))}

    # Merge: for bins with n < min_bin_n, try combining with next bin
    merged = []
    used = set()
    for b, rec in records.items():
        if b in used:
            continue
        if rec["n"] >= int(min_bin_n):
            merged.append(rec)
            used.add(b)
        else:
            # try merge with next
            nb = b + 1
            if nb in records:
                nn = rec["n"] + records[nb]["n"]
                q = float(np.quantile(np.concatenate([
                    H[(bins==b)], H[(bins==nb)]
                ]), envelope_q))
                d_center = 0.5 * (rec["d"] + records[nb]["d"]) 
                merged.append({"n": int(nn), "d": d_center, "q": q})
                used.add(b); used.add(nb)
            else:
                # keep only if it meets min
                if rec["n"] >= int(min_bin_n):
                    merged.append(rec)
                    used.add(b)

    # Filter by height envelope
    kept = [m for m in merged if m["q"] > float(h_min_m)]
    diag.update({
        "total_runups": int(len(D)),
        "bins_total": int(len(records)),
        "bins_kept": int(len(kept)),
        "bins_dropped": int(len(records) - len(kept)),
    })

    if len(kept) < 2:
        diag.update({"fallback": True, "reason": "insufficient_bins"})
        return float(fallback_r0_km), float(fallback_search_km), diag

    d_arr = np.array([m["d"] for m in kept], dtype=float)
    q_arr = np.array([m["q"] for m in kept], dtype=float)
    y = np.log(q_arr + 1e-6)
    # Fit y = a - b*d => slope = -b
    try:
        coeffs = np.polyfit(d_arr, y, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        beta = -slope
        yhat = intercept + slope * d_arr
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - np.mean(y))**2))
        r2 = 0.0 if ss_tot == 0.0 else 1.0 - ss_res/ss_tot
    except Exception as e:
        diag.update({"fallback": True, "reason": f"ols_fail:{e}"})
        return float(fallback_r0_km), float(fallback_search_km), diag

    if not np.isfinite(beta) or beta <= 0:
        diag.update({"fallback": True, "reason": "beta_nonpositive"})
        return float(fallback_r0_km), float(fallback_search_km), diag

    R0_km = float(max(r0_min_km, min(r0_max_km, 1.0 / beta)))
    R_search_km = float(max(search_min_km, min(search_max_km, search_mult * R0_km)))

    def w_sample(d):
        return math.exp(-float(d) / R0_km)

    diag.update({
        "alpha": intercept,
        "beta": beta,
        "r2": r2,
        "R0_km": R0_km,
        "R_search_km": R_search_km,
        "w(0)": w_sample(0),
        "w(500)": w_sample(500),
        "w(1000)": w_sample(1000),
        "w(2000)": w_sample(2000),
        "w(5000)": w_sample(5000),
        "fallback": False,
    })
    return R0_km, R_search_km, diag
