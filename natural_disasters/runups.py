# runups.py
from __future__ import annotations
import math, numbers, logging, time, heapq, inspect
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict, Set
from collections import defaultdict, Counter
from contextlib import contextmanager, nullcontext
from .infill_config import InfillConfig

import numpy as np
import pandas as pd
import geopandas as gpd
import os
import threading
from collections import Counter
from shapely.geometry import Point, LineString, Polygon, box
from shapely.ops import unary_union, substring, linemerge
from shapely import STRtree

from .tsunami_helpers import (
    ensure_wgs84 as _h_ensure_wgs84,
    to_metric as _h_to_metric,
    to_wgs as _h_to_wgs,
    _detect_lon_lat as _h_detect_lon_lat,
    _detect_event_id_col as _h_detect_event_id_col,
    _detect_height_col as _h_detect_height_col,
    _merge_intervals as _h_merge_intervals,
    _cut_parent_substrings as _h_cut_parent_substrings,
    _tree_query_pairs as _h_tree_query_pairs,
    _timer as _h_timer,
    _log_gdf as _h_log_gdf,
)
from .tsunami_mem_cache import TsunamiMemCache, _MC

WGS84 = 4326
WEBM = 3857

logger = logging.getLogger("tsunami.runups")

def _trace_enabled():
    return logger.isEnabledFor(logging.DEBUG) or os.environ.get("ND_TRACE", "0") == "1"

@contextmanager
def _trace(tag: str, **fields):
    if _trace_enabled():
        logger.info(">> %s %s", tag, fields if fields else "")
        t0 = time.time()
        try:
            yield
        finally:
            logger.info("<< %s dt=%.2fs", tag, time.time() - t0)
    else:
        yield

def _timer(tag: str):
    # shim to keep local signature; delegate to helpers
    with _h_timer(tag):
        yield

def _maybe_timer(tag: str, enabled: bool):
    """
    Conditional timer helper so callers can do:  with _maybe_timer('tag', log_timing): ...
    """
    return _timer(tag) if enabled else nullcontext()

try:
    if " _maybe_timer" in globals():
        pass
except NameError:
    pass

if " _maybe_timer" in globals():
    _mt = globals()["_maybe_timer"]
    if inspect.isgeneratorfunction(_mt):
        globals()["_maybe_timer"] = contextmanager(_mt)
else:
    # fallback shim if _maybe_timer is absent
    def _maybe_timer(tag: str, enabled: bool):
        return nullcontext()

def _log_gdf(tag, gdf):
    return _h_log_gdf(tag, gdf)

@dataclass
class _Edge:
    u: int
    v: int
    length: float
    geom: LineString           # metric CRS (WEBM)
    parent_idx: int            # index in parts list
    parent_feature_id: int     # original parent_feature_id

@dataclass
class CoastGraph:
    nodes_xy: List[Tuple[float, float]]    # node_id -> (x,y)
    edges: List[_Edge]                     # edge_id -> edge
    node_edges: List[List[int]]            # node_id -> [edge_ids]
    edge_tree: STRtree                    # STRtree over edge geoms
    edge_idx_by_id: Dict[int, int]         # id(geom) -> edge_id
    node_comp_id: List[int]                # node_id -> component id

_COAST_GRAPH_MEMO: Dict[Tuple[int,int,int,int,float], CoastGraph] = {}
# ---------- helpers ----------

def ensure_wgs84(g):
    # shim to keep name local if referenced elsewhere
    return _h_ensure_wgs84(g)

def to_metric(g):
    return _h_to_metric(g)

def to_wgs(g):
    return _h_to_wgs(g)

try:
    from shapely.ops import substring as _substring
except Exception:
    def _substring(ls, s0, s1):
        # simple meters-based slice of a LineString
        from shapely.geometry import LineString, Point
        if s0 == s1: return Point(ls.interpolate(s0))
        if s0 > s1: s0, s1 = s1, s0
        coords = [ls.coords[0]]
        acc = 0.0
        out = []
        for (x0,y0),(x1,y1) in zip(ls.coords[:-1], ls.coords[1:]):
            seg = ((x0,y0),(x1,y1))
            dx, dy = (x1-x0), (y1-y0)
            L = (dx*dx+dy*dy)**0.5
            if L == 0: continue
            n0, n1 = acc, acc+L
            # segment overlaps [s0,s1]?
            a = max(s0, n0); b = min(s1, n1)
            if a < b:
                t0 = (a-n0)/L; t1 = (b-n0)/L
                xa, ya = x0+dx*t0, y0+dy*t0
                xb, yb = x0+dx*t1, y0+dy*t1
                if not out: out.append((xa,ya))
                out.append((xb,yb))
            acc = n1
        return LineString(out) if len(out) > 1 else Point(out[0])

def _coast_parts_3857(coast_lines: gpd.GeoSeries):
    """Wrapper: retrieve exploded 3857 parts and parent ids from in-memory cache."""
    gs = gpd.GeoDataFrame(geometry=gpd.GeoSeries(ensure_wgs84(coast_lines)), crs=WGS84)
    parts_m, parent_ids, tree, _key = _MC.get_parts_3857_and_tree(gs)
    return parts_m, parent_ids

def _substring(line_meters: LineString, s0: float, s1: float) -> Optional[LineString]:
    if s1 <= s0: return None
    try:
        from shapely.ops import substring
        return substring(line_meters, s0, s1, normalized=False)
    except Exception:
        n = max(2, int((s1 - s0) / 50.0))
        pts = [line_meters.interpolate(t) for t in np.linspace(s0, s1, n)]
        return LineString(pts)

def _project_to_line_s(line_meters: LineString, pt_m: Point) -> float:
    return float(line_meters.project(pt_m))

def _tree_query_pairs(tree, geoms_list, target_geom, max_snap_meters):
    return _h_tree_query_pairs(tree, geoms_list, target_geom, max_snap_meters)

def _snap_key(pt: Point, snap_m: float) -> Tuple[int,int]:
    return (int(round(pt.x / snap_m)), int(round(pt.y / snap_m)))

def _dsu_build(n: int):
    p = list(range(n))
    r = [0]*n
    def f(x):
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x
    def u(a,b):
        ra, rb = f(a), f(b)
        if ra == rb: return
        if r[ra] < r[rb]: p[ra] = rb
        elif r[ra] > r[rb]: p[rb] = ra
        else: p[rb] = ra; r[ra]+=1
    return f, u

# add at top of runups.py if not present
# from .tsunami_mem_cache import _MC  # if _MC already imported skip

#def build_coast_graph_from_coast(coast_lines_gs, *, snap_m: float = 250.0, log_timing: bool = False):
    import time, logging, geopandas as gpd
    from shapely.geometry import Point
    from shapely.strtree import STRtree
    logger = logging.getLogger("natural_disasters.coast_graph")

    t0 = time.time()
    coast_wgs = gpd.GeoSeries(ensure_wgs84(coast_lines_gs)) \
        if not isinstance(coast_lines_gs, gpd.GeoSeries) else ensure_wgs84(coast_lines_gs)

    parts_m, part_ids, _parts_tree, _idmap, _key = _MC.get_parts_tree_index(
        gpd.GeoDataFrame(geometry=coast_wgs, crs=WGS84), simplify_m=0.0
    )

    def _snap_xy(pt): return (round(pt.x/snap_m)*snap_m, round(pt.y/snap_m)*snap_m)

    nodes_xy, node_index, node_edges, edges = [], {}, [], []
    def _nid(pt):
        xy = _snap_xy(pt)
        i = node_index.get(xy)
        if i is None:
            i = len(nodes_xy); nodes_xy.append(xy); node_index[xy] = i; node_edges.append([])
        return i

    for i, ln in enumerate(parts_m):
        if not ln or ln.is_empty: continue
        u = _nid(Point(ln.coords[0])); v = _nid(Point(ln.coords[-1]))
        eidx = len(edges)
        edges.append(_Edge(u=u, v=v, length=float(ln.length), geom=ln,
                           parent_idx=i, parent_feature_id=int(part_ids[i])))
        node_edges[u].append(eidx); node_edges[v].append(eidx)

    find, unite = _dsu_build(len(nodes_xy))
    for e in edges: unite(e.u, e.v)
    comp = [find(i) for i in range(len(nodes_xy))]

    edge_geoms = [e.geom for e in edges]
    edge_tree = STRtree(edge_geoms)
    edge_idx_by_id = {id(g): i for i, g in enumerate(edge_geoms)}

    if log_timing:
        logger.info("coast_graph summary: nodes=%d edges=%d comps=%d snap_m=%.1f built=%.2fs",
                    len(nodes_xy), len(edges), len(set(comp)), float(snap_m), time.time()-t0)

    return CoastGraph(nodes_xy=nodes_xy, edges=edges, node_edges=node_edges,
                      edge_tree=edge_tree, edge_idx_by_id=edge_idx_by_id, node_comp_id=comp)

def _nearest_edge_id(cg, pt_m):
    """
    Return edge index for a metric point. Works with STRtree.nearest() that
    returns either an int index or a geometry object.
    """
    obj = cg.edge_tree.nearest(pt_m)
    # Shapely 2.x may return index
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    # Shapely 1.8 returns geometry; map via id
    ei = cg.edge_idx_by_id.get(id(obj))
    if ei is not None:
        return ei
    # Last resort: distance match
    d = [np.inf if e.geom is None else pt_m.distance(e.geom) for e in cg.edges]
    return int(np.argmin(d))

def _graph_io_smoketest(cg, nodes, *, log_fn=print):
    """Validate graph + router I/O. Run once after graph build."""
    ok = True
    need = ["edges","edge_tree","edge_idx_by_id","node_edges","node_comp_id"]
    for k in need:
        if not hasattr(cg, k):
            log_fn(f"[IO] missing attr: {k}"); ok = False
    if getattr(cg, "edges", None):
        e0 = cg.edges[0]
        for k in ["geom","length","u","v","parent_feature_id"]:
            if not hasattr(e0, k):
                log_fn(f"[IO] _Edge missing: {k}"); ok = False
    # probe nearest return kind
    p = nodes[0]["pt"]
    obj = cg.edge_tree.nearest(p)
    log_fn(f"[IO] nearest type: {type(obj).__name__}")
    try:
        eid = _nearest_edge_id(cg, p)
        g = cg.edges[eid].geom
        log_fn(f"[IO] nearest eid: {eid}, geom_type={getattr(g, 'geom_type', None)}")
    except Exception as ex:
        log_fn(f"[IO] nearest map failed: {ex}"); ok = False
    # tiny A* check between two close nodes
    try:
        q = nodes[min(1, len(nodes)-1)]["pt"]
        test_path = route_along_coast_astar(cg, p, q, delta_max_m=100_000, window_m=None, log_timing=False)
        ok_path = bool(test_path) and not getattr(test_path, "is_empty", False)
        log_fn(f"[IO] astar smoke ok={ok_path} len={getattr(test_path,'length',None)}")
    except Exception as ex:
        log_fn(f"[IO] astar threw: {ex}"); ok = False
    return ok

def _node_dist(nxy: Tuple[float,float], pt: Point) -> float:
    dx = nxy[0] - pt.x; dy = nxy[1] - pt.y
    return math.hypot(dx, dy)

def _pt_on_edge(e: _Edge, s: float) -> Point:
    # s in meters along geometry
    try:
        return e.geom.interpolate(s, normalized=False)
    except Exception:
        # fallback: substring midpoint
        seg = _substring(e.geom, max(0.0, min(s, e.length-1e-6)), max(0.0, min(s+1e-6, e.length)))
        if seg.is_empty:
            return Point(e.geom.coords[0])
        return seg.centroid

def _allowed_nodes_mask(cg: CoastGraph, A: Point, B: Point, window_m: Optional[float]) -> Optional[np.ndarray]:
    if window_m is None or not np.isfinite(window_m):
        return None
    mask = np.zeros(len(cg.nodes_xy), dtype=bool)
    wx = float(window_m)
    for i,(x,y) in enumerate(cg.nodes_xy):
        if (x - A.x)*(x - A.x) + (y - A.y)*(y - A.y) <= wx*wx:
            mask[i] = True
            continue
        if (x - B.x)*(x - B.x) + (y - B.y)*(y - B.y) <= wx*wx:
            mask[i] = True
    return mask

def route_along_coast_astar(
    cg: CoastGraph,
    A_m: Point,
    B_m: Point,
    *,
    delta_max_m: float = 250_000.0,
    split_tol_m: float = 5.0,
    window_m: Optional[float] = None,
    log_timing: bool = False,
) -> Optional[LineString]:
    """
    Coast-constrained shortest path from A to B along the coastline graph.
    Returns a LineString in metric CRS (WEBM) or None if no path found/over cap.
    The caller is responsible for any terminal stubs beyond the coast path.
    """
    # ---- map terminals to host edges ----
    start_eid = _nearest_edge_id(cg, A_m)
    goal_eid  = _nearest_edge_id(cg, B_m)
    if start_eid is None or goal_eid is None:
        return None
    E_a = cg.edges[start_eid]
    E_b = cg.edges[goal_eid]

    # fast component gate
    if cg.node_comp_id[E_a.u] != cg.node_comp_id[E_b.u]:
        return None

    # distances along host edges to terminal projections
    sa = float(E_a.geom.project(A_m))
    sb = float(E_b.geom.project(B_m))

    # create temp nodes if terminals are not near existing endpoints
    def _snap_or_temp(edge: _Edge, s: float, pt: Point) -> Tuple[int, Optional[Tuple]]:
        du = _node_dist((cg.nodes_xy[edge.u][0], cg.nodes_xy[edge.u][1]), pt)
        if du <= split_tol_m:
            return edge.u, None
        dv = _node_dist((cg.nodes_xy[edge.v][0], cg.nodes_xy[edge.v][1]), pt)
        if dv <= split_tol_m:
            return edge.v, None
        # negative ids to denote ephemeral nodes; not used as indices into cg arrays
        temp_id = -(1 + int(round(s)))
        return temp_id, (edge, s)

    start_node, start_tmp = _snap_or_temp(E_a, sa, A_m)
    goal_node,  goal_tmp  = _snap_or_temp(E_b, sb, B_m)

    # optional spatial window mask
    allowed_mask = _allowed_nodes_mask(cg, A_m, B_m, window_m)

    # ---- A* over nodes (+ ephemeral terminals) ----
    def _neighbors(node_id: int):
        if node_id >= 0:
            for eid in cg.node_edges[node_id]:
                e = cg.edges[eid]
                other = e.v if e.u == node_id else e.u
                if allowed_mask is not None and not allowed_mask[other]:
                    continue
                yield other, eid, None  # via real edge
        else:
            # ephemeral node connects to both endpoints of its host edge
            edge, s = (start_tmp if node_id == start_node else goal_tmp)
            yield edge.u, None, ("temp", edge, s, edge.u)
            yield edge.v, None, ("temp", edge, s, edge.v)

    def _heuristic(nid: int) -> float:
        # Euclidean to B_m
        if nid >= 0:
            x, y = cg.nodes_xy[nid]
            return math.hypot(x - B_m.x, y - B_m.y)
        edge, s = (start_tmp if nid == start_node else goal_tmp)
        p = _pt_on_edge(edge, s)
        return p.distance(B_m)

    g_cost: Dict[int, float] = {start_node: 0.0}
    parent: Dict[int, Tuple[int, Optional[int], Optional[Tuple]]] = {}
    openpq: List[Tuple[float, int]] = []
    heapq.heappush(openpq, (_heuristic(start_node), start_node))

    max_f = float(delta_max_m) + 1e-6
    visited: Set[int] = set()
    _visits = 0

    with _maybe_timer("astar:total", log_timing):
        while openpq:
            f, u = heapq.heappop(openpq)
            if u in visited:
                continue
            visited.add(u)

            if u == goal_node:
                break
            if f > max_f:
                return None

            gu = g_cost[u]
            _visits += 1
            if _trace_enabled() and (_visits % 5000 == 0):
                logger.debug("astar step visits=%d open=%d best_f=%.0f g=%.0f", _visits, len(openpq), f, gu)

            for v, via_eid, via_tmp in _neighbors(u):
                # step cost
                if via_eid is not None:
                    e = cg.edges[via_eid]
                    step = e.length
                else:
                    # temp hop along a host edge from s to endpoint
                    kind, e, s, end_node = via_tmp
                    step = abs(s - 0.0) if end_node == e.u else abs(e.length - s)

                alt = gu + step
                if alt > delta_max_m + 1e-6:
                    continue
                if (v not in g_cost) or (alt < g_cost[v] - 1e-9):
                    g_cost[v] = alt
                    parent[v] = (u, via_eid, via_tmp)
                    fv = alt + _heuristic(v)
                    heapq.heappush(openpq, (fv, v))

    if goal_node not in parent and goal_node != start_node:
        return None

    # ---- reconstruct and emit coast path geometry ----
    chain: List[Tuple[int, Optional[int], Optional[Tuple]]] = []
    cur = goal_node
    while cur != start_node:
        prev, via_eid, via_tmp = parent[cur]
        chain.append((cur, via_eid, via_tmp))
        cur = prev
    chain.reverse()

    parts: List[LineString] = []

    def _append_edge_segment(e: _Edge, n_from: int, n_to: int,
                             from_tmp: Optional[Tuple], to_tmp: Optional[Tuple]):
        if from_tmp is not None:
            _, eF, sF, endF = from_tmp
            s0 = float(sF) if eF is e else (0.0 if n_from == e.u else e.length)
        else:
            s0 = 0.0 if n_from == e.u else e.length

        if to_tmp is not None:
            _, eT, sT, endT = to_tmp
            s1 = float(sT) if eT is e else (0.0 if n_to == e.u else e.length)
        else:
            s1 = 0.0 if n_to == e.u else e.length

        a, b = (s0, s1) if s0 <= s1 else (s1, s0)
        if abs(b - a) < 1e-6:
            return
        seg = _substring(e.geom, a, b)
        if seg is not None and not seg.is_empty:
            parts.append(seg)

    prev_node = start_node
    prev_tmp = start_tmp
    for node, via_eid, via_tmp in chain:
        if via_eid is not None:
            e = cg.edges[via_eid]
            n_from = prev_node if prev_node >= 0 else prev_tmp[3]
            n_to   = node      if node      >= 0 else via_tmp[3]
            _append_edge_segment(e, n_from, n_to,
                                 prev_tmp if prev_node < 0 else None,
                                 via_tmp  if node      < 0 else None)
        # temp→endpoint hop contributes when the subsequent real edge is appended
        prev_node = node
        prev_tmp = via_tmp

    if not parts:
        return None

    u = unary_union(parts)
    merged = u if isinstance(u, LineString) else linemerge(u)
    if merged.geom_type == "MultiLineString":
        merged = max(merged.geoms, key=lambda g: g.length)

    if log_timing:
        try:
            logger.info("astar: visits=%d explored=%d chain=%d length_km=%.3f",
                        _visits, len(visited), len(chain), float(merged.length) / 1000.0)
        except Exception:
            pass

    return merged

def _emit_with_terminal_stubs(route_geom: LineString,
                              A_pt: Point, B_pt: Point,
                              cg: CoastGraph,
                              tol_m: float = 5.0) -> Optional[LineString]:
    if not route_geom or getattr(route_geom, "is_empty", False):
        return None

    # host edges at terminals
    a_eid = _nearest_edge_id(cg, A_pt)
    b_eid = _nearest_edge_id(cg, B_pt)
    Ea = cg.edges[a_eid]; Eb = cg.edges[b_eid]

    # distances along host edges to terminals
    sA = float(Ea.geom.project(A_pt))
    sB = float(Eb.geom.project(B_pt))

    # distances along host edges to where the ROUTE actually begins/ends
    first_pt = Point(route_geom.coords[0])
    last_pt  = Point(route_geom.coords[-1])
    tA = float(Ea.geom.project(first_pt))
    tB = float(Eb.geom.project(last_pt))

    # build stubs robustly by projection, order-agnostic
    def _seg(ls, s0, s1):
        a, b = (s0, s1) if s0 <= s1 else (s1, s0)
        if abs(b - a) <= tol_m:  # ignore tiny gaps
            return None
        return _substring(ls, a, b)

    pre  = _seg(Ea.geom, sA, tA)
    post = _seg(Eb.geom, tB, sB)

    pieces = []
    if pre  is not None and not pre.is_empty:  pieces.append(pre)
    pieces.append(route_geom)
    if post is not None and not post.is_empty: pieces.append(post)

    u = unary_union(pieces)
    out = u if isinstance(u, LineString) else linemerge(u)
    if out.geom_type == "MultiLineString":
        out = max(out.geoms, key=lambda g: g.length)
    return None if out.is_empty else out


# ---------- normalize ----------
def _detect_lon_lat(df):
    return _h_detect_lon_lat(df)

def _detect_event_id_col(df, preferred: str | None):
    return _h_detect_event_id_col(df, preferred)

def _detect_height_col(df, preferred: str | None):
    return _h_detect_height_col(df, preferred)

def _endpoints(line: LineString):
    return Point(line.coords[0]), Point(line.coords[-1])

def _host_proj(pt, parts_tree, parts_m, part_ids, _idx_of_near):
    j = _idx_of_near(parts_tree.nearest(pt))
    host = parts_m[j]
    s = float(host.project(pt))
    return {
        "j": j,
        "pid": int(part_ids[j]),
        "s": s,
        "pt_on_host": host.interpolate(s),
    }

def normalize_runups(
        csv_path: str,
        *,
        id_col: str | None = None,
        height_col: str | None = None,
        lon_col: str | None = None,
        lat_col: str | None = None,
        event_id_filter: str | int | None = None,
    ) -> gpd.GeoDataFrame:

    df = pd.read_csv(csv_path)
    lon_col, lat_col = (lon_col, lat_col) if lon_col and lat_col else _detect_lon_lat(df)
    id_col = _detect_event_id_col(df, id_col)
    height_col = _detect_height_col(df, height_col)

    df_std = df.rename(columns={id_col: "tsunamiEventId", height_col: "runupHt"})
    gdf = gpd.GeoDataFrame(
        df_std,
        geometry=gpd.points_from_xy(df_std[lon_col], df_std[lat_col]),
        crs=WGS84,
    )

    if event_id_filter is not None:
        want = str(event_id_filter).strip().lower()
        pre = len(gdf)
        gdf = gdf[gdf["tsunamiEventId"].astype(str).str.strip().str.lower() == want].copy()
        logger.info(
            "normalize_runups filter event_id=%station_meters → %d/%d rows", 
            str(event_id_filter), 
            len(gdf), 
            pre
            )

    return gdf[["tsunamiEventId","runupHt","geometry"]]

# ---------- snapping ----------
def snap_runups_to_coast(
    runups_gdf: gpd.GeoDataFrame,
    coast_lines_gs: gpd.GeoSeries,
    *,
    max_snap_km: float = 15.0,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    
    t0 = time.perf_counter()
    runups = ensure_wgs84(runups_gdf).copy()
    coast_lines = ensure_wgs84(coast_lines_gs)
    
    logger.info("snap_runups_to_coast max_snap_km=%.2f", max_snap_km)
    logger.debug("runups CRS=%s, coast CRS=%s", runups.crs, coast_lines.crs)

    valid = runups.geometry.apply(
        lambda p: isinstance(p, Point) and -90 <= p.y <= 90 and -180 <= p.x <= 180
        )
    invalid = runups.loc[~valid].copy()
    invalid["reason"] = "invalid_coord"
    runups = runups.loc[valid].copy()

    runups_meters = runups.to_crs(WEBM)

    with _timer("snap:prepare_parts"):
        parts_meters, parent_ids = _coast_parts_3857(coast_lines)
        if not parts_meters:
            empty = gpd.GeoDataFrame(
                columns=[
                    "tsunamiEventId",
                    "runupHt",
                    "parent_feature_id",
                    "station_m",
                    "geometry"
                ], 
                geometry="geometry", 
                crs=WGS84
                )
            return empty, invalid
        logger.info("  coast parts: %d", len(parts_meters))

    with _timer("snap:build_stree"):
        # Tree already built and cached inside TsunamiMemCache; rebuild here only if needed
        try:
            _parts, _parents, tree, _k = _MC.get_parts_3857_and_tree(gpd.GeoDataFrame(geometry=gpd.GeoSeries(coast_lines), crs=WGS84))
        except Exception:
            tree = STRtree(parts_meters)
        max_snap_meters = max_snap_km * 1000.0
        logger.info("  spatial index built")

    snapped_rows = []
    # debug: track how many distinct parts are used per parent_feature_id
    _parent_to_parts: Dict[int, set] = defaultdict(set)
    with _timer("snap:iterate_points"):
        rejected_entries = []
        snapped_count = 0; rejected_count = 0

    for idx, row in runups_meters.iterrows():
        runup_point_meters = row.geometry
        candidate_pairs = [
            (part_idx, line_m)
            for part_idx, line_m in _tree_query_pairs(tree, parts_meters, runup_point_meters, max_snap_meters)
            if line_m.distance(runup_point_meters) <= max_snap_meters
        ]
        
        if not candidate_pairs:
            rejected_count += 1
            logger.debug("runup idx=%s rejected: no nearby coast within %.1f km", idx, max_snap_km)
            rejected_entry = runups.loc[[idx]].copy()
            rejected_entry["reason"] = "beyond_max_snap"
            rejected_entries.append(rejected_entry)
            continue

        else:
            snapped_count += 1
            #logger.debug("runup idx=%station_meters snapped to %d candidates within %.1f km", idx, len(candidate_pairs), max_snap_km)

        part_index, part_line_meters = min(candidate_pairs, key=lambda t: t[1].distance(runup_point_meters))
        parent_feature_id = int(parent_ids[part_index])
        # debug: record mapping for diagnostics
        try:
            _parent_to_parts[parent_feature_id].add(int(part_index))
        except Exception:
            pass

        station_m = float(part_line_meters.project(runup_point_meters))
        snapped_rows.append({
            "tsunamiEventId": row.get("tsunamiEventId", None),
            "runupHt": row.get("runupHt", None),
            "parent_feature_id": parent_feature_id,
            "station_m": station_m,
            "geometry": runup_point_meters,
        })
    logger.info("  snapped: %d, rejected: %d (beyond_max=%d, invalid_coord=%d)",
            snapped_count, rejected_count + len(invalid), rejected_count, len(invalid))

    # debug: log distribution of distinct parts per parent_feature_id
    try:
        parts_per_parent = {pid: len(s) for pid, s in _parent_to_parts.items()}
        hist = Counter(parts_per_parent.values())
        multi = sum(1 for v in parts_per_parent.values() if v > 1)
        logger.info(
            "snap_runups_to_coast debug: parents=%d with_multi=%d max_parts=%d hist=%s",
            len(parts_per_parent), multi, (max(parts_per_parent.values()) if parts_per_parent else 0), dict(hist)
        )
        if multi:
            examples = [pid for pid, n in parts_per_parent.items() if n > 1][:10]
            logger.debug("parents with >1 distinct parts (first 10): %s", examples)
    except Exception:
        pass

    snapped_m = gpd.GeoDataFrame(snapped_rows, geometry="geometry", crs=WEBM) if snapped_rows else gpd.GeoDataFrame(columns=["tsunamiEventId","runupHt","parent_feature_id","station_m","geometry"], geometry="geometry", crs=WEBM)
    snapped = snapped_m.to_crs(WGS84)
    rejected = pd.concat(rejected_entries, ignore_index=True) if rejected_entries else pd.DataFrame(columns=list(runups.columns)+["reason"])
    rejected = gpd.GeoDataFrame(rejected, geometry="geometry", crs=WGS84)
    if not invalid.empty:
        rejected = pd.concat([rejected, invalid], ignore_index=True)
        rejected = gpd.GeoDataFrame(rejected, geometry="geometry", crs=WGS84)
    return snapped, rejected

# ---------- segments from runups ----------
def _span_km_from_height(H: float, alongshore_km_min=4.0, km_per_meter_factor=8.0, height_exponent=1.0, alongshore_km_max=60.0) -> float:
    if H is None or np.isnan(H):
        return alongshore_km_min
    L = km_per_meter_factor * (max(0.0, float(H)) ** height_exponent)
    return float(np.clip(L, alongshore_km_min, alongshore_km_max))


def _height_to_length_km(h_m: float, *, alongshore_km_min: float, alongshore_km_max: float, km_per_meter_factor: float, height_exponent: float) -> float:
    if h_m is None or not np.isfinite(h_m) or h_m <= 0:
        return alongshore_km_min
    L = (km_per_meter_factor * float(h_m)**float(height_exponent))
    return float(max(alongshore_km_min, min(alongshore_km_max, L)))

def _merge_intervals(intervals: list[tuple[float,float]], *, gap_merge_m: float = 0.0) -> list[tuple[float,float]]:
    return _h_merge_intervals(intervals, gap_merge_m=gap_merge_m)

def _cut_parent_substrings(parent_line_m, ivals_m: list[tuple[float,float]]) -> list[LineString]:
    return _h_cut_parent_substrings(parent_line_m, ivals_m)

def _explode_coast_parts(coast_wgs) -> tuple[list[LineString], list[int]]:
    gser = gpd.GeoSeries(coast_wgs, crs=WGS84) if isinstance(coast_wgs, (list, tuple)) else (
        coast_wgs if isinstance(coast_wgs, gpd.GeoSeries) else coast_wgs.geometry
    )
    gser = gser.explode(index_parts=False, ignore_index=True)
    parts = []
    parent = []
    for i, g in enumerate(gser.values):
        if g is None or g.is_empty: continue
        if g.geom_type == "LineString":
            parts.append(g); parent.append(i)
        elif g.geom_type == "MultiLineString":
            for gg in g.geoms:
                parts.append(gg); parent.append(i)
    return parts, parent

def build_runup_segments(
    snapped_points: gpd.GeoDataFrame,
    coast_wgs, *,
    height_col: str = "runupHt",
    height_meters_min: float = 0.5,
    alongshore_km_min: float = 75.0,
    alongshore_km_max: float = 250.0,
    km_per_meter_factor: float = 80.0,
    height_exponent: float = 1.0,
    merge_tol_meters: float = 1000.0,
    length_multiplier: float = 1.0,
) -> gpd.GeoDataFrame:
    
    logger.info(
        "build_runup_segments_v2 height_meters_min=%.2f L[min=%.1f max=%.1f] beta=%.1f height_exponent=%.2f merge_tol_meters=%.0f",
        height_meters_min, 
        alongshore_km_min, 
        alongshore_km_max, 
        km_per_meter_factor, 
        height_exponent, 
        merge_tol_meters
        )
    
    if snapped_points.empty:
        return gpd.GeoDataFrame(
            columns=[
                "parent_feature_id", 
                "geometry"
                ], 
            geometry=
            "geometry", 
            crs=WGS84
            )
    snapped_points = ensure_wgs84(snapped_points)

    # Use in-memory cache for exploded WGS84 parts
    with _timer("segments:get_parts_wgs_exploded"):
        parts_wgs, parent_ids, _key = _MC.get_parts_wgs_exploded(
            gpd.GeoDataFrame(geometry=gpd.GeoSeries(ensure_wgs84(coast_wgs)), crs=WGS84)
        )
        if parts_wgs is None or getattr(parts_wgs, "empty", True):
            return gpd.GeoDataFrame(
                columns=["parent_feature_id", "geometry"], geometry="geometry", crs=WGS84
            )
    with _timer("segments:to_metric_parts"):
        parts_meters = [gpd.GeoSeries([g], crs=WGS84).to_crs(WEBM).iloc[0] for g in parts_wgs]

    j_by_pid = {int(pid): j for j, pid in enumerate(parent_ids)}
    pid_by_j = {j: int(pid) for j, pid in enumerate(parent_ids)}

    candidates = snapped_points.loc[snapped_points[height_col] >= float(height_meters_min)]
    if candidates.empty:
        return gpd.GeoDataFrame(columns=["parent_feature_id", "geometry"], geometry="geometry", crs=WGS84)

    segments_WEBM: list[LineString] = []
    parents: list[int] = []
    
    with _timer("segments:per_parent_build"):
        for parent_feature_id, df_parent in candidates.groupby("parent_feature_id", sort=False):
            try:
                j = j_by_pid.get(int(parent_feature_id))
                if j is None:
                    continue
                line_meters = parts_meters[int(j)]
            except Exception:
                continue

            intervals_meters: list[tuple[float, float]] = []
            for _, row in df_parent.iterrows():
                station_meters = float(row.get("station_m", float("nan")))
                if not np.isfinite(station_meters):
                    continue
                length_km = _height_to_length_km(
                    row[height_col],
                    alongshore_km_min=alongshore_km_min,
                    alongshore_km_max=alongshore_km_max,
                    km_per_meter_factor=km_per_meter_factor,
                    height_exponent=height_exponent,
                )
                half = (length_km * float(length_multiplier)) * 500.0
                intervals_meters.append((station_meters - half, station_meters + half))
                segs = _cut_parent_substrings(line_meters, intervals_meters)
                segments_WEBM.extend(segs)
                parents.extend([parent_feature_id]*len(segs))

    if not segments_WEBM:
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)

    logger.info("segments: built=%d parents=%d", len(segments_WEBM), len(set(parents)))

    segments_wgs84 = gpd.GeoSeries(segments_WEBM, crs=WEBM).to_crs(WGS84)
    out = gpd.GeoDataFrame({"parent_feature_id": parents, "geometry": segments_wgs84}, geometry="geometry", crs=WGS84)
    return out

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

    with _timer("poly:to_metric"):
        segments_WEBM = to_metric(segs)

    # Build per-parent snapped points in metric CRS
    pts_by_parent: Dict[int, np.ndarray] = {}
    H_by_parent: Dict[int, np.ndarray] = {}
    with _timer("poly:prep_snapped_lookup"):
        for parent_feature_id, group in snapped.groupby("parent_feature_id"):
            arr = np.array([[float(row["station_m"]), float(row.get(height_col, 0.0))] for _, row in group.iterrows()], dtype=float)
            if arr.size == 0:
                continue
            arr = arr[arr[:,0].argsort()]
            pts_by_parent[int(parent_feature_id)] = arr[:,0]
            H_by_parent[int(parent_feature_id)] = arr[:,1]

    discs = []
    with _timer("poly:sample_and_buffer"):
        for _, row in segments_WEBM.iterrows():
            parent_feature_id = int(row.get("parent_feature_id", -1))
            line = row.geometry
            L = line.length
            if L <= 0:
                continue
            n = max(2, int(math.ceil(L / float(sample_step_m))))
            stations = np.linspace(0.0, L, n)
            for station_meters in stations:
                p = line.interpolate(station_meters)
                # nearest snapped height on same parent
                H = 0.0
                if parent_feature_id in pts_by_parent:
                    S = pts_by_parent[parent_feature_id]
                    hvals = H_by_parent[parent_feature_id]
                    j = int(np.searchsorted(S, station_meters))
                    candidates = []
                    if j < len(S): candidates.append((abs(S[j]-station_meters), hvals[j]))
                    if j > 0: candidates.append((abs(S[j-1]-station_meters), hvals[j-1]))
                    if candidates:
                        H = max(0.0, float(min(candidates, key=lambda t: t[0])[1]))
                D_km = max(D_min_km, min(D_max_km, alpha_km_per_m * (H ** gamma)))
                discs.append(p.buffer(D_km * 1000.0))

    # Prefer union_all if available (Shapely 2), else fallback to unary_union
    if discs:
        with _timer("poly:union"):
            try:
                from shapely import union_all as _union_all
                poly = _union_all(discs)
            except Exception:
                from shapely.ops import unary_union as _uun
                poly = _uun(discs)
    else:
        poly = None
    logger.info("runup_segments_to_inland_poly discs=%d has_poly=%s", len(discs), bool(poly))
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
        with _timer("buffer_on_land:buffer_only"):
            return lines.to_crs(WEBM).buffer(width_m).to_crs(WGS84)

    with _timer("buffer_on_land:buffer"):
        polys = lines.to_crs(WEBM).buffer(width_m).to_crs(WGS84)
    gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys), crs=WGS84)
    with _timer("buffer_on_land:intersect_land"):
        return gpd.overlay(gdf, land, how="intersection", keep_geom_type=True)

# Drop-in replacement for runups.py:infill_coast_unified
# New design: endpoint chaining using cache part-ids and coast_key.
# - Uses tsunami_mem_cache to align terminals/boosters and the coast graph topology.
# - Prefers explicit (coast_key, part_id, s_m) if present on inputs; otherwise falls back to nearest-part projection.
# - Multi-pass daisy chaining between terminals with optional boosters. A* along coast with hard hop/total caps.
# - Detailed tracing to identify bottlenecks.

def infill_coast_unified(
    snapped_gdf,                          # optional boosters source (runups_snapped)
    base_segs_gdf,                        # REQUIRED: runups_segments_linestring; ideally has part_id + coast_key
    coast_lines_gs,                       # REQUIRED: coastline lines (WGS84 or any)
    ray_hits_gdf=None,                    # optional boosters source
    *,
    # chaining controls
    hop_max_km: float | None = None,      # max coastal distance per hop; default=delta_max_km
    total_max_km: float | None = None,    # total chain cap; default=delta_max_km
    k_neighbors: int = 6,                 # candidate neighbors per node
    passes: int = 3,                      # multi-pass daisy chain
    # coastal graph connectivity
    graph_snap_m: float = 250.0,          # snapping tolerance when building the coast graph
    gap_padding_m: float | None = None,   # alias for graph_snap_m
    # cross-parent allowed
    allow_cross_parent: bool = True,
    # boosters
    use_lowht_seed: bool = False,
    use_rays_seeds: bool = False,
    ht_lo_m: float = 0.2,
    ht_hi_m: float = 0.5,
    # legacy CLI knobs kept for compatibility (not used by chaining core)
    pad_km: float = 1.0,
    delta_max_km: float = 750.0,
    # logging and extras
    log_decisions: bool = False,
    log_rejects_limit: int = 200,
    log_timing: bool = False,
    **_ignore_kwargs,
):
    """
    Endpoint-to-endpoint chaining with strict cache alignment.

    Inputs:
      - base_segs_gdf: LINESTRING features that represent existing runup segments.
        Preferred columns: 'coast_key', 'part_id'. If absent, the function will project to the nearest
        exploded coast part from tsunami_mem_cache for this call's coast and log a warning.
      - coast_lines_gs: coastline lines; used to build a cached graph and as host for projection.
      - snapped_gdf (optional boosters): points with 'runupHt' and preferably 'coast_key'/'part_id'.
      - ray_hits_gdf (optional boosters): points; preferably include ('parent_id' or 'part_id') and 's_m'.

    Returns: GeoDataFrame (WGS84) with columns
      - geometry: LINESTRING
      - coast_key: cache key for the coast used
      - part_id_start, part_id_end: exploded-part ids at endpoints
      - parent_feature_id: compatibility tag (set to part_id_start)
    """
    # ---- local imports / globals from module ----
    import math, time
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point, LineString, MultiLineString
    from shapely.ops import linemerge
    from contextlib import contextmanager
    import logging as _logging

    # These utilities must exist in runups.py module scope
    # - ensure_wgs84, to_metric, build_coast_graph_from_coast, route_along_coast_astar
    # - WGS84, WEBM, _MC (TsunamiMemCache instance)
    logger = _logging.getLogger("natural_disasters.infill.unified.cachechain")

    # --- timers
    @contextmanager
    def _timer(tag: str):
        t0 = time.time()
        yield
        if logger.isEnabledFor(_logging.DEBUG):
            logger.debug("%s in %.2fs", tag, time.time() - t0)

    # --- defaults
    if hop_max_km is None:
        hop_max_km = float(delta_max_km)
    if total_max_km is None:
        total_max_km = float(delta_max_km)
    hop_max_m = float(hop_max_km) * 1000.0
    total_max_m = float(total_max_km) * 1000.0
    snap_m = float(gap_padding_m) if gap_padding_m is not None else float(graph_snap_m)

    # --- normalize inputs
    coast_wgs = gpd.GeoSeries(ensure_wgs84(coast_lines_gs)) if not isinstance(coast_lines_gs, gpd.GeoSeries) else ensure_wgs84(coast_lines_gs)
    base = gpd.GeoDataFrame(geometry=ensure_wgs84(base_segs_gdf).geometry, crs=WGS84).join(ensure_wgs84(base_segs_gdf).drop(columns=["geometry"], errors="ignore"))
    snaps = None if snapped_gdf is None else ensure_wgs84(snapped_gdf)
    rays  = None if ray_hits_gdf is None else ensure_wgs84(ray_hits_gdf)

    if base is None or base.empty:
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)

    parts_m, part_ids, parts_tree, idmap, coast_key = _MC.get_parts_tree_index(
        gpd.GeoDataFrame(geometry=coast_wgs, crs=WGS84), simplify_m=0.0
    )

    pid_to_idx = {int(pid): i for i, pid in enumerate(part_ids)}  # replaces all pid_to_idx

    def _idx_of_near(near_obj):
        """
        Map STRtree.nearest(...) result to parts_m index.
        Supports Shapely/PyGEOS variants that return either an int index or a geometry.
        """
        import numpy as _np

        # Case 1: tree returns index directly
        if isinstance(near_obj, (int, _np.integer)):
            return int(near_obj)

        # Case 2: tree returns geometry; use id->idx map first
        i = idmap.get(id(near_obj))
        if i is not None:
            return i

        # Case 3: fallback by identity/equality
        for j, g in enumerate(parts_m):
            if g is near_obj:
                return j
            try:
                if hasattr(near_obj, "equals_exact") and near_obj.equals_exact(g, 0.0):
                    return j
            except Exception:
                pass

        # Case 4: last resort distance match (should rarely run)
        d = [_np.inf if g is None else near_obj.distance(g) for g in parts_m]
        return int(_np.argmin(d))

    # --- pull exploded parts and tree from cache to get a stable coast_key and part ids
    #with _timer("cache:get_parts_3857_and_tree"):
    #    parts_m, part_ids, tree, coast_key = _MC.get_parts_3857_and_tree(gpd.GeoDataFrame(geometry=coast_wgs, crs=WGS84))
    #pid_to_idx = {int(pid): j for j, pid in enumerate(part_ids)}

    # --- build coast graph once with desired snap
    with _timer("coast_graph"):
        cg = _MC.get_coast_graph(coast_wgs, snap_m=snap_m, simplify_m=None)

    # ---------- node constructors ----------
    def _host_for_row(row) -> tuple[int, object] | None:
        """Resolve a row to (part_id, host_line_m). Prefer explicit part_id; else nearest part via STRtree."""
        pid_val = row.get("part_id", None)
        if pid_val is not None and not pd.isna(pid_val):
            pid = int(pid_val)
            j = pid_to_idx.get(pid)
            if j is not None:
                return pid, parts_m[j]
        # fallback: nearest part in this coast
        geom = row.geometry
        if geom is None or geom.is_empty:
            return None
        # STRtree query
        #try:
        #    near = tree.nearest(geom)
        #except Exception:
            # fallback linear scan
        #    dists = [geom.distance(ln) if ln is not None else math.inf for ln in parts_m]
        #    j = int(np.argmin(dists))
        #    return int(part_ids[j]), parts_m[j]
        # find index of 'near' inside parts_m
        # STRtree returns the geometry; map back by identity
        #idx = None
        #for jj, ln in enumerate(parts_m):
        #    if ln is near:
        #        idx = jj; break
        #if idx is None:
            # fallback distance match
        #    dists = [geom.distance(ln) if ln is not None else math.inf for ln in parts_m]
        #    idx = int(np.argmin(dists))
        geom_m = to_metric(gpd.GeoSeries([geom], crs=WGS84)).iloc[0]
        near = parts_tree.nearest(geom_m)
        idx = _idx_of_near(near)
        
        return int(part_ids[idx]), parts_m[idx]

    def _make_terminal_rows(seg_row):
        """
        Build two terminal nodes (start/end) for one base segment.
        Returns: list[dict] with fields: kind, pid, s, pt, seg_id, seg_end.
        - kind   : "term"
        - pid    : coast part id hosting this endpoint
        - s      : arclength (m) along host part
        - pt     : shapely Point in metric CRS on host part at s
        - seg_id : identifier of the source segment (best-effort)
        - seg_end: 0 for first coord, 1 for last coord
        Requires outer-scope: parts_tree, parts_m, part_ids, pid_to_idx (optional),
                            to_metric, WGS84, _idx_of_near.
        """
        import pandas as pd
        from shapely.geometry import Point
        import geopandas as gpd

        g = seg_row.geometry
        if g is None or g.is_empty:
            return []

        # metric copy of this segment; pick longest LineString if multipart
        seg_m = to_metric(gpd.GeoSeries([g], crs=WGS84)).iloc[0]
        ls = max(seg_m.geoms, key=lambda x: x.length) if hasattr(seg_m, "geoms") else seg_m
        endpoints = [Point(ls.coords[0]), Point(ls.coords[-1])]

        # best-effort segment id
        seg_id = seg_row.get("fid", seg_row.get("segment_id", seg_row.get("id", -1)))
        try:
            seg_id = int(seg_id) if not pd.isna(seg_id) else -1
        except Exception:
            seg_id = -1

        # optional hint: if the segment row already has a part_id, prefer it
        seg_part_hint = seg_row.get("part_id", None)
        try:
            if pd.isna(seg_part_hint):
                seg_part_hint = None
            else:
                seg_part_hint = int(seg_part_hint)
        except Exception:
            seg_part_hint = None

        out = []
        for k, pt in enumerate(endpoints):
            # choose host part: prefer hinted part_id when available, else nearest
            if seg_part_hint is not None and "pid_to_idx" in globals():
                j = pid_to_idx.get(seg_part_hint)
                if j is None:
                    # fall back to nearest if hint missing in cache
                    near = parts_tree.nearest(pt)
                    j = _idx_of_near(near)
            else:
                near = parts_tree.nearest(pt)
                j = _idx_of_near(near)

            host = parts_m[j]
            s = float(host.project(pt))
            out.append({
                "kind": "term",
                "pid":  int(part_ids[j]),
                "s":    s,
                "pt":   host.interpolate(s),
                "seg_id": seg_id,
                "seg_end": 0 if k == 0 else 1,
            })

        return out



    def _make_booster_rows(df_like, kind: str) -> list[dict]:
        out = []
        if df_like is None or df_like.empty:
            return out
        Dm = to_metric(df_like)
        #has_pid = ("part_id" in Dm.columns) or ("parent_id" in Dm.columns) or ("parent_feature_id" in Dm.columns)
        for _, r in Dm.iterrows():
            g = r.geometry
            if g is None or g.is_empty:
                continue
            pid = None
            if "part_id" in r and not pd.isna(r["part_id"]):
                pid = int(r["part_id"])
            elif "parent_id" in r and not pd.isna(r["parent_id"]):
                pid = int(r["parent_id"])
            elif "parent_feature_id" in r and not pd.isna(r["parent_feature_id"]):
                # accept as part-id if it matches cache set; else fallback to nearest
                v = int(r["parent_feature_id"])
                if v in pid_to_idx:
                    pid = v
            
            if pid is not None and pid in pid_to_idx:
                host_m = parts_m[p] if (p := pid_to_idx[pid]) is not None else None
                if host_m is None:
                    continue
                s = float(host_m.project(g))
                out.append({"kind": kind, "pid": pid, "s": s, "pt": host_m.interpolate(s)})
            else:
                # nearest-part fallback
                #try:
                #    near = parts_tree.nearest(g)
                #except Exception:
                #    dists = [g.distance(ln) if ln is not None else math.inf for ln in parts_m]
                #    j = int(np.argmin(dists)); host_m = parts_m[j]; pid2 = int(part_ids[j])
                #else:
                #    # map geom back to index
                #    idx = None
                #    for jj, ln in enumerate(parts_m):
                #        if ln is near:
                #            idx = jj; break
                #    if idx is None:
                #        dists = [g.distance(ln) if ln is not None else math.inf for ln in parts_m]
                #        idx = int(np.argmin(dists))
                #    host_m = parts_m[idx]; pid2 = int(part_ids[idx])
                near = parts_tree.nearest(g)
                idx  = _idx_of_near(near)
                host_m = parts_m[idx]; pid2 = int(part_ids[idx])
                s = float(host_m.project(g))
                out.append({"kind": kind, "pid": pid2, "s": s, "pt": host_m.interpolate(s)})
        return out

    # ---------- collect nodes ----------
    nodes: list[dict] = []
    with _timer("collect_terminals"):
        for _, row in base.iterrows():
            nodes.extend(_make_terminal_rows(row))
    boosters = []
    with _timer("collect_boosters"):
        if use_lowht_seed and snaps is not None and not snaps.empty and "runupHt" in snaps.columns:
            # filter booster heights
            S = snaps[(snaps["runupHt"].astype(float) >= float(ht_lo_m)) & (snaps["runupHt"].astype(float) < float(ht_hi_m))]
            boosters.extend(_make_booster_rows(S, "boost"))
        if use_rays_seeds and rays is not None and not rays.empty:
            boosters.extend(_make_booster_rows(rays, "boost"))
    nodes.extend(boosters)

    if not nodes:
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)

    logger.info(
        "cachechain setup: coast_key=%s terminals=%d boosters=%d nodes=%d",
        str(coast_key), sum(1 for n in nodes if n["kind"]=="term"), len(boosters), len(nodes)
    )
    logger.info(
        "params: hop_max_km=%.1f total_max_km=%.1f k_neighbors=%d passes=%d snap_m=%.1f",
        float(hop_max_km), float(total_max_km), int(k_neighbors), int(passes), float(snap_m)
    )
    if logger.isEnabledFor(_logging.DEBUG) and nodes:
        _graph_io_smoketest(cg, nodes, log_fn=lambda m: logger.debug(m))

    # coordinates for neighbor work (WEBM)
    coords = np.array([(n["pt"].x, n["pt"].y) for n in nodes], dtype=float)

    # --- meta-edge routing cache
    edges: dict[tuple[int,int], tuple[float, object] | None] = {}
    edge_attempts = edge_euclid_skips = edge_astar_calls = edge_astar_succ = 0
    
    require_terminal_end = True

    def _edge(i, j):
        nonlocal edge_attempts, edge_euclid_skips, edge_astar_calls, edge_astar_succ
        edge_attempts += 1
        a, b = (i, j) if i < j else (j, i)
        key = (a, b)
        if key in edges: return edges[key]

        if not (nodes[i]["kind"] == "term" or nodes[j]["kind"] == "term"):
            edges[key] = None
            return None

        dx = math.hypot(coords[i,0]-coords[j,0], coords[i,1]-coords[j,1])
        if dx > hop_max_m:
            edge_euclid_skips += 1
            edges[key] = None
            return None

        edge_astar_calls += 1
        path = route_along_coast_astar(cg, nodes[i]["pt"], nodes[j]["pt"],
                               delta_max_m=hop_max_m, window_m=None, log_timing=False)
        if not path or getattr(path, "is_empty", False):
            edges[key] = None; return None

        bridge = _emit_with_terminal_stubs(path, nodes[i]["pt"], nodes[j]["pt"], cg, tol_m=0.0)
        if not bridge:
            edges[key] = None; return None
        edges[key] = (float(bridge.length), bridge)
        return edges[key]

    out_paths_m: list[LineString | MultiLineString] = []
    out_meta: list[tuple[int,int]] = []  # (start_term_idx, end_term_idx)
    import heapq

    for it in range(int(passes)):
        logger.info("pass %d start", it+1)
        term_indices = [i for i,n in enumerate(nodes) if n["kind"]=="term"]
        new_paths_this_pass = 0

        # Precompute neighbors once per pass using partial sort and Euclidean cutoff
        max_euclid = hop_max_m * 1.05
        d2 = ((coords[:,None,:] - coords[None,:,:])**2).sum(axis=2)
        nbrs: list[list[int]] = []
        k = int(k_neighbors) + 1
        for i in range(d2.shape[0]):
            cand = np.argpartition(d2[i], min(k, d2.shape[1]-1))[:k*3]  # widen pool
            by_parent = {}
            for j in cand:
                if j == i: continue
                if d2[i, j] > (max_euclid*max_euclid): continue
                pid = nodes[j]["pid"]
                if (pid not in by_parent) or d2[i, j] < d2[i, by_parent[pid]]:
                    by_parent[pid] = j
            idx = list(by_parent.values())
            if not idx:
                jmin = int(np.argmin(d2[i]))
                if jmin != i: idx = [jmin]
            nbrs.append(idx)

        for si in term_indices:
            # Dijkstra over the meta-graph
            dist = {si: 0.0}
            prev = {}
            done = set()
            heap = [(0.0, si)]
            goal = None

            while heap:
                d0, u = heapq.heappop(heap)
                if u in done: continue
                done.add(u)

                # goal: reach different-part terminal
                if u != si and nodes[u]["kind"]=="term" and nodes[u]["pid"] != nodes[si]["pid"]:
                    goal = u
                    break

                for v in nbrs[u]:
                    e = _edge(u, v)
                    if e is None:
                        continue
                    w, _geom = e
                    nd = d0 + w
                    if nd > total_max_m + 1e-6:
                        continue
                    if nd < dist.get(v, math.inf):
                        dist[v] = nd
                        prev[v] = u
                        heapq.heappush(heap, (nd, v))

            if goal is None:
                continue

            # reconstruct chain
            path_idxs = [goal]
            while path_idxs[-1] != si:
                path_idxs.append(prev[path_idxs[-1]])
            path_idxs.reverse()

            segs = []
            for a, b in zip(path_idxs[:-1], path_idxs[1:]):
                L, geom = _edge(a, b)
                segs.append(geom)
            merged = linemerge(MultiLineString(segs)) if len(segs) > 1 else segs[0]
            out_paths_m.append(merged)
            out_meta.append((si, goal))
            new_paths_this_pass += 1

        logger.info("pass %d new_paths=%d", it+1, new_paths_this_pass)
        if new_paths_this_pass == 0:
            break

        # Grow node set with endpoints of new paths to enable daisy chaining
        new_nodes = []
        for g_m in out_paths_m[-new_paths_this_pass:]:
            # endpoints p0, p1 (works for LineString or MultiLineString)
            if g_m.geom_type == "LineString":
                p0 = Point(g_m.coords[0]); p1 = Point(g_m.coords[-1])
            elif g_m.geom_type == "MultiLineString":
                geoms = list(g_m.geoms)
                p0 = Point(geoms[0].coords[0]); p1 = Point(geoms[-1].coords[-1])
            else:
                try:
                    geoms = list(getattr(g_m, "geoms", []))
                    p0 = Point(geoms[0].coords[0]); p1 = Point(geoms[-1].coords[-1])
                except Exception:
                    # as last resort, use endpoints of the merged path
                    seq = list(g_m.coords)
                    p0 = Point(seq[0]); p1 = Point(seq[-1])

            # choose host part independently for each end
            j0 = _idx_of_near(parts_tree.nearest(p0)); host0 = parts_m[j0]; pid0 = int(part_ids[j0])
            j1 = _idx_of_near(parts_tree.nearest(p1)); host1 = parts_m[j1]; pid1 = int(part_ids[j1])
            s0 = float(host0.project(p0)); s1 = float(host1.project(p1))

            new_nodes.append({"kind":"aux","pid":pid0,"s":s0,"pt":host0.interpolate(s0)})
            new_nodes.append({"kind":"aux","pid":pid1,"s":s1,"pt":host1.interpolate(s1)})
           
        if new_nodes:
            nodes.extend(new_nodes)
            coords = np.array([(n["pt"].x, n["pt"].y) for n in nodes], dtype=float)
        term_indices = [i for i,n in enumerate(nodes) if n["kind"] == "term"]

    logger.info("edge stats: attempts=%d euclid_skip=%d astar_calls=%d astar_succ=%d",
            edge_attempts, edge_euclid_skips, edge_astar_calls, edge_astar_succ)

    # --- emit
    if not out_paths_m:
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)

    out_wgs = gpd.GeoSeries(out_paths_m, crs=WEBM).to_crs(WGS84)
    out = gpd.GeoDataFrame({"geometry": out_wgs}, geometry="geometry", crs=WGS84)

    # tag endpoints
    part_start = []
    part_end   = []

    for (si, gi), g_m in zip(out_meta, out_paths_m):
        pid_s = int(nodes[si]["pid"]) if 0 <= si < len(nodes) else None
        pid_e = int(nodes[gi]["pid"]) if 0 <= gi < len(nodes) else None

        if g_m.geom_type == "LineString":
            p0 = Point(g_m.coords[0]); p1 = Point(g_m.coords[-1])
        elif g_m.geom_type == "MultiLineString":
            geoms = list(g_m.geoms)
            p0 = Point(geoms[0].coords[0]); p1 = Point(geoms[-1].coords[-1])
        else:
            try:
                geoms = list(getattr(g_m, "geoms", []))
                p0 = Point(geoms[0].coords[0]); p1 = Point(geoms[-1].coords[-1])
            except Exception:
                seq = list(g_m.coords)
                p0 = Point(seq[0]); p1 = Point(seq[-1])

        if pid_s is None:
            near0 = parts_tree.nearest(p0)
            j0 = _idx_of_near(near0)
            pid_s = int(part_ids[j0])

        if pid_e is None:
            near1 = parts_tree.nearest(p1)
            j1 = _idx_of_near(near1)
            pid_e = int(part_ids[j1])

        part_start.append(pid_s)
        part_end.append(pid_e)

    out["coast_key"] = str(coast_key)
    out["part_id_start"] = part_start
    out["part_id_end"] = part_end
    # compatibility tag
    out["parent_feature_id"] = out["part_id_start"].astype(int)

    # simple length metric for visibility
    try:
        out["length_km"] = to_metric(out).length.values / 1000.0
    except Exception:
        pass

    logger.info("cachechain done: rows=%d total_km=%.1f", len(out), float(out.get("length_km", pd.Series([])).sum()))
    return out
