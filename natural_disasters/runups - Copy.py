# runups.py
from __future__ import annotations
import math, numbers, logging, time, heapq, inspect
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict
from collections import defaultdict, Counter
from contextlib import contextmanager, nullcontext
from .infill_config import InfillConfig

import numpy as np
import pandas as pd
import geopandas as gpd
import os
import networkx as nx
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
from .tsunami_mem_cache import TsunamiMemCache

# Singleton in-memory coastline cache
_MC = TsunamiMemCache()

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

def build_coast_graph_from_coast(coast_lines_wgs: "gpd.GeoSeries",
                                 *,
                                 snap_m: float = 15.0,
                                 simplify_m: float = 0.0,
                                 log_timing: bool = False) -> CoastGraph:
    """
    Build a coastline graph once from WGS84 coast lines.
    Nodes = snapped endpoints of parts. Edges = parts geometries in WEBM.
    """
    with _maybe_timer("coast_graph:get_parts_3857", log_timing):
        parts_meters, parent_ids = _coast_parts_3857(ensure_wgs84(coast_lines_wgs))
    # memo key: (count, hash of bounds roughly, ...)
    bounds = [g.bounds if g is not None and not g.is_empty else (0,0,0,0) for g in parts_meters]
    if bounds:
        xs = [int(b[0]) for b in bounds] + [int(b[2]) for b in bounds]
        ys = [int(b[1]) for b in bounds] + [int(b[3]) for b in bounds]
        k = (len(parts_meters), min(xs), min(ys), max(xs), float(snap_m))
    else:
        k = (0,0,0,0,float(snap_m))
    if k in _COAST_GRAPH_MEMO:
        return _COAST_GRAPH_MEMO[k]

    nodes: Dict[Tuple[int,int], int] = {}          # snapped key -> node_id
    nodes_xy: List[Tuple[float,float]] = []
    edges: List[_Edge] = []
    node_edges: List[List[int]] = []

    def _get_node(pt: Point) -> int:
        key = _snap_key(pt, snap_m)
        nid = nodes.get(key)
        if nid is not None:
            return nid
        nid = len(nodes_xy)
        nodes[key] = nid
        nodes_xy.append((pt.x, pt.y))
        node_edges.append([])
        return nid

    # build nodes+edges
    with _maybe_timer("coast_graph:get_parts_3857", log_timing):
        for i, g in enumerate(parts_meters):
            if g is None or g.is_empty:
                continue
            ls = g
            if simplify_m > 0.0:
                ls = g.simplify(simplify_m, preserve_topology=True)
                if ls.is_empty:
                    continue
            if ls.geom_type == "MultiLineString":
                for sub in ls.geoms:
                    if sub.is_empty: continue
                    p0 = Point(sub.coords[0]); p1 = Point(sub.coords[-1])
                    u = _get_node(p0); v = _get_node(p1)
                    e = _Edge(u=u, v=v, length=float(sub.length), geom=sub,
                            parent_idx=i, parent_feature_id=int(parent_ids[i]))
                    eid = len(edges)
                    edges.append(e)
                    node_edges[u].append(eid)
                    node_edges[v].append(eid)
            elif ls.geom_type == "LineString":
                p0 = Point(ls.coords[0]); p1 = Point(ls.coords[-1])
                u = _get_node(p0); v = _get_node(p1)
                e = _Edge(u=u, v=v, length=float(ls.length), geom=ls,
                        parent_idx=i, parent_feature_id=int(parent_ids[i]))
                eid = len(edges)
                edges.append(e)
                node_edges[u].append(eid)
                node_edges[v].append(eid)

    # components on nodes (union endpoints for each edge)
    with _maybe_timer("coast_graph:components", log_timing):
        find, unite = _dsu_build(len(nodes_xy))
        for e in edges:
            unite(e.u, e.v)
        roots: Dict[int,int] = {}
        node_comp_id = [0]*len(nodes_xy)
        for nid in range(len(nodes_xy)):
            r = find(nid)
            if r not in roots: roots[r] = len(roots)
            node_comp_id[nid] = roots[r]

    # edge spatial index
    with _maybe_timer("coast_graph:edge_strtree", log_timing):
        edge_geoms = [e.geom for e in edges]
        edge_tree = STRtree(edge_geoms)
    edge_idx_by_id = {id(g): i for i, g in enumerate(edge_geoms)}

    if log_timing:
        logger.info("coast_graph summary: nodes=%d edges=%d comps=%d", len(nodes_xy), len(edges), 1+max(node_comp_id) if node_comp_id else 0)

    cg = CoastGraph(
        nodes_xy=nodes_xy,
        edges=edges,
        node_edges=node_edges,
        edge_tree=edge_tree,
        edge_idx_by_id=edge_idx_by_id,
        node_comp_id=node_comp_id,
    )
    _COAST_GRAPH_MEMO[k] = cg
    return cg

def _nearest_edge_id(cg: CoastGraph, pt: Point) -> Optional[int]:
    """Robust nearest edge index from STRtree that may return ints or geoms."""
    near = cg.edge_tree.nearest(pt)
    if near is None:
        return None
    if isinstance(near, numbers.Integral):
        return int(near)
    # shapely returns geometry
    return cg.edge_idx_by_id.get(id(near))

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

def route_along_coast_astar(cg: CoastGraph,
                            A_m: Point,
                            B_m: Point,
                            *,
                            delta_max_m: float = 250_000.0,
                            split_tol_m: float = 5.0,
                            window_m: Optional[float] = None,
                            log_timing: bool = False) -> Optional[LineString]:
    """
    Coast-constrained shortest path from A to B along the coastline graph.
    Returns LineString in metric CRS (WEBM) or None if no path found/over cap.
    """
    with _maybe_timer("astar:total", log_timing):
    # find nearest edges
        ea = _nearest_edge_id(cg, A_m)
        eb = _nearest_edge_id(cg, B_m)
        if ea is None or eb is None:
            return None
        E_a = cg.edges[ea]; E_b = cg.edges[eb]

    # quick component gate based on host edges
    comp_a = cg.node_comp_id[E_a.u]
    comp_b = cg.node_comp_id[E_b.u]
    if comp_a != comp_b:
        return None

    # project endpoints on host edges
    sa = float(E_a.geom.project(A_m))
    sb = float(E_b.geom.project(B_m))

    # snap to existing node if near
    def _snap_or_temp(edge: _Edge, s: float, pt: Point):
        # near u?
        du = _node_dist((cg.nodes_xy[edge.u][0], cg.nodes_xy[edge.u][1]), pt)
        if du <= split_tol_m:
            return edge.u, None  # base node
        # near v?
        dv = _node_dist((cg.nodes_xy[edge.v][0], cg.nodes_xy[edge.v][1]), pt)
        if dv <= split_tol_m:
            return edge.v, None
        # create temp node id and local temp edges description
        temp_id = -(1 + int(round(s)))  # negative ids to mark temp; not used as index
        return temp_id, (edge, s)

    start_node, start_tmp = _snap_or_temp(E_a, sa, A_m)
    goal_node,  goal_tmp  = _snap_or_temp(E_b, sb, B_m)

    # precompute allowed nodes window
    allowed_mask = _allowed_nodes_mask(cg, A_m, B_m, window_m)

    # A*: state is node_id (>=0 for base, <0 for temp)
    def _neighbors(node_id: int):
        # base node
        if node_id >= 0:
            for eid in cg.node_edges[node_id]:
                e = cg.edges[eid]
                other = e.v if e.u == node_id else e.u
                # window prune
                if allowed_mask is not None and not allowed_mask[other]:
                    continue
                yield other, eid, None  # (next_node, via_edge_id, via_tmpinfo)
        else:
            # temp nodes connect only to their host edge endpoints
            edge, s = (start_tmp if node_id == start_node else goal_tmp)
            # two legs: to u and to v
            yield edge.u, None, ("temp", edge, s, edge.u)
            yield edge.v, None, ("temp", edge, s, edge.v)

    def _heuristic(nid: int) -> float:
        # Euclidean to B_m
        if nid >= 0:
            x,y = cg.nodes_xy[nid]
            dx = x - B_m.x; dy = y - B_m.y
            return math.hypot(dx, dy)
        # temp node: use its point
        edge, s = (start_tmp if nid == start_node else goal_tmp)
        p = _pt_on_edge(edge, s)
        return p.distance(B_m)

    # distances and parents
    g = {start_node: 0.0}
    parent: Dict[int, Tuple[int, Optional[int], Optional[Tuple]]] = {}  # nid -> (prev_nid, via_eid, via_tmpinfo)
    openpq: List[Tuple[float,int]] = []
    heapq.heappush(openpq, ( _heuristic(start_node), start_node ))

    max_f = float(delta_max_m) + 1e-6

    visited = set()
    _visits = 0
    _t_start = time.time()
    while openpq:
        f, u = heapq.heappop(openpq)
        if u in visited:
            continue
        visited.add(u)

        if u == goal_node:
            break
        if f > max_f:
            # best possible already exceeds cap
            return None

        gu = g[u]
        _visits += 1
        if _trace_enabled and (_visits % 5000 == 0):
            logger.debug("astar step visits=%d open=%d best_f=%.0f g=%.0f", _visits, len(openpq), f, gu)
        for v, via_eid, via_tmp in _neighbors(u):
            # compute step cost
            if via_eid is not None:
                e = cg.edges[via_eid]
                step = e.length
                # if neighbor is temp (goal temp), cost is handled when expanding temp, so full length here is fine
            else:
                # temp hop: distance along edge between s and endpoint
                kind, e, s, end_node = via_tmp
                if end_node == e.u:
                    step = abs(s - 0.0)
                else:
                    step = abs(e.length - s)
            alt = gu + step
            if alt > delta_max_m + 1e-6:
                continue
            if v not in g or alt < g[v] - 1e-9:
                g[v] = alt
                parent[v] = (u, via_eid, via_tmp)
                fv = alt + _heuristic(v)
                heapq.heappush(openpq, (fv, v))

    if goal_node not in parent and goal_node != start_node:
        return None

    # reconstruct node chain
    chain: List[Tuple[int, Optional[int], Optional[Tuple]]] = []
    cur = goal_node
    while cur != start_node:
        prev, via_eid, via_tmp = parent[cur]
        chain.append((cur, via_eid, via_tmp))
        cur = prev
    chain.reverse()

    # assemble geometry parts
    parts: List[LineString] = []

    # helper to append substring for an edge between two node "positions"
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

    # seed with a tiny stub from A_m to its projection if needed
    # (not strictly necessary; coast-only path suffices)
    prev_node = start_node
    prev_tmp = start_tmp
    for node, via_eid, via_tmp in chain:
        # figure the edge we traversed
        if via_eid is not None:
            e = cg.edges[via_eid]
            n_from = prev_node if prev_node >= 0 else (prev_tmp[3])  # endpoint we stepped from
            n_to   = node      if node      >= 0 else (via_tmp[3])
            _append_edge_segment(e, n_from, n_to,
                                 prev_tmp if prev_node < 0 else None,
                                 via_tmp  if node      < 0 else None)
        else:
            # came via temp hop, that already encoded in parent[v]
            # previous parent step must include a real edge; nothing to add here
            pass
        prev_node = node
        prev_tmp = via_tmp

    if not parts:
        return None

    u = unary_union(parts)
    if isinstance(u, LineString):
        merged = u
    else:
        merged = linemerge(u)

    # ensure a LineString result
    if merged.geom_type == "MultiLineString":
        # pick the longest component
        merged = max(merged.geoms, key=lambda g: g.length)

    if log_timing:
        try:
            logger.info(
                "astar: visits=%d explored=%d chain=%d length_km=%.3f",
                _visits, len(visited), len(chain), float(merged.length)/1000.0
            )
        except Exception:
            pass

    return merged
# ---------- normalize ----------
def _detect_lon_lat(df):
    return _h_detect_lon_lat(df)

def _detect_event_id_col(df, preferred: str | None):
    return _h_detect_event_id_col(df, preferred)

def _detect_height_col(df, preferred: str | None):
    return _h_detect_height_col(df, preferred)

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

def infill_coast_unified(
    snapped_gdf,
    base_segs_gdf,
    coast_lines_gs,
    ray_hits_gdf=None,
    *,
    pad_km: float = 1.0,
    delta_max_km: float = 250.0,
    allow_cross_parent: bool = True,
    # seed length from runup Ht
    seed_height_col: str = "runupHt",
    seed_height_min_m: float = 0.2,
    seed_L_min_km: float = 20.0,
    seed_L_max_km: float = 200.0,
    seed_km_per_meter: float = 60.0,
    seed_height_exp: float = 1.0,
    seed_length_mult: float = 1.0,
    seed_merge_tol_m: float = 2000.0,
    # optional low-Ht and rays as edge amplifiers, re-using rulebased idea
    use_lowht_seed: bool = True,
    use_rays_seeds: bool = True,
    ht_lo_m: float = 0.2,
    ht_hi_m: float = 0.5,
    # logging
    log_decisions: bool = False,
    log_rejects_limit: int = 50,
    log_timing: bool = False,
):
    """
    Unified infill that subsumes both `infill_bridge_rulebased` and `infill_runup_segments`.
    Core: rule-based gap bridging with coverage built from (a) base segments and (b) runup-derived
    seed intervals sized by runup height. Optional ray hits and low-Ht runups act as edge amplifiers.

    Inputs are WGS84. Outputs are WGS84 LineStrings with column `parent_feature_id` when applicable.

    Parameters
    ----------
    pad_km : float
        Padding applied when merging intervals and at segment ends. Replaces `eps_edge_km`.
    delta_max_km : float
        Hard cap for any bridged gap length along the coastline network. Also caps cross-parent bridging.
    allow_cross_parent : bool
        If True, attempt coast-graph A* routing to bridge edge gaps across parents under `delta_max_km`.
    """
    import math
    import time
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    from shapely.strtree import STRtree

    import logging as _logging

    # Local shims to names already defined in runups.py
    from .runups import (
        WGS84, WEBM, ensure_wgs84, to_metric, to_wgs,
        _coast_parts_3857, _substring, _merge_intervals,
        build_coast_graph_from_coast, route_along_coast_astar,
        _maybe_timer as _tm,
    )

    tm = lambda tag: _tm(tag, log_timing)

    # Normalize inputs
    snapped = ensure_wgs84(snapped_gdf) if snapped_gdf is not None else gpd.GeoDataFrame(geometry=[], crs=WGS84)
    base = ensure_wgs84(base_segs_gdf) if base_segs_gdf is not None else gpd.GeoDataFrame(geometry=[], crs=WGS84)
    coast = ensure_wgs84(coast_lines_gs) if coast_lines_gs is not None else gpd.GeoSeries([], crs=WGS84)
    rays = ensure_wgs84(ray_hits_gdf) if (ray_hits_gdf is not None and not getattr(ray_hits_gdf, "empty", True)) else None

    pad_m = float(pad_km) * 1000.0
    delta_max_m = float(delta_max_km) * 1000.0

    logger = logging.getLogger("natural_disasters.infill.unified")
    if not logger.handlers:
        handler = logging.StreamHandler()      # write to stdout
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.propagate = True  # let root handle it too
    # level switches with --debug via root; also set here for safety:
    logger.setLevel(logging.DEBUG if log_timing else logging.INFO)

    @contextmanager
    def _timer(tag: str):
        t0 = time.time(); yield
        if log_timing: logger.info("%s done in %.2fs", tag, time.time() - t0)

    def tm(tag: str):
        return _timer(tag) if log_timing else nullcontext()

    logger.info("unified start pad_km=%.2f delta_max_km=%.1f cross_parent=%s",
                float(pad_km), float(delta_max_km), str(allow_cross_parent))
    # Coast parts and index maps
    with tm("unified:parts_3857"):
        parts_meters, parent_ids = _coast_parts_3857(coast)
    parts_list = list(parts_meters)
    j_by_pid = {int(pid): j for j, pid in enumerate(parent_ids)}
    parent_len_m = {int(pid): float(geom.length) for pid, geom in zip(parent_ids, parts_list)}

    # --- Build seed coverage from runups ---
    seeds_gdf = gpd.GeoDataFrame(columns=["parent_feature_id", "geometry"], geometry="geometry", crs=WGS84)
    if snapped is not None and not snapped.empty:
        with tm("unified:build_seeds"):
            ru = snapped[["geometry", "parent_feature_id"]].copy()
            # ensure station_m present; if missing, project per-parent
            if "station_m" in snapped.columns:
                ru["station_m"] = snapped["station_m"].astype(float)
            else:
                # project points on their parent part
                ru_m = ru.to_crs(WEBM)
                s_vals = np.full(len(ru_m), np.nan, float)
                for idx, (pid, pt_m) in enumerate(zip(ru["parent_feature_id"].to_numpy(), ru_m.geometry.to_numpy())):
                    j = j_by_pid.get(int(pid))
                    if j is None:
                        continue
                    s_vals[idx] = float(parts_list[int(j)].project(pt_m))
                ru["station_m"] = s_vals

            # compute lengths per runup using height
            if seed_height_col in snapped.columns:
                ht = snapped[seed_height_col].astype(float)
                ht = ht.where(ht >= float(seed_height_min_m), float(seed_height_min_m))
                L_km = (ht ** float(seed_height_exp)) * float(seed_km_per_meter) * float(seed_length_mult)
                L_km = L_km.clip(lower=float(seed_L_min_km), upper=float(seed_L_max_km))
            else:
                L_km = pd.Series(float(seed_L_min_km), index=ru.index)
            half_m = (L_km.values * 1000.0) * 0.5

            out_rows = []
            for pid, grp in ru.dropna(subset=["parent_feature_id", "station_m"]).groupby("parent_feature_id", sort=False):
                j = j_by_pid.get(int(pid))
                if j is None:
                    continue
                plen = parent_len_m.get(int(pid), 0.0)
                S = grp["station_m"].to_numpy(float)
                H = half_m[grp.index.to_numpy()]
                ab = np.stack([S - H - pad_m, S + H + pad_m], axis=1)
                # clip to parent length
                ab[:, 0] = np.clip(ab[:, 0], 0.0, plen)
                ab[:, 1] = np.clip(ab[:, 1], 0.0, plen)
                merged = _merge_intervals([(float(x0), float(x1)) for x0, x1 in ab.tolist()], gap_merge_m=float(seed_merge_tol_m))
                line_m = parts_list[int(j)]
                for a, b in merged:
                    if b - a <= 0:
                        continue
                    seg_m = _substring(line_m, float(a), float(b))
                    if seg_m is None or getattr(seg_m, "is_empty", False):
                        continue
                    seg_wgs = gpd.GeoSeries([seg_m], crs=WEBM).to_crs(WGS84).iloc[0]
                    out_rows.append({"parent_feature_id": int(pid), "geometry": seg_wgs})
            if out_rows:
                seeds_gdf = gpd.GeoDataFrame(out_rows, geometry="geometry", crs=WGS84)

    # --- Build coverage from base + seeds ---
    def _intervals_from_lines(lines_gdf):
        if lines_gdf is None or getattr(lines_gdf, "empty", True):
            return {}
        lines_m = to_metric(lines_gdf)
        iv_by_parent = {}
        for _, row in lines_m.iterrows():
            g = row.geometry
            if g is None or g.is_empty:
                continue
            if "parent_feature_id" not in row or pd.isna(row["parent_feature_id"]):
                # No implicit nearest-parent assignment; ignore to avoid bleed-over.
                continue
            pid = int(row["parent_feature_id"])
            j = j_by_pid.get(pid)
            if j is None:
                continue
            a = float(parts_list[j].project(Point(g.coords[0])))
            b = float(parts_list[j].project(Point(g.coords[-1])))
            lo, hi = (a, b) if a <= b else (b, a)
            lo = max(0.0, lo - pad_m)
            hi = min(float(parts_list[j].length), hi + pad_m)
            iv_by_parent.setdefault(pid, []).append((lo, hi))
        for pid in list(iv_by_parent.keys()):
            iv_by_parent[pid] = _merge_intervals(iv_by_parent[pid], gap_merge_m=pad_m)
        return iv_by_parent

    with tm("unified:coverage"):
        base_iv = _intervals_from_lines(base)
        seed_iv = _intervals_from_lines(seeds_gdf)

        cover_by_parent: dict[int, list[tuple[float, float]]] = {}
        parent_keys = set(base_iv.keys()) | set(seed_iv.keys())

        for pid in parent_keys:
            ab: list[tuple[float, float]] = []
            if pid in base_iv:
                ab.extend(base_iv[pid])
            if pid in seed_iv:
                ab.extend(seed_iv[pid])

            ab_list = [(float(a), float(b)) for (a, b) in ab]
            cover_by_parent[int(pid)] = _merge_intervals(ab_list, gap_merge_m=pad_m) if ab_list else []

    # --- Optional amplifiers: add thin edge points inside long gaps ---
    edge_pts = {}
    if use_lowht_seed and snapped is not None and not snapped.empty and "station_m" in snapped.columns:
        mask = snapped[seed_height_col].between(float(ht_lo_m), float(ht_hi_m), inclusive="left") if seed_height_col in snapped.columns else pd.Series(False, index=snapped.index)
        for pid, grp in snapped[mask].groupby("parent_feature_id", sort=False):
            edge_pts[int(pid)] = grp["station_m"].to_numpy(float)
    if use_rays_seeds and rays is not None and not rays.empty and "s_m" in rays.columns and "parent_id" in rays.columns:
        for pid, grp in rays.groupby("parent_id", sort=False):
            arr = grp["s_m"].to_numpy(float)
            edge_pts[int(pid)] = np.concatenate([edge_pts.get(int(pid), np.array([])), arr]) if int(pid) in edge_pts else arr

    def _gaps_from_cover(cover, plen: float):
        cov = list(cover) if cover is not None else []
        if not cov:
            return []
        gaps, a0 = [], 0.0
        for a, b in cov:
            a = float(a); b = float(b)
            if a > a0:
                gaps.append((a0, a))
            a0 = b if b > a0 else a0
        if a0 < plen:
            gaps.append((a0, float(plen)))
        return gaps

    # --- Same-parent bridging within cap ---
    decisions = []
    extra_parts_m = []
    with tm("unified:gaps_same_parent"):
        for pid, cover in cover_by_parent.items():
            plen = parent_len_m.get(int(pid), 0.0)
            if plen <= 0 or cover is None or getattr(cover, "size", 0) == 0:
                # No evidence on this parent → skip entirely.
                continue
            gaps = _gaps_from_cover(cover, plen)
            E = edge_pts.get(int(pid))
            for (a, b) in gaps:
                L = float(b - a)
                if L <= 0:
                    continue
                # Split by edge points if present; all sub-gaps must be ≤ cap
                sub_ok = True
                if E is not None and getattr(E, "size", 0) > 0:
                    mids = E[(E > a + pad_m) & (E < b - pad_m)]
                    if getattr(mids, "size", 0) > 0:
                        mids = np.sort(mids)
                        cuts = [a] + list(mids) + [b]
                        for u, v in zip(cuts[:-1], cuts[1:]):
                            if (v - u) > (delta_max_m + 1e-6):
                                sub_ok = False; break
                if L <= delta_max_m or sub_ok:
                    j = j_by_pid.get(int(pid))
                    if j is None:
                        continue
                    seg_m = _substring(parts_list[int(j)], float(max(0.0, a)), float(min(plen, b)))
                    if seg_m is not None and not seg_m.is_empty:
                        extra_parts_m.append(seg_m)

    # --- Cross-parent bridging via coast graph ---
    if allow_cross_parent:
        with tm("unified:coast_graph"):
            cg = build_coast_graph_from_coast(coast, log_timing=log_timing)

        # parents that actually have some coverage
        relevant_pids = {
            int(pid)
            for pid, cov in cover_by_parent.items()
            if (cov.tolist() if hasattr(cov, "tolist") else list(cov))
        }

        endpoints = []  # (pid, s, point_m)
        for pid in relevant_pids:
            cov = cover_by_parent[pid]
            cov = cov.tolist() if hasattr(cov, "tolist") else list(cov)
            plen = float(parent_len_m.get(pid, 0.0))
            j = j_by_pid.get(pid)
            if j is None or plen <= 0 or not cov:
                continue
            # Only consider coverage boundaries near each parent's ends
            for a, b in cov:
                for s in (float(a), float(b)):
                    if s <= pad_m or (plen - s) <= pad_m:
                        pt_m = parts_list[int(j)].interpolate(s)
                        endpoints.append((pid, s, pt_m))

        if endpoints:
            from shapely.strtree import STRtree
            pts = [pt for (_, _, pt) in endpoints]
            tree = STRtree(pts)
            for i, (pid_i, s_i, pt_i) in enumerate(endpoints):
                try:
                    j_idx = int(tree.nearest(pt_i)) if hasattr(tree, "nearest") else None
                except Exception:
                    j_idx = None
                if j_idx is None or j_idx == i:
                    continue
                pid_j, s_j, pt_j = endpoints[j_idx]
                if pid_i == pid_j:
                    continue

                path_m = route_along_coast_astar(
                    cg, pt_i, pt_j,
                    delta_max_m=delta_max_m,
                    window_m=None,
                    log_timing=log_timing,
                )
                if not path_m or getattr(path_m, "is_empty", False):
                    continue
                if float(path_m.length) > delta_max_m + 1e-6:
                    continue

                extra_parts_m.append(path_m)
                if log_decisions and len(decisions) < log_rejects_limit:
                    decisions.append({
                        "pid_a": int(pid_i), "pid_b": int(pid_j),
                        "len_km": float(path_m.length)/1000.0,
                        "pass": True, "reason": "cross_parent<=cap"
                    })

    # --- Build output ---
    if not extra_parts_m:
        if log_decisions:
            logger.info("unified decisions: tested=%d pass=%d fail=%d", len(decisions), sum(d.get("pass", False) for d in decisions), sum(not d.get("pass", False) for d in decisions))
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)

    out_wgs = gpd.GeoSeries(extra_parts_m, crs=WEBM).to_crs(WGS84)
    out = gpd.GeoDataFrame({"geometry": out_wgs}, geometry="geometry", crs=WGS84)
    # parent_feature_id is undefined for cross-parent routes; keep where determinable by nearest host
    host_pid = []
    for geom_m in gpd.GeoSeries(extra_parts_m, crs=WEBM).geometry:
        rep = geom_m.representative_point()
        dists = [rep.distance(ln) if ln is not None else math.inf for ln in parts_list]
        j = int(np.argmin(dists)) if dists else None
        host_pid.append(int(parent_ids[j]) if j is not None else -1)
    out["parent_feature_id"] = host_pid

    try:
        iv_by_parent = {}
        parts_series_m = gpd.GeoSeries(extra_parts_m, crs=WEBM)
        for geom_m, pid in zip(parts_series_m.geometry, host_pid):
            j = j_by_pid.get(int(pid))
            if j is None or geom_m is None or geom_m.is_empty:
                continue
            a = float(parts_list[j].project(Point(geom_m.coords[0])))
            b = float(parts_list[j].project(Point(geom_m.coords[-1])))
            lo, hi = (a, b) if a <= b else (b, a)
            iv_by_parent.setdefault(int(pid), []).append((lo, hi))

        rows = []
        for pid, iv in iv_by_parent.items():
            merged_iv = _merge_intervals(iv, gap_merge_m=pad_m)
            line_m = parts_list[j_by_pid[int(pid)]]
            for a, b in merged_iv:
                if b <= a:
                    continue
                seg_m = _substring(line_m, a, b)
                if seg_m is None or seg_m.is_empty:
                    continue
                seg_wgs = gpd.GeoSeries([seg_m], crs=WEBM).to_crs(WGS84).iloc[0]
                rows.append({"parent_feature_id": int(pid), "geometry": seg_wgs})

        out_merged = gpd.GeoDataFrame(rows, geometry="geometry", crs=WGS84)
        # Prefer merged emission if non-empty
        if not out_merged.empty:
            out = out_merged
    except Exception as e:
        logger.debug("interval-merge emission skipped: %s", e)

    if log_decisions:
        pass_n = sum(d.get("pass", False) for d in decisions)
        fail_n = sum(not d.get("pass", False) for d in decisions)
        logger.info("unified summary: produced=%d tested=%d pass=%d fail=%d", len(out), len(decisions), pass_n, fail_n)
        logger.info("unified done: rows=%d", 0 if out is None or out.empty else len(out))
    return out
