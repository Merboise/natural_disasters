# runups.py
from __future__ import annotations
import math, numbers, logging, time, heapq
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
    same_parent_only: bool = False,
    max_bridge_gap_km: float = 500.0,
    min_combined_coverage: float = 0.6,
    opts: Optional[InfillConfig] = None,
) -> gpd.GeoDataFrame:
    """Fill gaps using lower-height evidence. Returns extra segments to union with base.
    This version fixes parent_feature_id vs j index confusion and undefined variables.
    """
    logger.info(
        ">>> entered infill_runup_segments: snapped=%d segments=%d eps_km=%.2f min_samples=%d majority=%.2f same_parent=%s bridge_km=%.1f min_cov=%.2f",
        0 if snapped_gdf is None else len(snapped_gdf),
        0 if base_segs_gdf is None else len(base_segs_gdf),
        eps_km, min_samples, majority_threshold, same_parent_only,
        max_bridge_gap_km, min_combined_coverage,
    )

    # Options
    _allow_zero_cover_bridge = bool(getattr(opts, 'bridge_allow_zero_cover', False))
    _bridge_one_side = bool(getattr(opts, 'bridge_one_side', False))
    _accept_singleton = bool(getattr(opts, 'accept_singleton', False))
    _reject_log_limit = int(getattr(opts, 'log_rejects_limit', 50) or 50)
    _seed_without_base = bool(getattr(opts, 'seed_without_base', False))
    _seed_min_m = float(getattr(opts, 'seed_min_km', 1.0) or 1.0) * 1000.0

    logger.debug(
        "infill params effective: eps_km=%.2f min_samples=%d majority=%.2f bridge_max_km=%.1f min_cov=%.2f allow_zero_cover_bridge=%s one_side=%s accept_singleton=%s log_rejects_limit=%d seed_without_base=%s seed_min_km=%.2f",
        eps_km, min_samples, majority_threshold, max_bridge_gap_km, min_combined_coverage,
        _allow_zero_cover_bridge, _bridge_one_side, _accept_singleton,
        _reject_log_limit, _seed_without_base, (_seed_min_m/1000.0),
    )

    # Heartbeat
    _start_ts = time.time()
    _stop_evt = threading.Event()
    def _hb():
        while not _stop_evt.wait(30.0):
            logger.debug("infill_runup_segments heartbeat: running for %.1fs", time.time() - _start_ts)
    _hb_thread = threading.Thread(target=_hb, name="infill_runup_segments_heartbeat", daemon=True)
    _hb_thread.start()

    try:
        snapped = ensure_wgs84(snapped_gdf) if snapped_gdf is not None else gpd.GeoDataFrame(geometry=[], crs=WGS84)
        base = ensure_wgs84(base_segs_gdf) if base_segs_gdf is not None else gpd.GeoDataFrame(geometry=[], crs=WGS84)
        coast_lines = ensure_wgs84(coast_lines_gs) if coast_lines_gs is not None else gpd.GeoSeries([], crs=WGS84)

        eps_m = float(eps_km) * 1000.0
        bridge_m = float(max_bridge_gap_km) * 1000.0

        # Coast parts and mappings
        parts_meters, parent_ids = _coast_parts_3857(coast_lines)
        parts_list = list(parts_meters)
        if not parts_list:
            return gpd.GeoDataFrame(geometry=[], crs=WGS84)
        try:
            _p, _pid, tree, _k = _MC.get_parts_3857_and_tree(gpd.GeoDataFrame(geometry=gpd.GeoSeries(coast_lines), crs=WGS84))
        except Exception:
            tree = STRtree(parts_list)
        idx_by_geom = {id(g): j for j, g in enumerate(parts_list)}  # geom id -> j
        j_by_pid = {int(pid): j for j, pid in enumerate(parent_ids)}  # pid -> j
        pid_by_j = {j: int(pid) for j, pid in enumerate(parent_ids)}   # j -> pid
        len_by_j = np.asarray([float(g.length) for g in parts_list], dtype=float)

        # Base intervals keyed by j
        base_intervals_j: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        if not base.empty:
            base_m = to_metric(base)
            for _, row in base_m.iterrows():
                j = None
                if "parent_feature_id" in row.index and row["parent_feature_id"] is not None:
                    try:
                        j = j_by_pid.get(int(row["parent_feature_id"]))
                    except Exception:
                        j = None
                if j is None:
                    rep = row.geometry.representative_point()
                    ret = tree.nearest(rep)
                    j = int(ret) if isinstance(ret, numbers.Integral) else idx_by_geom.get(id(ret))
                if j is None:
                    continue
                line = parts_list[int(j)]
                coords = list(row.geometry.coords)
                svals = [line.project(Point(*c)) for c in coords]
                s0, s1 = float(min(svals)), float(max(svals))
                base_intervals_j[int(j)].append((s0, s1))
            # merge per j
            for jj in list(base_intervals_j.keys()):
                base_intervals_j[jj] = _merge_intervals(base_intervals_j[jj], gap_merge_m=1.0)
        logger.info(
            "infill: prepared base_intervals for %d parents from %d base segments",
            len([k for k, v in base_intervals_j.items() if v]),
            0 if base is None else len(base),
        )

        # Active parent js to evaluate
        active_from_snapped_j = set()
        if not snapped.empty and {"parent_feature_id", "station_m"}.issubset(snapped.columns):
            for pid in snapped["parent_feature_id"].astype(int).unique().tolist():
                jj = j_by_pid.get(int(pid))
                if jj is not None:
                    active_from_snapped_j.add(int(jj))
        active_from_base_j = set(j for j, v in base_intervals_j.items() if v)
        parents_j = sorted(active_from_snapped_j.union(active_from_base_j))

        extra_intervals_j: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        reject_logs = 0
        reason_counts = Counter()

        # Cluster low-band points per j then decide acceptability relative to base on same j
        for jj in parents_j:
            pid = pid_by_j[int(jj)]
            sub = snapped[
                (snapped["parent_feature_id"].astype(int) == pid)
                & (snapped.get("runupHt", 0.0) >= ht_lo_m)
                & (snapped.get("runupHt", 0.0) < ht_hi_m)
            ] if not snapped.empty else snapped
            if sub is None or sub.empty:
                continue
            s_vals = np.array(sub["station_m"].astype(float).tolist(), dtype=float)
            if s_vals.size == 0:
                continue
            s_vals.sort()
            # DBSCAN-like 1D clustering by gap>eps
            breaks = np.where(np.diff(s_vals) > eps_m)[0]
            clusters = np.split(s_vals, breaks + 1)

            base_ivals = base_intervals_j.get(int(jj), [])
            for cl in clusters:
                if len(cl) < int(min_samples):
                    if not (_accept_singleton and len(cl) == 1):
                        if reject_logs < _reject_log_limit:
                            logger.debug("infill reject pid=%d reason=min_samples len=%d min=%d", pid, len(cl), int(min_samples))
                            reject_logs += 1
                        reason_counts['min_samples'] += 1
                        continue
                s0, s1 = float(cl[0]), float(cl[-1])

                # coverage fraction by base on same j
                cover = 0.0
                overlap_any = False
                contains_point = False
                if base_ivals:
                    covered = []
                    for a, b in base_ivals:
                        if not (b < s0 or a > s1):
                            overlap_any = True
                        if a <= s0 <= b:
                            contains_point = True
                        lo, hi = max(a, s0), min(b, s1)
                        if hi >= lo:
                            covered.append((lo, max(hi, lo + 1e-6)))
                    if covered and (s1 - s0) > 0:
                        merged = _merge_intervals(covered, gap_merge_m=1.0)
                        cover_len = sum(b - a for a, b in merged)
                        cover = cover_len / max(1.0, (s1 - s0))
                accept = cover >= float(majority_threshold)

                # Bridging acceptance on same parent only
                if same_parent_only and base_ivals:
                    left = [b for b in base_ivals if b[1] <= s0]
                    right = [b for b in base_ivals if b[0] >= s1]
                    gap = float('inf')
                    if left and right:
                        L_end = max(b[1] for b in left)
                        R_start = min(b[0] for b in right)
                        gap = max(0.0, float(R_start) - float(L_end))
                        if gap <= bridge_m:
                            if cover >= float(min_combined_coverage) or _allow_zero_cover_bridge:
                                accept = True
                    if (not accept) and _bridge_one_side and (left or right):
                        gap_left = float('inf')
                        gap_right = float('inf')
                        if left:
                            L_end = max(b[1] for b in left)
                            gap_left = max(0.0, float(s0) - float(L_end))
                        if right:
                            R_start = min(b[0] for b in right)
                            gap_right = max(0.0, float(R_start) - float(s1))
                        one_gap = min(gap_left, gap_right)
                        if np.isfinite(one_gap) and one_gap <= bridge_m:
                            if cover >= float(min_combined_coverage) or _allow_zero_cover_bridge:
                                accept = True

                # Degenerate cluster handling
                span = float(s1 - s0)
                if not accept:
                    if span <= 1e-6 and (contains_point or overlap_any):
                        accept = True
                    elif overlap_any and span > 1e-6:
                        accept = True

                # Nearest-base proximity for singletons when no left/right
                if not accept and span <= 1e-6 and (not base_ivals):
                    # if allowed to seed without base, expand
                    if _seed_without_base:
                        mid = 0.5 * (s0 + s1)
                        a = mid - 0.5 * max(_seed_min_m, span)
                        b = mid + 0.5 * max(_seed_min_m, span)
                        extra_intervals_j[int(jj)].append((float(a), float(b)))
                        continue

                if accept:
                    extra_intervals_j[int(jj)].append((float(s0), float(s1)))
                else:
                    if reject_logs < _reject_log_limit:
                        reasons = []
                        if cover < float(majority_threshold):
                            reasons.append("majority_fail")
                        if same_parent_only:
                            if not base_ivals:
                                reasons.append("no_base_for_parent")
                            else:
                                reasons.append("bridge_failed")
                        logger.debug(
                            "infill reject pid=%d s0=%.0f s1=%.0f len_km=%.2f cover=%.2f reasons=%s",
                            pid, s0, s1, (span/1000.0), cover, ",".join(reasons) or "unknown",
                        )
                        reject_logs += 1
                        for r in reasons:
                            reason_counts[r] += 1

        # ---------------- Bridging pass (additive) ----------------
        b_eps_m = float(getattr(opts, 'eps_km', 8.0)) * 1000.0
        delta_std_m = float(getattr(opts, 'delta_std_km', 40.0)) * 1000.0
        delta_long_m = float(getattr(opts, 'delta_long_km', 100.0)) * 1000.0
        wA = float(getattr(opts, 'wA', 1.0))
        wL = float(getattr(opts, 'wL', 0.7))
        p_exp = float(getattr(opts, 'p_exp', 1.2))
        tau_base = float(getattr(opts, 'tau_base', 0.2))
        tau_step = float(getattr(opts, 'tau_step', 0.15))
        pad_m = min(b_eps_m, 0.5 * delta_std_m)

        # Low-band s-values per j
        svals_by_j: Dict[int, np.ndarray] = {}
        if not snapped.empty:
            low = snapped[(snapped.get("runupHt", 0.0) >= ht_lo_m) & (snapped.get("runupHt", 0.0) < ht_hi_m)]
            if not low.empty and {"parent_feature_id", "station_m"}.issubset(low.columns):
                for pid, grp in low.groupby(low["parent_feature_id"].astype(int)):
                    jj = j_by_pid.get(int(pid))
                    if jj is None:
                        continue
                    sarr = np.asarray(grp["station_m"].astype(float).tolist(), dtype=float)
                    sarr.sort()
                    svals_by_j[int(jj)] = sarr

        def _edge_hits(sarr: np.ndarray, b0: float, a1: float, tol_m: float) -> int:
            if sarr is None or sarr.size == 0:
                return 0
            A = 0
            j = int(np.searchsorted(sarr, b0))
            near = []
            if j < len(sarr): near.append(abs(sarr[j] - b0))
            if j > 0: near.append(abs(sarr[j-1] - b0))
            if near and min(near) <= tol_m: A += 1
            k = int(np.searchsorted(sarr, a1))
            near2 = []
            if k < len(sarr): near2.append(abs(sarr[k] - a1))
            if k > 0: near2.append(abs(sarr[k-1] - a1))
            if near2 and min(near2) <= tol_m: A += 1
            return min(2, A)

        gaps_checked = 0
        bridges_added = 0
        pass_counts = [0, 0, 0]
        unconditional_count = 0
        seed_bridge_count = 0

        for jj, ivals in base_intervals_j.items():
            if not ivals or len(ivals) < 2:
                continue
            ivals_sorted = sorted(ivals)
            parent_line = parts_list[int(jj)]
            parent_len = float(len_by_j[int(jj)])
            sarr = svals_by_j.get(int(jj), np.array([], dtype=float))
            for k in range(len(ivals_sorted) - 1):
                a0, b0 = ivals_sorted[k]
                a1, b1 = ivals_sorted[k + 1]
                L = max(0.0, float(a1) - float(b0))
                if L <= 0:
                    continue
                if L > delta_long_m:
                    continue
                A = _edge_hits(sarr, float(b0), float(a1), b_eps_m)
                Ln = (L / delta_long_m) ** p_exp
                S = wA * A - wL * Ln
                accepted = False

                unc_km = getattr(opts, 'bridge_unconditional_under_km', None)
                if unc_km is not None:
                    try:
                        if L <= float(unc_km) * 1000.0:
                            accepted = True
                            unconditional_count += 1
                    except Exception:
                        pass

                if (not accepted) and (unc_km is not None) and (sarr is not None) and (sarr.size > 0):
                    inside = sarr[(sarr > float(b0)) & (sarr < float(a1))]
                    if inside.size > 0:
                        diffs = np.maximum(inside - float(b0), float(a1) - inside)
                        jbest = int(np.argmin(diffs))
                        s = float(inside[jbest])
                        left = s - float(b0)
                        right = float(a1) - s
                        if left <= float(unc_km) * 1000.0 and right <= float(unc_km) * 1000.0:
                            aL = max(0.0, min(parent_len, float(b0) - pad_m))
                            bL = max(0.0, min(parent_len, s + pad_m))
                            aR = max(0.0, min(parent_len, s - pad_m))
                            bR = max(0.0, min(parent_len, float(a1) + pad_m))
                            if bL > aL:
                                extra_intervals_j[int(jj)].append((aL, bL))
                                bridges_added += 1
                            if bR > aR:
                                extra_intervals_j[int(jj)].append((aR, bR))
                                bridges_added += 1
                            accepted = True
                            seed_bridge_count += 1

                if (not accepted) and (L <= delta_std_m) and (S >= tau_base):
                    accepted = True; pass_counts[0] += 1
                if (not accepted) and (L <= 0.5 * (delta_std_m + delta_long_m)) and (S >= (tau_base + tau_step)):
                    accepted = True; pass_counts[1] += 1
                if (not accepted) and (L <= delta_long_m) and (S >= (tau_base + 2.0 * tau_step)):
                    accepted = True; pass_counts[2] += 1

                gaps_checked += 1
                if accepted:
                    a = max(0.0, min(parent_len, float(b0) - pad_m))
                    b = max(0.0, min(parent_len, float(a1) + pad_m))
                    if b > a:
                        extra_intervals_j[int(jj)].append((a, b))
                        bridges_added += 1

        # Emit geoms for extra_intervals_j keyed by j
        geoms: List[object] = []
        pids: List[int] = []
        for jj, ivals in extra_intervals_j.items():
            if not ivals:
                continue
            ivals = _merge_intervals(ivals, gap_merge_m=1.0)
            line = parts_list[int(jj)]
            L = float(len_by_j[int(jj)])
            pid = pid_by_j[int(jj)]
            for a, b in ivals:
                a = max(0.0, min(L, float(a)))
                b = max(0.0, min(L, float(b)))
                if b <= a:
                    continue
                seg = _substring(line, float(a), float(b))
                if seg is not None and not seg.is_empty:
                    geoms.append(seg)
                    pids.append(int(pid))

        gdf = gpd.GeoDataFrame(
            {"parent_feature_id": pids, "geometry": geoms},
            geometry="geometry",
            crs=WEBM,
        ).to_crs(WGS84)
        try:
            logger.info(
                "infill summary: produced=%d rejects_logged=%d reason_counts=%s",
                0 if gdf is None or gdf.empty else len(gdf), _reject_log_limit, dict(reason_counts),
            )
        except Exception:
            pass
        return gdf
    finally:
        _stop_evt.set()
        _hb_thread.join(timeout=1.0)
        logger.info(
            "<<< exited infill_runup_segments in %.1fs",
            time.time() - _start_ts,
        )


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

# ---------- rule-based conservative infill bridging ----------

def infill_bridge_rulebased(
    snapped_gdf: gpd.GeoDataFrame,
    base_segs_gdf: gpd.GeoDataFrame,
    coast_lines_gs: gpd.GeoSeries,
    ray_hits_gdf: Optional[gpd.GeoDataFrame] = None,
    *,
    delta_max_km: float = 250.0,
    eps_edge_km: float = 10.0,
    ht_lo_m: float = 0.2,
    ht_hi_m: float = 0.5,
    same_parent_only: bool = False,
    allow_cross_parent: bool = True,
    use_rays_seeds: bool = True,
    use_lowht_seed: bool = True,
    log_decisions: bool = False,
    log_rejects_limit: int = 50,
    log_timing: bool = False,
) -> gpd.GeoDataFrame:
    """Rule-based conservative infill that bridges base→base gaps on the same parent.
    - Unconditionally bridges gaps ≤ delta_max_km.
    - Uses low-Ht runups and ray hits as amplifiers to split longer gaps; sub-gaps must also be ≤ delta_max_km.
    - Rays act only as amplifiers; no magnitude weighting here.
    - If same_parent_only=False, a conservative cross-parent bridge is attempted by nearest endpoints, capped by delta_max_km.
    Adds per-gap decision logs: pass/fail and reason.
    """
    t0 = time.time()
    logger = logging.getLogger(__name__).getChild("runups")
    tm = (lambda tag: _maybe_timer(tag, log_timing))

    # Normalize inputs
    snapped = ensure_wgs84(snapped_gdf) if snapped_gdf is not None else gpd.GeoDataFrame(geometry=[], crs=WGS84)
    base = ensure_wgs84(base_segs_gdf) if base_segs_gdf is not None else gpd.GeoDataFrame(geometry=[], crs=WGS84)
    coast = ensure_wgs84(coast_lines_gs) if coast_lines_gs is not None else gpd.GeoSeries([], crs=WGS84)
    rays = ensure_wgs84(ray_hits_gdf) if (ray_hits_gdf is not None and not ray_hits_gdf.empty) else None

    # Coast parts and parent lines in 3857
    with tm("bridge:parts_3857"):
        parts_meters, parent_ids = _coast_parts_3857(coast)
    parts_list = list(parts_meters)
    j_by_pid = {int(pid): j for j, pid in enumerate(parent_ids)}
    len_by_j = np.asarray([float(g.length) for g in parts_list], dtype=float)
    from pyproj import Transformer
    _tf_3857_to_4326 = Transformer.from_crs(WEBM, WGS84, always_xy=True).transform

    t_graph = time.time()
    coast_graph = build_coast_graph_from_coast(coast, log_timing=log_timing)  # coast_lines is WGS84 GeoSeries
    if log_timing:
        try:
            logger.info("coast_graph: nodes=%d edges=%d build=%.2fs",
                        coast_graph.number_of_nodes(),
                        coast_graph.number_of_edges(),
                        time.time() - t_graph)
        except Exception:
            logger.info("coast_graph built in %.2fs", time.time() - t_graph)

    # Build base intervals per parent (meters alongshore)
    base_intervals_j: Dict[int, List[Tuple[float, float]]] = {}
    with tm("bridge:base_to_metric"):
        base_m = to_metric(base)
    for _, row in base_m.iterrows():
        # Prefer inherent parent id if present to avoid drift
        j = None
        if "parent_feature_id" in row.index:
            try:
                j = j_by_pid.get(int(row["parent_feature_id"]))
            except Exception:
                j = None
        if j is None:
            # Fallback to nearest parent by representative point
            rep = row.geometry.representative_point()
            dists = [rep.distance(ln) for ln in parts_list]
            j = int(np.argmin(dists)) if dists else None
        if j is None:
            continue
        line = parts_list[int(j)]
        coords = list(row.geometry.coords)
        svals = [line.project(Point(*c)) for c in coords]
        s0, s1 = float(min(svals)), float(max(svals))
        base_intervals_j.setdefault(int(j), []).append((s0, s1))

    # Merge base intervals
    for pid in list(base_intervals_j.keys()):
        base_intervals_j[pid] = _h_merge_intervals(base_intervals_j[pid], gap_merge_m=1.0)

    # Collect amplifiers per parent: low-Ht runups and ray hits, as station arrays
    ampl_by_parent: Dict[int, np.ndarray] = {}
    if use_lowht_seed and snapped is not None and not snapped.empty:
        if ("runupHt" in snapped.columns) and ("parent_feature_id" in snapped.columns) and ("station_m" in snapped.columns):
            low = snapped[(snapped["runupHt"] >= float(ht_lo_m)) & (snapped["runupHt"] < float(ht_hi_m))]
            for pid, grp in low.groupby(low["parent_feature_id"].astype(int)):
                arr = np.array(grp["station_m"].astype(float).tolist(), dtype=float)
                if arr.size:
                    arr.sort()
                    ampl_by_parent[int(pid)] = arr
    if use_rays_seeds and rays is not None and not rays.empty:
        if ("parent_id" in rays.columns) and ("s_m" in rays.columns):
            for pid, grp in rays.groupby(rays["parent_id"].astype(int)):
                arr = np.array(grp["s_m"].astype(float).tolist(), dtype=float)
                if arr.size:
                    arr.sort()
                    ampl_by_parent.setdefault(int(pid), np.array([], dtype=float))
                    ampl_by_parent[int(pid)] = np.sort(np.concatenate([ampl_by_parent[int(pid)], arr]))

    delta_max_m = float(delta_max_km) * 1000.0
    eps_m = float(eps_edge_km) * 1000.0

    # Bridge intervals to add
    extra_by_parent: Dict[int, List[Tuple[float, float]]] = {}
    total_gaps = 0
    bridged_gaps = 0
    added_km = 0.0
    base_km = sum((b - a) for iv in base_intervals_j.values() for (a, b) in iv) / 1000.0

    # Per-gap decision records
    decisions: List[Dict[str, object]] = []

    # Endpoint caches for optional cross-parent bridging
    right_pts_m: list[tuple[int, float, object, object]] = []  # (pid, s_right, point_m, point_wgs)
    left_pts_m: list[tuple[int, float, object, object]] = []   # (pid, s_left, point_m, point_wgs)

    _last_hb = time.time()
    for pid, ivals in base_intervals_j.items():
        if _trace_enabled and (time.time() - _last_hb) > 30.0:
            logger.debug("bridge hb: pid=%d gaps_so_far=%d bridged=%d", int(pid), total_gaps, bridged_gaps)
            _last_hb = time.time()
        if not ivals or len(ivals) < 2:
            # Still record endpoints
            j = int(pid)
            line = parts_list[j]
            for a, b in sorted(ivals):
                pm_r = line.interpolate(float(b)); pm_l = line.interpolate(float(a))
                xr, yr = _tf_3857_to_4326(pm_r.x, pm_r.y); xl, yl = _tf_3857_to_4326(pm_l.x, pm_l.y)
                right_pts_m.append((j, float(b), pm_r, Point(xr, yr)))
                left_pts_m.append((j, float(a), pm_l, Point(xl, yl)))
            continue

        ivals_sorted = sorted(ivals)
        S = ampl_by_parent.get(int(pid), None)

        # Iterate gaps between consecutive base intervals on this parent
        for k in range(len(ivals_sorted) - 1):
            a0, b0 = ivals_sorted[k]
            a1, b1 = ivals_sorted[k + 1]
            L = float(a1) - float(b0)
            if L <= 0:
                continue
            total_gaps += 1
            if log_timing and (total_gaps % 200 == 0):
                logger.info("progress: checked %d gaps (bridged=%d) in %.2fs",
                            total_gaps, bridged_gaps, time.time() - t0)

            rec = {"pid": int(pid), "b0_m": float(b0), "a1_m": float(a1), "L_km": L / 1000.0, "pass": False, "reason": ""}

            # Unconditional bridge for gaps within cap
            if L <= delta_max_m:
                extra_by_parent.setdefault(int(pid), []).append((float(b0), float(a1)))
                bridged_gaps += 1
                added_km += L / 1000.0
                rec["pass"] = True
                rec["reason"] = "unconditional"
                decisions.append(rec)
                continue

            # Try amplifiers near edges
            made_any = False
            near_left, near_right = [], []

            if S is not None and S.size:
                # near left edge
                j = int(np.searchsorted(S, float(b0)))
                candidates = []
                if j < len(S): candidates.append(S[j])
                if j > 0: candidates.append(S[j - 1])
                near_left = [float(s) for s in candidates if abs(float(s) - float(b0)) <= eps_m]

                # near right edge
                j2 = int(np.searchsorted(S, float(a1)))
                candidates2 = []
                if j2 < len(S): candidates2.append(S[j2])
                if j2 > 0: candidates2.append(S[j2 - 1])
                near_right = [float(s) for s in candidates2 if abs(float(s) - float(a1)) <= eps_m]

            # Build sub-bridges for anchors found
            for s in near_left:
                Lsub = float(s) - float(b0)
                if Lsub > 0 and Lsub <= delta_max_m:
                    extra_by_parent.setdefault(int(pid), []).append((float(b0), float(s)))
                    bridged_gaps += 1
                    added_km += Lsub / 1000.0
                    made_any = True

            for s in near_right:
                Lsub = float(a1) - float(s)
                if Lsub > 0 and Lsub <= delta_max_m:
                    extra_by_parent.setdefault(int(pid), []).append((float(s), float(a1)))
                    bridged_gaps += 1
                    added_km += Lsub / 1000.0
                    made_any = True

            if made_any:
                rec["pass"] = True
                if near_left and near_right:
                    rec["reason"] = "amp_both_edges_ok"
                elif near_left:
                    rec["reason"] = "amp_left_ok"
                elif near_right:
                    rec["reason"] = "amp_right_ok"
            else:
                if not S is None and S.size and (near_left or near_right):
                    rec["reason"] = "amp_near_edge_but_subgap_too_long"
                elif S is None or S.size == 0:
                    rec["reason"] = "no_amplifiers_available"
                else:
                    rec["reason"] = "no_amplifiers_near_edges"
            decisions.append(rec)

        # Record endpoints for cross-parent bridging
        j = int(pid)
        line = parts_list[j]
        for a, b in ivals_sorted:
            pm_r = line.interpolate(float(b)); pm_l = line.interpolate(float(a))
            xr, yr = _tf_3857_to_4326(pm_r.x, pm_r.y); xl, yl = _tf_3857_to_4326(pm_l.x, pm_l.y)
            right_pts_m.append((j, float(b), pm_r, Point(xr, yr)))
            left_pts_m.append((j, float(a), pm_l, Point(xl, yl)))

    # Merge intervals per parent and cut substrings
    geoms = []
    pids = []
    for j_key, ivals in extra_by_parent.items():
        ivals = _h_merge_intervals(ivals, gap_merge_m=1.0)
        j = int(j_key)
        line = parts_list[j]
        with tm("bridge:cut_substrings"):
            L = float(len_by_j[j])
            for a, b in ivals:
                a = max(0.0, min(L, float(a)))
                b = max(0.0, min(L, float(b)))
                if b <= a:
                    continue
                seg = _substring(line, float(a), float(b))
                if seg and not seg.is_empty:
                    geoms.append(seg)
                    pids.append(int(parent_ids[j]))

        # Optional cross-parent bridges (conservative straight chord with coastal cap)
    t_cp = time.time()
    cp_bridges = 0
    right_pts_m = right_pts_m if 'right_pts_m' in locals() else []
    left_pts_m  = left_pts_m  if 'left_pts_m'  in locals() else []
    try:
        if allow_cross_parent and right_pts_m and left_pts_m:
            import numbers
            from shapely.strtree import STRtree

            left_geoms_m = [pm for (_j_l, _s, pm, _pw) in left_pts_m]
            left_tree = STRtree(left_geoms_m)
            left_idx_by_id = {id(g): i for i, g in enumerate(left_geoms_m)}
            left_xy = np.asarray([(p.x, p.y) for p in left_geoms_m], dtype=float)

            delta_max_m = float(delta_max_km * 1000.0)
            near_gate_lo = 0.9 * delta_max_m  
            K = 10

            # Build the coastal graph once
            coast_graph = coast_graph if 'coast_graph' in locals() else build_coast_graph_from_coast(coast)

            # graph indices (CoastGraph, not networkx)
            comp_label = coast_graph.node_comp_id  # list[int], index by node id

            # nearest-node cache over CoastGraph node coordinates
            node_points = [Point(x, y) for (x, y) in coast_graph.nodes_xy]
            nodes_tree = STRtree(node_points)
            node_ids = list(range(len(node_points)))
            idx_by_geom_node = {id(g): i for i, g in enumerate(node_points)}
            _nn_cache = {}

            def nearest_node(pt: Point) -> int | None:
                key = (round(pt.x, 1), round(pt.y, 1))
                nid = _nn_cache.get(key)
                if nid is not None:
                    return nid
                g = nodes_tree.nearest(pt)
                idx = idx_by_geom_node.get(id(g))
                nid = node_ids[int(idx)] if idx is not None else None
                _nn_cache[key] = nid
                return nid

            def nearest_node(pt):
                key = (round(pt.x, 1), round(pt.y, 1))
                nid = _nn_cache.get(key)
                if nid is not None:
                    return nid
                g = nodes_tree.nearest(pt)
                idx = idx_by_geom_node.get(id(g))
                nid = node_ids[int(idx)] if idx is not None else None
                _nn_cache[key] = nid
                return nid

            seen = set()
            if _trace_enabled():
                # how many endpoints per parent
                cnt_r = Counter([int(j) for (j,_,_,_) in right_pts_m])
                cnt_l = Counter([int(j) for (j,_,_,_) in left_pts_m])
                parents_r = set(cnt_r.keys()); parents_l = set(cnt_l.keys())
                crossable = len(parents_r.intersection(parents_l))  # should be > 1 to be interesting
                logger.debug("cp: parents_right=%d parents_left=%d intersect=%d", len(parents_r), len(parents_l), crossable)

            for j_r, s_r, pm_r, pw_r in right_pts_m:
                xr, yr = pm_r.x, pm_r.y

                # box query
                cand = left_tree.query(box(xr - delta_max_m, yr - delta_max_m, xr + delta_max_m, yr + delta_max_m))
                if hasattr(cand, "tolist"): cand = cand.tolist()
                cand = list(cand or [])
                if _trace_enabled():
                    logger.debug("cp[%d]: box=%d", int(j_r), len(cand))
                if not cand:
                    continue

                # indices
                if cand and isinstance(cand[0], numbers.Integral):
                    idxs = np.asarray([int(i) for i in cand], dtype=int)
                else:
                    idxs = np.asarray([left_idx_by_id.get(id(g)) for g in cand if left_idx_by_id.get(id(g)) is not None], dtype=int)
                if idxs.size == 0:
                    if _trace_enabled(): logger.debug("cp[%d]: idxs=0", int(j_r))
                    continue

                # distances and radius cut
                dx = left_xy[idxs, 0] - xr; dy = left_xy[idxs, 1] - yr
                d = np.hypot(dx, dy)
                keep = d <= delta_max_m
                if not np.any(keep):
                    if _trace_enabled(): logger.debug("cp[%d]: within_cap=0", int(j_r))
                    continue
                idxs = idxs[keep]; d = d[keep]
                if _trace_enabled():
                    logger.debug("cp[%d]: within_cap=%d d_min=%.0f d_med=%.0f", int(j_r), idxs.size, float(d.min()), float(np.median(d)))

                # remove same-parent here and report
                same_parent = (np.asarray([left_pts_m[int(i)][0] for i in idxs], dtype=int) == int(j_r))
                if np.any(same_parent):
                    idxs = idxs[~same_parent]; d = d[~same_parent]
                if _trace_enabled():
                    logger.debug("cp[%d]: after_same_parent=%d", int(j_r), idxs.size)
                if idxs.size == 0:
                    continue

                # component precheck in bulk
                comp_ok = []
                for li in idxs.tolist():
                    j_l, s_l, pm_l, _ = left_pts_m[int(li)]
                    u = nearest_node(pm_r); v = nearest_node(pm_l)
                    if (u is None) or (v is None):
                        continue
                    try:
                        if comp_label[u] != comp_label[v]:
                            continue
                    except Exception:
                        pass
                    comp_ok.append(li)
                if _trace_enabled():
                    logger.debug("cp[%d]: after_comp=%d", int(j_r), len(comp_ok))
                if not comp_ok:
                    continue

                idxs = np.asarray(comp_ok, dtype=int)
                d = np.asarray([float(np.hypot(left_xy[i,0]-xr, left_xy[i,1]-yr)) for i in idxs], dtype=float)

                # density-aware budget
                area = math.pi * (delta_max_m ** 2)
                rho = (len(idxs) / area) if area > 0 else 1.0
                Kloc = int(np.clip(math.ceil(8.0 / max(rho, 1e-6)), 6, 24))
                if len(idxs) > Kloc:
                    part = np.argpartition(d, Kloc - 1)[:Kloc]
                    idxs, d = idxs[part], d[part]
                order = np.argsort(d); idxs, d = idxs[order], d[order]
                if _trace_enabled():
                    logger.debug("cp[%d]: final_cand=%d Kloc=%d", int(j_r), len(idxs), Kloc)

                for li, d_webm in zip(idxs.tolist(), d.tolist()):
                    j_l, s_l, pm_l, pw_l = left_pts_m[int(li)]
                    key = (int(j_r), int(j_l), float(s_r), float(s_l))
                    if key in seen:
                        continue
                    seen.add(key)

                    win = float(min(delta_max_m, max(d_webm * 1.5, 2000.0)))
                    with _trace("cp:route_try", jr=int(j_r), jl=int(j_l), d=int(d_webm)):
                        path_m = route_along_coast_astar(
                            coast_graph, pm_r, pm_l,
                            delta_max_m=delta_max_m, window_m=win, log_timing=True
                        )
                    if path_m is None:
                        continue

                    geoms.append(path_m)
                    pids.append(int(parent_ids[int(j_r)]))
                    cp_bridges += 1
                    break


        logger.info(
            "cross_parent summary: endpoints_right=%d endpoints_left=%d cp_bridges=%d",
            len(right_pts_m), len(left_pts_m), cp_bridges
        )
    except Exception:
        logger.exception("cross_parent bridging failed")

    out = gpd.GeoDataFrame(
        {
            "parent_feature_id": pids, 
            "geometry": geoms
        },
        geometry="geometry", 
        crs=WEBM
    ).to_crs(WGS84) if geoms else gpd.GeoDataFrame(geometry=[], crs=WGS84)
    # Decision logs
    try:
        fails = [d for d in decisions if not d.get("pass")]
        pass_n = len(decisions) - len(fails)
        logger.info("infill decisions: tested=%d pass=%d fail=%d", len(decisions), pass_n, len(fails))
        if log_decisions and fails:
            limit = int(max(0, log_rejects_limit))
            for d in fails[:limit]:
                logger.debug("infill FAIL pid=%d L_km=%.2f reason=%s", int(d["pid"]), float(d["L_km"]), str(d["reason"]))
    except Exception:
        # Do not fail the pipeline on logging issues
        pass

    logger.info(
        "infill_bridge_rulebased: parents=%d total_gaps=%d bridged_gaps=%d added_km=%.1f base_km=%.1f elapsed=%.1fs",
        len(base_intervals_j), total_gaps, bridged_gaps, added_km, base_km, time.time() - t0
    )
    return out