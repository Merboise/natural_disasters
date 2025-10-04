from __future__ import annotations

import json
import hashlib
import threading
import time
import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, base, Point
from shapely.strtree import STRtree

WGS84 = "EPSG:4326"
WEBM  = "EPSG:3857"

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

def _ensure_wgs84(gs: gpd.GeoSeries | gpd.GeoDataFrame) -> gpd.GeoSeries:
    if isinstance(gs, gpd.GeoDataFrame):
        gs = gs.geometry
    if gs.crs is None:
        return gs.set_crs(WGS84)
    return gs if str(gs.crs) == WGS84 else gs.to_crs(WGS84)


def _explode_lines(gs: gpd.GeoSeries) -> gpd.GeoSeries:
    # Fully explode into LineStrings. Keep a flat index 0..N-1 for part ids.
    exploded = gs.explode(index_parts=False, ignore_index=True)
    # Normalize: keep only LineString parts.
    out = []
    for geom in exploded.values:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            out.append(geom)
        elif geom.geom_type == "MultiLineString":
            out.extend(list(geom.geoms))
    return gpd.GeoSeries(out, crs=WGS84)


def _hash_coast(gs: gpd.GeoSeries, simplify_m: float) -> str:
    # Stable content hash: bounds + count + first/last 100 WKB bytes in 4326.
    b = gs.total_bounds.tobytes()
    n = len(gs)
    h = hashlib.sha256()
    h.update(b)
    h.update(str(n).encode())
    h.update(str(float(simplify_m)).encode())
    for i in (0, n//2 if n else 0, max(0, n-1)):
        if n == 0: break
        try:
            wkb = gs.iloc[i].wkb
        except Exception:
            wkb = b""
        h.update(wkb[:100])
        h.update(wkb[-100:])
    return h.hexdigest()[:16]  

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

def _build_stitched_graph(parts_m, part_ids, *, snap_m: float):
        # grid-bucket endpoints → union-find stitch (faster than N^2 buffers)

        def cell(pt):
            return (math.floor(pt.x / snap_m), math.floor(pt.y / snap_m))

        ends = []        # Points
        owner = []       # (edge_idx, 'u'/'v')
        for i, ln in enumerate(parts_m):
            if not ln or ln.is_empty: continue
            p0, p1 = Point(ln.coords[0]), Point(ln.coords[-1])
            ends.extend([p0, p1])
            owner.extend([(i,'u'), (i,'v')])

        n = len(ends)
        parent = list(range(n))
        def find(x):
            while parent[x]!=x:
                parent[x]=parent[parent[x]]; x=parent[x]
            return x
        def unite(a,b):
            ra, rb = find(a), find(b)
            if ra!=rb: parent[rb]=ra

        buckets = {}
        for idx, p in enumerate(ends):
            buckets.setdefault(cell(p), []).append(idx)

        # neighbor cells 3x3 around each endpoint
        for idx, p in enumerate(ends):
            cx, cy = cell(p)
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    lst = buckets.get((cx+dx, cy+dy))
                    if not lst: continue
                    for j in lst:
                        if j == idx: continue
                        if p.distance(ends[j]) <= snap_m:
                            unite(idx, j)

        # canonical node ids
        canon = [find(i) for i in range(n)]
        node_map = {}
        nodes_xy = []
        for i, ci in enumerate(canon):
            if ci not in node_map:
                node_map[ci] = len(nodes_xy)
                nodes_xy.append((ends[i].x, ends[i].y))

        # edges with stitched endpoints
        edges = []
        node_edges = [[] for _ in range(len(nodes_xy))]
        for i, ln in enumerate(parts_m):
            if not ln or ln.is_empty: continue
            u = node_map[canon[2*i+0]]
            v = node_map[canon[2*i+1]]
            eidx = len(edges)
            edges.append(_Edge(u=u, v=v, length=float(ln.length), geom=ln,
                            parent_idx=i, parent_feature_id=int(part_ids[i])))
            node_edges[u].append(eidx); node_edges[v].append(eidx)

        # components
        find2, unite2 = _dsu_build(len(nodes_xy))
        for e in edges: unite2(e.u, e.v)
        comp = [find2(i) for i in range(len(nodes_xy))]

        # edge STRtree + id map
        edge_geoms = [e.geom for e in edges]
        edge_tree = STRtree(edge_geoms)
        edge_idx_by_id = {id(g): i for i, g in enumerate(edge_geoms)}

        return CoastGraph(nodes_xy=nodes_xy, edges=edges, node_edges=node_edges,
                        edge_tree=edge_tree, edge_idx_by_id=edge_idx_by_id,
                        node_comp_id=comp)

def set_default_simplify_m(self, val: float):
    with self._lock:
        self._default_simplify_m = float(val)

@dataclass(frozen=True)
class CoastBase3857:
    key: str
    simplify_m: float
    parts_wgs84_count: int
    parts_m: List[base.BaseGeometry]          
    lengths_m: np.ndarray                     
    strtree_m: STRtree                        
    bounds_wgs84: Tuple[float, float, float, float]  
    built_ms: float


class TsunamiMemCache:
    """Process-local coastline cache. Read-only once built. Thread-safe gets."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._store: Dict[Tuple[str, float], CoastBase3857] = {}
        self._stats = {"builds": 0, "hits": 0, "misses": 0}
        self._idmaps: Dict[Tuple[str, float], Dict[int, int]] = {}
        self._graphs = {}
        self._default_simplify_m = 0.0

    # Public API --------------------------------------------------------------
    def get_parts_3857_and_tree(self, coast_wgs: gpd.GeoSeries | gpd.GeoDataFrame,
                                *, simplify_m: float = 0.0) -> Tuple[List[base.BaseGeometry], List[int], STRtree, str]:
        """Return (parts_m, parent_ids, strtree, key). Parent_ids are simple 0..N-1 part ids."""
        cb = self._get_or_build(coast_wgs, simplify_m=simplify_m)
        parent_ids = list(range(len(cb.parts_m)))
        return cb.parts_m, parent_ids, cb.strtree_m, cb.key

    def get_parts_wgs_exploded(self, coast_wgs: gpd.GeoSeries | gpd.GeoDataFrame,
                               *, simplify_m: float = 0.0) -> Tuple[gpd.GeoSeries, List[int], str]:
        """Return exploded WGS84 LineStrings, their ids 0..N-1, and cache key."""
        cb = self._get_or_build(coast_wgs, simplify_m=simplify_m)
        # Reconstruct parts_wgs84 by reprojecting back from metric list to WGS84 for callers that need it.
        parts_wgs = gpd.GeoSeries(cb.parts_m, crs=WEBM).to_crs(WGS84)
        parent_ids = list(range(len(parts_wgs)))
        return parts_wgs, parent_ids, cb.key

    def cache_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._stats)

    def write_meta_json(self, path: str) -> None:
        """Write a small JSON index for introspection. Trees/geoms are not serialized."""
        with self._lock:
            meta = {
                "entries": [
                    {
                        "key": key[0],
                        "simplify_m": key[1],
                        "parts_count": len(cb.parts_m),
                        "bounds_wgs84": cb.bounds_wgs84,
                        "built_ms": cb.built_ms,
                    }
                    for key, cb in self._store.items()
                ],
                "stats": self._stats,
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # Internals ---------------------------------------------------------------
    def _get_or_build(self, coast_wgs: gpd.GeoSeries | gpd.GeoDataFrame,
                      *, simplify_m: float) -> CoastBase3857:
        # Normalize and explode WGS84
        gs = _ensure_wgs84(coast_wgs)
        exploded = _explode_lines(gs)
        if simplify_m and simplify_m > 0:
            # Simplify only if it keeps topology for lines (preserve topology=True is for polygons;
            # for lines, use standard simplify).
            try:
                exploded = exploded.simplify(simplify_m, preserve_topology=False)
                # Remove empties that could appear after simplify
                exploded = exploded[~exploded.is_empty].reset_index(drop=True)
            except Exception:
                pass

        key = _hash_coast(exploded, simplify_m)
        k = (key, float(simplify_m))

        with self._lock:
            if k in self._store:
                self._stats["hits"] += 1
                return self._store[k]
            self._stats["misses"] += 1

        t0 = time.perf_counter()
        # Project to metric and build artifacts
        parts_m = exploded.to_crs(WEBM).tolist()
        lengths = np.fromiter((g.length for g in parts_m), dtype=float, count=len(parts_m))
        tree = STRtree(parts_m) if len(parts_m) else STRtree([])  # empty tree allowed
        built_ms = (time.perf_counter() - t0) * 1000.0

        cb = CoastBase3857(
            key=key,
            simplify_m=float(simplify_m),
            parts_wgs84_count=int(len(exploded)),
            parts_m=parts_m,
            lengths_m=lengths,
            strtree_m=tree,
            bounds_wgs84=tuple(exploded.total_bounds),
            built_ms=built_ms,
        )
        with self._lock:
            self._store[k] = cb
            self._stats["builds"] += 1
        return cb

    def _ensure_idmap(self, cb: CoastBase3857) -> Dict[int, int]:
        k = (cb.key, float(cb.simplify_m))
        with self._lock:
            m = self._idmaps.get(k)
            if m is None:
                m = {id(g): i for i, g in enumerate(cb.parts_m)}
                self._idmaps[k] = m
            return m

    def get_tree_and_index_map(self, coast_wgs, *, simplify_m: float = 0.0):
        """Return (strtree_m, idmap, key). Geometries must be in metric (WEBM) for tree.nearest()."""
        cb = self._get_or_build(coast_wgs, simplify_m=simplify_m)
        idmap = self._ensure_idmap(cb)
        return cb.strtree_m, idmap, cb.key

    def get_parts_tree_index(self, coast_wgs, *, simplify_m: float = 0.0):
        """Return (parts_m, part_ids, strtree_m, idmap, key)."""
        cb = self._get_or_build(coast_wgs, simplify_m=simplify_m)
        idmap = self._ensure_idmap(cb)
        part_ids = list(range(len(cb.parts_m)))
        return cb.parts_m, part_ids, cb.strtree_m, idmap, cb.key

    def nearest_part_idx_metric(self, coast_wgs, geom_m, *, simplify_m: float = 0.0) -> int:
        """Nearest exploded-part index for a METRIC geometry."""
        cb = self._get_or_build(coast_wgs, simplify_m=simplify_m)
        idmap = self._ensure_idmap(cb)
        near = cb.strtree_m.nearest(geom_m)
        idx = idmap.get(id(near))
        if idx is not None:
            return idx
        # rare fallback if identity changed
        import numpy as _np
        d = [_np.inf if g is None else geom_m.distance(g) for g in cb.parts_m]
        return int(_np.argmin(d))
    
    def get_coast_graph(self, coast_wgs, *, simplify_m: float | None = None, snap_m: float = 250.0):
        sm = self._default_simplify_m if simplify_m is None else float(simplify_m)
        cb = self._get_or_build(coast_wgs, simplify_m=sm)
        k = (cb.key, sm, float(snap_m))
        with self._lock:
            g = self._graphs.get(k)
            if g is not None:
                return g
            part_ids = list(range(len(cb.parts_m)))  # <-- fix
            g = _build_stitched_graph(cb.parts_m, part_ids, snap_m=float(snap_m))  # <-- call module fn
            self._graphs[k] = g
            return g

    
_MC = TsunamiMemCache()  # singleton instance