from __future__ import annotations

import json
import hashlib
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, base
from shapely.strtree import STRtree

WGS84 = "EPSG:4326"
WEBM  = "EPSG:3857"

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
