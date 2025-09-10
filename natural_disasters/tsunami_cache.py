
# -*- coding: utf-8 -*-
"""
tsunami_cache.py — unified persistent cache for coast + stationing.
"""
from __future__ import annotations

import hashlib, json, os, time, logging, shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import geopandas as gpd


DEFAULT_SIZE_THRESHOLD_MB = 250
LOCK_TIMEOUT_S = 300


def _utcnow():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _short_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mkdir_clean(p: Path) -> None:
    if p.exists(): shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class CoastCache:
    coast_wgs: gpd.GeoDataFrame
    coast_3857: gpd.GeoDataFrame
    parent_centers_3857: np.ndarray        # (P,2) float32
    coords_by_parent: Dict[int, np.ndarray]
    s_by_parent: Dict[int, np.ndarray]
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def coast_m(self): return self.coast_3857
    @property
    def coast_metric(self): return self.coast_3857
    @property
    def step_m(self) -> int: return int(self.meta.get("step_m", 250))
    @step_m.setter
    def step_m(self, v: float) -> None:
        self.meta["step_m"] = int(v)


def build_cache(*,
    coast_lines_wgs: gpd.GeoDataFrame,
    step_m: int = 250,
    simplify_m: float = 0.0,
    cache_dir: str = "cache",
    cache_mode: str = "auto",
    overwrite: bool = False,
) -> CoastCache:
    cache_root = Path(cache_dir).expanduser().absolute() / "tsunami"
    cache_root.mkdir(parents=True, exist_ok=True)

    key = _short_hash(json.dumps({"step_m": step_m, "simplify_m": simplify_m}, sort_keys=True))
    subdir = cache_root / key

    if overwrite and subdir.exists():
        logging.info("[cache] overwrite requested; deleting %s", subdir)
        shutil.rmtree(subdir, ignore_errors=True)

    if cache_mode in ("auto", "ro") and subdir.exists():
        try:
            return _load_cache_from_dir(subdir)
        except Exception as e:
            logging.warning("[cache] failed to reuse: %s", e)
            if cache_mode == "ro":
                raise

    if cache_mode == "off":
        return _build_cache_in_memory(coast_lines_wgs, step_m, simplify_m)

    tmp = subdir.with_suffix(".tmp")
    _mkdir_clean(tmp)

    coast_wgs = coast_lines_wgs.copy()
    if simplify_m > 0:
        coast_wgs["geometry"] = coast_wgs.geometry.simplify(float(simplify_m), preserve_topology=False)
    coast_3857 = coast_wgs.to_crs(3857)

    coords_by_parent: Dict[int, np.ndarray] = {}
    s_by_parent: Dict[int, np.ndarray] = {}
    centers = []
    for pid, Lm in enumerate(coast_3857.geometry):
        length = float(Lm.length)
        s = np.arange(0.0, length + step_m * 0.5, step_m, dtype=np.float32)
        if len(s) < 2:
            s = np.linspace(0.0, max(1.0, length), 2, dtype=np.float32)
        pts = [Lm.interpolate(float(d)) for d in s]
        xy = np.array([[p.x, p.y] for p in pts], dtype=np.float32)
        coords_by_parent[pid] = xy
        s_by_parent[pid] = s.astype(np.float32)
        c = Lm.centroid
        centers.append((np.float32(c.x), np.float32(c.y)) if not c.is_empty else (np.nan, np.nan))

    parent_centers_3857 = np.asarray(centers, np.float32)

    coast_wgs.to_file(tmp / "coast_wgs.gpkg", layer="coast_wgs", driver="GPKG")
    coast_3857.to_file(tmp / "coast_3857.gpkg", layer="coast_3857", driver="GPKG")
    np.save(tmp / "parent_centers_3857.npy", parent_centers_3857)
    np.save(tmp / "coords_by_parent.npy", np.array(list(coords_by_parent.values()), dtype=object), allow_pickle=True)
    np.save(tmp / "s_by_parent.npy", np.array(list(s_by_parent.values()), dtype=object), allow_pickle=True)
    _write_json(tmp / "meta.json", {"step_m": int(step_m), "simplify_m": float(simplify_m), "created_utc": _utcnow()})

    if subdir.exists():
        shutil.rmtree(subdir, ignore_errors=True)
    os.replace(str(tmp), str(subdir))

    return _load_cache_from_dir(subdir)


def _load_cache_from_dir(subdir: Path) -> CoastCache:
    coast_wgs = gpd.read_file(subdir / "coast_wgs.gpkg", layer="coast_wgs")
    coast_3857 = gpd.read_file(subdir / "coast_3857.gpkg", layer="coast_3857")
    parent_centers_3857 = np.load(subdir / "parent_centers_3857.npy")
    coords_arr = np.load(subdir / "coords_by_parent.npy", allow_pickle=True)
    s_arr = np.load(subdir / "s_by_parent.npy", allow_pickle=True)
    coords_by_parent = {i: coords_arr[i] for i in range(len(coords_arr))}
    s_by_parent = {i: s_arr[i] for i in range(len(s_arr))}
    meta = _read_json(subdir / "meta.json")
    return CoastCache(coast_wgs=coast_wgs, coast_3857=coast_3857,
                      parent_centers_3857=parent_centers_3857,
                      coords_by_parent=coords_by_parent, s_by_parent=s_by_parent,
                      meta=meta)


def _build_cache_in_memory(coast_lines_wgs: gpd.GeoDataFrame, step_m: int, simplify_m: float) -> CoastCache:
    coast_wgs = coast_lines_wgs.copy()
    if simplify_m > 0:
        coast_wgs["geometry"] = coast_wgs.geometry.simplify(float(simplify_m), preserve_topology=False)
    coast_3857 = coast_wgs.to_crs(3857)

    coords_by_parent = {}
    s_by_parent = {}
    centers = []
    for pid, Lm in enumerate(coast_3857.geometry):
        length = float(Lm.length)
        s = np.arange(0.0, length + step_m * 0.5, step_m, dtype=np.float32)
        if len(s) < 2:
            s = np.linspace(0.0, max(1.0, length), 2, dtype=np.float32)
        pts = [Lm.interpolate(float(d)) for d in s]
        xy = np.array([[p.x, p.y] for p in pts], dtype=np.float32)
        coords_by_parent[pid] = xy
        s_by_parent[pid] = s.astype(np.float32)
        c = Lm.centroid
        centers.append((np.float32(c.x), np.float32(c.y)) if not c.is_empty else (np.nan, np.nan))

    parent_centers_3857 = np.asarray(centers, np.float32)
    return CoastCache(coast_wgs=coast_wgs, coast_3857=coast_3857,
                      parent_centers_3857=parent_centers_3857,
                      coords_by_parent=coords_by_parent, s_by_parent=s_by_parent,
                      meta={"step_m": int(step_m), "simplify_m": float(simplify_m)})
