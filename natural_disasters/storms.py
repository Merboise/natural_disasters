# -*- coding: utf-8 -*-
"""
storms_refactored.py — unified cyclone & tornado pipeline

Goals of this refactor:
- Keep core geometry and blending logic intact while simplifying structure
- Remove dead/duplicated helpers and consolidate logging/timing
- Centralize all toggles in one Config dataclass (and expose via CLI)
- Make antimeridian handling explicit and traceable
- Ensure all text I/O is utf-8
- Provide quiet GPKG writer with consistent log prefixes

Notes:
- This file assumes you have a local ".helpers" module exposing the same
  utilities you already used (AGENCY_PREF, QUADS, write_gpkg, etc.).
- Where previous code had multiple overlapping helpers, these were merged.
- Debug/trace toggles can be flipped from CLI without touching code.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPoint, box, mapping
from shapely.ops import unary_union, split, transform
from shapely import make_valid

# -- Silence noisy GDAL OSR future warning unless explicitly desired
import warnings as _warnings
_warnings.filterwarnings(
    "ignore",
    message=r"Neither osr.UseExceptions\(\) nor osr.DontUseExceptions\(\) has been explicitly called",
    category=FutureWarning,
    module="osgeo.osr",
)

# -- Ensure GDAL/PROJ env is configured BEFORE importing geopandas/pyogrio (already imported)
try:
    from .bootstrap_gdal import verify_gdal_ready as _verify_gdal_ready
    _GDAL_DATA, _PROJ_LIB = _verify_gdal_ready()
    logging.getLogger(__name__).info(
        f"[Setup] GDAL/PROJ ok (GDAL_DATA={_GDAL_DATA}, PROJ_LIB={_PROJ_LIB})"
    )
    try:
        from osgeo import osr as _osr
        _osr.DontUseExceptions()
    except Exception:
        pass
except Exception as _e:
    _GDAL_DATA = _PROJ_LIB = None
    logging.getLogger(__name__).warning(f"[Setup] GDAL/PROJ setup warning: {_e}")

# ---- Local project helpers (expected to be present as before)
from .helpers import (
    AGENCY_PREF, RAD_TIERS, QUADS, NM_TO_KM,
    CLIMO_R34_NM, SSHS_SCALE_K,
    clean_radius, first_positive, build_circle, extract_polygons_only,
    diagnose_geom, log_polygon_failure, write_gpkg, spatial_temporal_filter,
)

# =============================================================================
# Config & Logging
# =============================================================================

@dataclass
class Config:
    # Debug/trace
    debug_all: bool = False
    debug_sids: set[str] = field(default_factory=set)
    trace_poly: bool = False
    verbose_seam: bool = False
    fail_dump_dir: Optional[Path] = None

    # Fallback magnitudes
    fallback_union_nm: float = 25.0
    fallback_linebuf_m: float = 5000.0

    # Quadrant weighting
    min_weighted_cov: float = 0.15
    uncertainty_scale: float = 1.0

    # Cleaning
    simplify_deg: float = 0.02
    openclose_km: float = 5.0

    # Behavior
    climo_fallback: bool = True
    switch_margin: float = 0.08

    # Output
    out_gpkg: Optional[Path] = None
    out_layer: str = "cyclones"
    out_csv: Optional[Path] = None

    # Filters
    year_ge: Optional[int] = None
    min_sshs: Optional[int] = None
    oecd_only: bool = False

    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """UTF-8 logging to console (and optional file)."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    sh = logging.StreamHandler(stream=sys.stdout)
    try:
        # ensure utf-8 on Windows + py>=3.9
        sh.stream.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


# Global registry for section timings
_section_times: Dict[str, float] = {}
_pipeline_start: float = time.time()


def _log_header(hazard: str, message: str) -> None:
    logging.info(f"[{hazard}] {message}")
    _section_times[hazard] = time.time()


def _log_processed(hazard: str, processed: int, skipped: int, make_separator: bool = True) -> None:
    elapsed = time.time() - _section_times.get(hazard, time.time())
    logging.info(f"[{hazard}] Processed={processed:,}  Skipped={skipped:,}")
    logging.info(f"[{hazard}] Created {processed:,} records")
    logging.info(f"[{hazard}] Elapsed: {elapsed:.2f}s")
    if make_separator:
        _log_separator()


def _log_separator() -> None:
    try:
        width = shutil.get_terminal_size().columns
    except Exception:
        width = 80
    logging.info("-" * width)


def _log_pipeline_done() -> None:
    total = time.time() - _pipeline_start
    logging.info(f"[Pipeline] Completed in {total:.2f}s")


# =============================================================================
# Utility geometry helpers (kept, trimmed, or consolidated)
# =============================================================================

AGENCY_WEIGHTS: Dict[str, float] = {
    "USA": 1.00, "REUNION": 0.95, "BOM": 0.95, "TOKYO": 0.90, "CMA": 0.90,
    "HKO": 0.85, "KMA": 0.85, "NADI": 0.85,
}

_SMOOTH_FRACTION_OF_RADIUS = 0.35
_DENSIFY_MAX_SEG_KM = 20.0


def _dbg_enabled_for(cfg: Config, sid: Optional[str]) -> bool:
    return bool(cfg.debug_all or (sid and sid in cfg.debug_sids))


def _ring_lon_jumps(coords: list[tuple[float, float]]) -> float:
    if not coords or len(coords) < 2:
        return 0.0
    mx = 0.0
    for i in range(1, len(coords)):
        dx = float(coords[i][0]) - float(coords[i - 1][0])
        dx_alt = ((dx + 180.0) % 360.0) - 180.0
        mx = max(mx, abs(dx), abs(dx_alt))
    return mx


def _log_geom_diag(tag: str, geom, frame_off: float = 0.0, sid: str | None = None, *, enabled: bool = False) -> None:
    if not enabled or geom is None:
        return
    try:
        gg = _wrap_to_180(geom)
        (minx, miny, maxx, maxy) = gg.bounds
        lon_span = float(maxx - minx)
        lat_span = float(maxy - miny)
        # count vertices across rings
        worst_jump = 0.0
        nverts = 0
        geoms = getattr(gg, "geoms", None) or [gg]
        for p in geoms:
            if p.geom_type == "Polygon":
                rings = [p.exterior] + list(p.interiors)
            elif p.geom_type == "MultiPolygon":
                rings = []
                for sub in p.geoms:
                    rings.extend([sub.exterior] + list(sub.interiors))
            else:
                rings = [getattr(p, "boundary", p)]
            for r in rings:
                coords = list(getattr(r, "coords", []))
                nverts += len(coords)
                worst_jump = max(worst_jump, _ring_lon_jumps(coords))
        logging.info(
            f"[Diag] {tag} | sid={sid or 'NA'} frame_off={frame_off:+.0f} "
            f"bounds=({minx:.3f},{miny:.3f},{maxx:.3f},{maxy:.3f}) "
            f"span=({lon_span:.2f}° x {lat_span:.2f}°) nverts={nverts} max|Δlon|={worst_jump:.2f}°"
        )
    except Exception:
        pass


def _unwrap_lon_seq(xs: list[float]) -> list[float]:
    if not xs:
        return []
    out = [float(xs[0])]
    for x in xs[1:]:
        x = float(x)
        prev = out[-1]
        while x - prev > 180.0:
            x -= 360.0
        while x - prev < -180.0:
            x += 360.0
        out.append(x)
    return out


def _wrap_lon(x: float) -> float:
    return ((float(x) + 180.0) % 360.0) - 180.0


def _shift_lon_geom(geom, xoff_deg: float):
    try:
        return transform(lambda x, y, z=None: (x + xoff_deg, y), geom)
    except Exception:
        return geom


def _wrap_to_180(geom):
    if geom is None or geom.is_empty:
        return geom
    try:
        from shapely.ops import snap
        g0360 = transform(lambda x, y, z=None: ((x % 360.0 + 360.0) % 360.0, y), geom)
        seam = LineString([(180.0, -90.0), (180.0, 90.0)])
        g0360 = snap(g0360, seam, 1e-9)
        strip = box(180.0 - 1e-6, -90.0, 180.0 + 1e-6, 90.0)
        g0360 = g0360.difference(strip)
        try:
            parts = split(g0360, seam)
        except Exception:
            left = box(0.0, -90.0, 180.0, 90.0)
            right = box(180.0, -90.0, 360.0, 90.0)
            parts = [g0360.intersection(left), g0360.intersection(right)]
        out_parts = []
        for p in getattr(parts, "geoms", [parts]):
            if p is None or p.is_empty:
                continue
            cx = p.representative_point().x
            if cx >= 180.0:
                p = transform(lambda x, y, z=None: (x - 360.0, y), p)
            out_parts.append(p)
        out = unary_union(out_parts)
        out = transform(lambda x, y, z=None: (((x + 180.0) % 360.0) - 180.0, y), out)
        try:
            out = make_valid(out)
        except Exception:
            out = out.buffer(0)
        return extract_polygons_only(out)
    except Exception:
        return transform(lambda x, y, z=None: (((x + 180.0) % 360.0) - 180.0, y), geom)


def _bbox_lon_width_points(pts: list[tuple[float, float]]) -> float:
    if not pts:
        return float("inf")
    xs = [p[0] for p in pts]
    return float(max(xs) - min(xs))


def _frame_offset_for_track(pts: list[tuple[float, float]]) -> float:
    if not pts:
        return 0.0
    lons = [((float(x) + 180.0) % 360.0) - 180.0 for (x, _) in pts]
    lons.sort()
    if len(lons) == 1:
        seam = ((lons[0] + 180.0) % 360.0) - 180.0
        return 180.0 - seam
    gaps, mids = [], []
    for i in range(1, len(lons)):
        gap = lons[i] - lons[i - 1]
        mid = 0.5 * (lons[i] + lons[i - 1])
        gaps.append(gap)
        mids.append(mid)
    wrap_gap = (lons[0] + 360.0) - lons[-1]
    wrap_mid = 0.5 * ((lons[0] + 360.0) + lons[-1])
    gaps.append(wrap_gap)
    mids.append(wrap_mid)
    k = int(np.argmax(gaps))
    seam = ((mids[k] + 180.0) % 360.0) - 180.0
    off = 180.0 - seam
    return off


def _proj_local_fns(center_lon: float, center_lat: float):
    try:
        from pyproj import CRS, Transformer
        center_lat = float(max(-85.0, min(85.0, center_lat)))
        lon0 = _wrap_lon(center_lon)
        laea = CRS.from_proj4(
            f"+proj=laea +lat_0={center_lat:.8f} +lon_0={lon0:.8f} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
        wgs84 = CRS.from_epsg(4326)
        fwd = Transformer.from_crs(wgs84, laea, always_xy=True).transform
        inv = Transformer.from_crs(laea, wgs84, always_xy=True).transform
        return fwd, inv
    except Exception:
        # Affine, metric fallback
        mx, my = _local_scale(center_lat)
        def fwd(x, y, z=None):
            return ((x - center_lon) * mx, (y - center_lat) * my)
        def inv(X, Y, Z=None):
            return (X / mx + center_lon, Y / my + center_lat)
        return fwd, inv


def _local_scale(lat: float) -> tuple[float, float]:
    lat = float(max(-85.0, min(85.0, lat)))
    # meters per degree (very rough)
    mx = 111_320.0 * math.cos(math.radians(lat))
    my = 110_540.0
    return mx, my


def _circle_laea(lat: float, lon: float, radius_nm: float, frame_off: float = 0.0):
    try:
        if not np.isfinite(radius_nm) or radius_nm <= 0:
            return None
        lon_c = float(lon) + float(frame_off)
        lat_c = float(lat)
        fwd, inv = _proj_local_fns(lon_c, lat_c)
        r_m = max(500.0, float(radius_nm) * NM_TO_KM * 1000.0)
        center_ll = Point(lon_c, lat_c)
        center_m = transform(fwd, center_ll)
        circ_m = center_m.buffer(r_m, resolution=64)
        if circ_m.is_empty:
            return None
        circ_ll = transform(inv, circ_m)
        return _shift_lon_geom(circ_ll, -frame_off)
    except Exception:
        return None


def _clean_poly_local(geom, lat_hint: float, simplify_m: float, openclose_m: float):
    if geom is None or geom.is_empty:
        return None
    try:
        fwd, inv = _proj_local_fns(geom.representative_point().x, lat_hint)
        g_m = transform(fwd, geom)
        s = max(300.0, openclose_m)
        g_m = g_m.buffer(s).buffer(-s).simplify(max(50.0, simplify_m), preserve_topology=True)
        g_ll = transform(inv, g_m)
        try:
            g_ll = make_valid(g_ll)
        except Exception:
            g_ll = g_ll.buffer(0)
        return extract_polygons_only(g_ll)
    except Exception:
        return extract_polygons_only(geom)


def _ensure_min_area(geom, lat_hint: float, min_area_km2: float, nudge_half_width_m: float = 3000.0):
    if geom is None or geom.is_empty:
        return None
    try:
        fwd, inv = _proj_local_fns(geom.representative_point().x, lat_hint)
        g_m = transform(fwd, geom)
        area_km2 = float(g_m.area) / 1_000_000.0
        if area_km2 >= float(min_area_km2):
            return geom
        nudge = max(500.0, float(nudge_half_width_m))
        g2_m = g_m.buffer(nudge).buffer(-nudge)
        if g2_m.is_empty:
            g2_m = g_m.buffer(nudge)
        return transform(inv, g2_m)
    except Exception:
        return geom


def _densify_track_points(pts: list[tuple[float, float]], max_seg_km: float = _DENSIFY_MAX_SEG_KM) -> list[tuple[float, float]]:
    if not pts:
        return []
    out = [pts[0]]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        dx = (x1 - x0) * (111.0 * math.cos(math.radians((y0 + y1) / 2)))
        dy = (y1 - y0) * 111.0
        dist_km = math.hypot(dx, dy)
        nseg = max(1, int(math.ceil(dist_km / max_seg_km)))
        for i in range(1, nseg + 1):
            t = i / nseg
            out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return out


# =============================================================================
# Cyclone-specific helpers (blending & selection)
# =============================================================================

def sshs_from_knots(v) -> float:
    try:
        v = float(v)
    except Exception:
        return np.nan
    if not np.isfinite(v):
        return np.nan
    if v < 34:
        return -1
    if v < 64:
        return 0
    if v < 83:
        return 1
    if v < 96:
        return 2
    if v < 113:
        return 3
    if v < 137:
        return 4
    return 5


def choose_latlon(row: pd.Series, allow_blend_when_master_missing: bool = True):
    def _as_num(v):
        try:
            v = float(str(v).strip())
            return v if np.isfinite(v) else np.nan
        except Exception:
            return np.nan
    lat_m = _as_num(row.get("LAT"))
    lon_m = _as_num(row.get("LON"))
    if pd.notna(lat_m) and pd.notna(lon_m):
        return float(lat_m), float(lon_m), "MASTER"
    agency_pairs_ordered = [
        ("USA", "USA_LAT", "USA_LON"),
        ("BOM", "BOM_LAT", "BOM_LON"),
        ("REUNION", "REUNION_LAT", "REUNION_LON"),
        ("TOKYO", "TOKYO_LAT", "TOKYO_LON"),
        ("CMA", "CMA_LAT", "CMA_LON"),
        ("HKO", "HKO_LAT", "HKO_LON"),
        ("KMA", "KMA_LAT", "KMA_LON"),
        ("NADI", "NADI_LAT", "NADI_LON"),
        ("DS824", "DS824_LAT", "DS824_LON"),
        ("TD9636", "TD9636_LAT", "TD9636_LON"),
        ("TD9635", "TD9635_LAT", "TD9635_LON"),
        ("NEWDELHI", "NEWDELHI_LAT", "NEWDELHI_LON"),
        ("WELLINGTON", "WELLINGTON_LAT", "WELLINGTON_LON"),
        ("NEUMANN", "NEUMANN_LAT", "NEUMANN_LON"),
        ("MLC", "MLC_LAT", "MLC_LON"),
    ]
    pts = []
    names = []
    for name, plat, plon in agency_pairs_ordered:
        lat = _as_num(row.get(plat))
        lon = _as_num(row.get(plon))
        if pd.notna(lat) and pd.notna(lon):
            wt = float(AGENCY_WEIGHTS.get(name, 0.75))
            pts.append((float(lat), float(lon), name, wt))
            names.append(name)
    if not pts:
        return np.nan, np.nan, None
    if len(pts) == 1 or not allow_blend_when_master_missing:
        lat, lon, name, _ = pts[0]
        return lat, lon, name
    lats = np.array([p[0] for p in pts], dtype=float)
    lons = np.array([p[1] for p in pts], dtype=float)
    wts = np.array([p[3] for p in pts], dtype=float)
    lats = np.clip(lats, -90.0, 90.0)
    ref = np.deg2rad(lons[0])
    angles = np.deg2rad(lons)
    delta = (angles - ref + np.pi) % (2 * np.pi) - np.pi
    angles_adj = ref + delta
    x = np.sum(wts * np.cos(angles_adj))
    y = np.sum(wts * np.sin(angles_adj))
    lon_mean = np.rad2deg(np.arctan2(y, x))
    lon_mean = ((lon_mean + 180.0) % 360.0) - 180.0
    lat_mean = float(np.sum(wts * lats) / np.sum(wts))
    src_tag = f"BLEND({','.join(names)})"
    return lat_mean, lon_mean, src_tag


def blend_family_row(row: pd.Series, family: str) -> float:
    vals, wts = [], []
    for ag, wt in AGENCY_WEIGHTS.items():
        col = f"{ag}_{family}"
        if col in row.index:
            v = pd.to_numeric(row[col], errors="coerce")
            if pd.notna(v):
                vals.append(float(v))
                wts.append(float(wt))
    if vals:
        return float(np.average(vals, weights=wts))
    if family in row.index:
        v = pd.to_numeric(row[family], errors="coerce")
        return float(v) if pd.notna(v) else np.nan
    return np.nan


def add_blended_sshs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["__WIND_BLEND"] = df.apply(lambda r: blend_family_row(r, "WIND"), axis=1)
    df["__SSHS_BLEND"] = pd.to_numeric(df["__WIND_BLEND"], errors="coerce").map(sshs_from_knots)
    need = df["__SSHS_BLEND"].isna()
    if "USA_SSHS" in df.columns:
        df.loc[need, "__SSHS_BLEND"] = pd.to_numeric(df.loc[need, "USA_SSHS"], errors="coerce")
    need = df["__SSHS_BLEND"].isna()
    if "WIND" in df.columns:
        df.loc[need, "__SSHS_BLEND"] = pd.to_numeric(df.loc[need, "WIND"], errors="coerce").map(sshs_from_knots)
    return df


# =============================================================================
# Per-SID polygon assembly (core)
# =============================================================================

def _smooth_series_3pt(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, copy=True)
    out = np.full_like(vals, np.nan)
    for i in range(len(vals)):
        p = vals[i - 1] if i - 1 >= 0 else np.nan
        c = vals[i]
        n = vals[i + 1] if i + 1 < len(vals) else np.nan
        parts = []
        if pd.notna(p):
            parts.append((float(p), 0.25))
        if pd.notna(c):
            parts.append((float(c), 0.5))
        if pd.notna(n):
            parts.append((float(n), 0.25))
        if parts:
            ssum = sum(v * w for v, w in parts)
            wsum = sum(w for _, w in parts)
            out[i] = ssum / wsum if wsum > 1e-9 else np.nan
        else:
            out[i] = np.nan
    return pd.Series(out, index=s.index)


def get_pref_radii_row(row: pd.Series):
    for a in AGENCY_PREF:
        for t in RAD_TIERS:
            vals = {}
            for q in QUADS:
                col = f"{a}_{t}_{q}"
                v = clean_radius(row.get(col))
                vals[q] = v * NM_TO_KM if pd.notna(v) else np.nan
            if any(pd.notna(vals[q]) and vals[q] > 0 for q in QUADS):
                return t, vals, a
    return None, None, None


def angular_curve_from_quads(ne, se, sw, nw, n_pts=360):
    r = np.full(n_pts, np.nan)
    def fill(a, b, ra, rb):
        if np.isnan(ra) and np.isnan(rb):
            return
        if np.isnan(ra):
            ra = rb
        if np.isnan(rb):
            rb = ra
        span = (b - a) % n_pts
        span = span if span > 0 else span + n_pts
        for i in range(span):
            t = i / max(span - 1, 1)
            r[(a + i) % n_pts] = (1 - t) * ra + t * rb
    fill(0, 90, ne, se)
    fill(90, 180, se, sw)
    fill(180, 270, sw, nw)
    fill(270, 360, nw, ne)
    out = np.copy(r)
    for i in range(n_pts):
        a = r[(i - 1) % n_pts]
        b = r[i]
        c = r[(i + 1) % n_pts]
        if np.isnan(a) and np.isnan(b) and np.isnan(c):
            out[i] = np.nan
            continue
        aa = a if not np.isnan(a) else b
        cc = c if not np.isnan(c) else b
        bb = b if not np.isnan(b) else 0.5 * (aa + cc)
        out[i] = (aa + 2 * bb + cc) / 4.0
    return out


def build_quadrant_polygon(lat, lon, ne_km, se_km, sw_km, nw_km, n_pts: int = 360, frame_off: float = 0.0) -> Optional[Polygon]:
    if all(pd.isna(v) or v <= 0 for v in [ne_km, se_km, sw_km, nw_km]):
        return None
    conv = lambda v: (v / NM_TO_KM) if pd.notna(v) else np.nan
    r_nm = angular_curve_from_quads(conv(ne_km), conv(se_km), conv(sw_km), conv(nw_km), n_pts)
    lon0 = float(lon) + float(frame_off)
    coslat = np.cos(np.radians(lat))
    denom = (111.0 * coslat) if abs(coslat) > 1e-12 else 111.0
    xs_tmp, ys = [], []
    for deg in range(n_pts):
        rr_nm = r_nm[deg]
        if pd.isna(rr_nm) or rr_nm <= 0:
            continue
        rr_km = rr_nm * NM_TO_KM
        th = np.radians(deg)
        x = lon0 + (rr_km * np.cos(th)) / denom
        y = lat + (rr_km * np.sin(th)) / 111.0
        xs_tmp.append(float(x))
        ys.append(float(y))
    if len(xs_tmp) < 3:
        return None
    xs = _unwrap_lon_seq(xs_tmp)
    pts = [(x - frame_off, y) for x, y in zip(xs, ys)]
    if len(pts) >= 2 and pts[0] != pts[-1]:
        pts.append(pts[0])
    try:
        return Polygon(pts)
    except Exception:
        return None


def per_sid_polygon(group: pd.DataFrame, cfg: Config, sid: Optional[str] = None) -> Optional[Polygon]:
    group = group.sort_values("ISO_TIME")
    def _trace(msg: str):
        if cfg.trace_poly and _dbg_enabled_for(cfg, sid):
            logging.info(f"[TRACE:{sid}] {msg}")

    # Collect base points / scales
    base_pts, lat_vals = [], []
    for _, r in group.iterrows():
        la, lo, _ = choose_latlon(r)
        if pd.notna(la) and pd.notna(lo):
            base_pts.append((float(lo), float(la)))
            lat_vals.append(float(la))
    lat_hint = float(np.median(lat_vals)) if lat_vals else 0.0
    lat_hint = float(max(-85.0, min(85.0, lat_hint)))
    simplify_m = float(abs(cfg.simplify_deg)) * 111_000.0
    openclose_m = float(abs(cfg.openclose_km)) * 1_000.0

    frame_off = _frame_offset_for_track(base_pts)
    _trace(
        f"entered per_sid_polygon | frame_off={frame_off:+.0f} deg | simplify_m={simplify_m:.1f} | openclose_m={openclose_m:.1f}"
    )

    # Adaptive fallbacks by intensity
    def _resolve_sshs(row: pd.Series):
        v = row.get("__SSHS_BLEND")
        if pd.isna(v):
            v = row.get("USA_SSHS")
        if pd.isna(v):
            w = pd.to_numeric(row.get("__WIND_BLEND"), errors="coerce")
            if pd.isna(w):
                w = pd.to_numeric(row.get("WIND"), errors="coerce")
            if pd.notna(w):
                v = sshs_from_knots(float(w))
        return v

    def _adaptive_params() -> tuple[float, float, float]:
        ss = pd.to_numeric(group.get("__SSHS_BLEND"), errors="coerce")
        mx = int(np.nanmax(ss)) if ss is not None and ss.notna().any() else -1
        if mx < 0:
            for _, r in group.iterrows():
                vv = _resolve_sshs(r)
                if pd.notna(vv):
                    mx = max(mx, int(vv))
        cat = max(0, mx)
        union_nm = max(cfg.fallback_union_nm, 20.0 + 12.0 * cat)
        linebuf_m = max(cfg.fallback_linebuf_m, 3000.0 * (cat + 1.0))
        min_area_km2 = 60.0 + 20.0 * cat
        return float(union_nm), float(linebuf_m), float(min_area_km2)

    union_nm_adapt, linebuf_m_adapt, min_area_km2 = _adaptive_params()
    _trace(
        f"adaptive: union_nm={union_nm_adapt:.1f} nm | linebuf_m={linebuf_m_adapt:.0f} m | min_area={min_area_km2:.0f} km²"
    )

    # Quadrant radii
    rad_series: Optional[dict[str, pd.Series]] = None
    mode_for_sid = "weighted"
    # attempt weighted first; if coverage low, fall back to first-agency mode
    rs: dict[str, pd.Series] = {}
    have_any = pd.Series(False, index=group.index)
    for q in QUADS:
        num = pd.Series(0.0, index=group.index)
        den = pd.Series(0.0, index=group.index)
        quad_any = pd.Series(False, index=group.index)
        for ag, wt in AGENCY_WEIGHTS.items():
            for tier in ("R34", "R50", "R64"):
                col = f"{ag}_{tier}_{q}"
                if col not in group.columns:
                    continue
                v = pd.to_numeric(group[col], errors="coerce")
                m = v.notna()
                if m.any():
                    num[m] += v[m].astype(float) * wt
                    den[m] += wt
                    quad_any[m] = True
        with np.errstate(invalid="ignore", divide="ignore"):
            rs[q] = (num / den) * NM_TO_KM
        have_any = have_any | quad_any
    cov = float(have_any.mean()) if len(have_any) else 0.0
    _trace(f"weighted quadrant coverage={cov:.3f}")
    if cov >= cfg.min_weighted_cov:
        rad_series = rs
    else:
        mode_for_sid = "first"
        tier, _, ag = get_pref_radii_row(group.iloc[-1])
        if tier and ag:
            rad_series = {}
            for q in QUADS:
                col = f"{ag}_{tier}_{q}"
                s = pd.to_numeric(group[col], errors="coerce") if col in group.columns else pd.Series(np.nan, index=group.index)
                rad_series[q] = _smooth_series_3pt(s) * NM_TO_KM * cfg.uncertainty_scale
        _trace(f"coverage<{cfg.min_weighted_cov:.2f} → switch to 'first' (tier={tier}, ag={ag})")

    # Build per-timestep polygons
    polys = []
    last_method = None
    for idx, row in group.iterrows():
        lat, lon, _src = choose_latlon(row)
        if pd.isna(lat) or pd.isna(lon):
            continue
        cands: list[tuple[str, float, Polygon]] = []

        if rad_series is not None:
            ne = float(rad_series["NE"].get(idx, np.nan))
            se = float(rad_series["SE"].get(idx, np.nan))
            sw = float(rad_series["SW"].get(idx, np.nan))
            nw = float(rad_series["NW"].get(idx, np.nan))
            qp = build_quadrant_polygon(lat, lon, ne, se, sw, nw, n_pts=360, frame_off=frame_off)
            _log_geom_diag("candidate:quadrant:preclean", qp, frame_off, sid, enabled=(cfg.trace_poly or cfg.verbose_seam))
            qp = _clean_poly_local(_shift_lon_geom(qp, frame_off), lat_hint, simplify_m, openclose_m)
            qp = _wrap_to_180(_shift_lon_geom(qp, -frame_off))
            _log_geom_diag("candidate:quadrant:postclean", qp, frame_off, sid, enabled=(cfg.trace_poly or cfg.verbose_seam))
            if qp is not None and not qp.is_empty:
                cands.append(("quadrant_smoothed", 0.92, qp))

        roci_nm = first_positive(row, [f"{a}_ROCI" for a in AGENCY_PREF])
        if pd.notna(roci_nm) and roci_nm > 0:
            c = _circle_laea(lat, lon, roci_nm, frame_off=frame_off)
            _log_geom_diag("candidate:roci:preclean", c, frame_off, sid, enabled=(cfg.trace_poly or cfg.verbose_seam))
            c = _clean_poly_local(_shift_lon_geom(c, frame_off), lat_hint, simplify_m, openclose_m)
            c = _wrap_to_180(_shift_lon_geom(c, -frame_off))
            _log_geom_diag("candidate:roci:postclean", c, frame_off, sid, enabled=(cfg.trace_poly or cfg.verbose_seam))
            if c is not None and not c.is_empty:
                cands.append(("roci", 0.80, c))

        rmw_nm = first_positive(row, [f"{a}_RMW" for a in AGENCY_PREF])
        if pd.notna(rmw_nm) and rmw_nm > 0:
            sshs_here = _resolve_sshs(row)
            k = SSHS_SCALE_K.get(int(sshs_here) if pd.notna(sshs_here) else 2, 2.0)
            c = _circle_laea(lat, lon, rmw_nm * k, frame_off=frame_off)
            _log_geom_diag("candidate:rmw_scaled:preclean", c, frame_off, sid, enabled=(cfg.trace_poly or cfg.verbose_seam))
            c = _clean_poly_local(_shift_lon_geom(c, frame_off), lat_hint, simplify_m, openclose_m)
            c = _wrap_to_180(_shift_lon_geom(c, -frame_off))
            _log_geom_diag("candidate:rmw_scaled:postclean", c, frame_off, sid, enabled=(cfg.trace_poly or cfg.verbose_seam))
            if c is not None and not c.is_empty:
                cands.append(("rmw_scaled", 0.65, c))

        if cfg.climo_fallback:
            sshs_here = _resolve_sshs(row)
            basin_here = row.get("BASIN")
            r34 = np.nan
            if pd.notna(sshs_here) and pd.notna(basin_here):
                try:
                    r34 = CLIMO_R34_NM.get(str(basin_here), {}).get(int(sshs_here), np.nan)
                except Exception:
                    r34 = np.nan
            if pd.notna(r34) and r34 > 0:
                c = _circle_laea(lat, lon, r34, frame_off=frame_off)
                _log_geom_diag("candidate:climo:preclean", c, frame_off, sid, enabled=(cfg.trace_poly or cfg.verbose_seam))
                c = _clean_poly_local(_shift_lon_geom(c, frame_off), lat_hint, simplify_m, openclose_m)
                c = _wrap_to_180(_shift_lon_geom(c, -frame_off))
                _log_geom_diag("candidate:climo:postclean", c, frame_off, sid, enabled=(cfg.trace_poly or cfg.verbose_seam))
                if c is not None and not c.is_empty:
                    cands.append(("climo_r34", 0.55, c))

        if not cands:
            continue

        method, conf, geom = max(cands, key=lambda x: x[1])
        if last_method and method != last_method:
            prev = next((c for c in cands if c[0] == last_method), None)
            if prev and (conf - prev[1]) < cfg.switch_margin:
                method, conf, geom = prev
        geom = _ensure_min_area(geom, lat_hint, min_area_km2, nudge_half_width_m=linebuf_m_adapt * 0.5)
        if geom is None or geom.is_empty:
            continue
        polys.append(geom)
        last_method = method

    if not polys:
        dpts = _densify_track_points(base_pts, max_seg_km=_DENSIFY_MAX_SEG_KM)
        if dpts:
            # Capsule via buffering track by union_nm_adapt (meters)
            line = LineString([(x + frame_off, y) for x, y in dpts])
            mid = line.interpolate(0.5, normalized=True)
            lon0, lat0 = float(mid.x), float(mid.y)
            fwd, inv = _proj_local_fns(lon0, lat0)
            line_m = transform(fwd, line)
            hw_m = max(500.0, float(union_nm_adapt) * NM_TO_KM * 1000.0)
            capsule_m = line_m.buffer(hw_m, cap_style=1, join_style=1)
            capsule_m = capsule_m.buffer(hw_m * _SMOOTH_FRACTION_OF_RADIUS).buffer(-hw_m * _SMOOTH_FRACTION_OF_RADIUS)
            cap_ll = transform(inv, capsule_m)
            cap_ll = _wrap_to_180(_shift_lon_geom(cap_ll, -frame_off))
            cap_ll = _ensure_min_area(cap_ll, lat_hint, min_area_km2, nudge_half_width_m=linebuf_m_adapt * 0.5)
            if cap_ll is not None and not cap_ll.is_empty:
                _trace(f"using capsule buffer (±{union_nm_adapt:.0f} nm)")
                return extract_polygons_only(cap_ll)
        # Last resort: convex hull of points
        if len(base_pts) >= 3:
            hull = MultiPoint([Point(x, y) for x, y in base_pts]).convex_hull
            hull = _wrap_to_180(_shift_lon_geom(hull, 0.0))
            hull = _ensure_min_area(hull, lat_hint, min_area_km2, nudge_half_width_m=linebuf_m_adapt)
            if hull is not None and not hull.is_empty:
                _trace("using convex hull")
                return extract_polygons_only(hull)
        return None

    # Union of per-step polygons (dateline-safe by wrapping once)
    try:
        u = unary_union(polys)
        return extract_polygons_only(_wrap_to_180(u))
    except Exception:
        try:
            return extract_polygons_only(_wrap_to_180(unary_union(polys)))
        except Exception:
            return None


# =============================================================================
# Public API
# =============================================================================

def build_storm_polygons(df: pd.DataFrame, cfg: Config) -> gpd.GeoDataFrame:
    """Group by SID and assemble polygons; returns GeoDataFrame (EPSG:4326)."""
    if df.empty:
        return gpd.GeoDataFrame(columns=["SID", "geometry"], geometry="geometry", crs="EPSG:4326")

    if "SID" not in df.columns:
        raise ValueError("Input DataFrame must include a 'SID' column")

    if cfg.year_ge is not None and "SEASON" in df.columns:
        df = df[pd.to_numeric(df["SEASON"], errors="coerce") >= int(cfg.year_ge)]

    df = add_blended_sshs(df)

    out_rows = []
    processed = skipped = 0

    for sid, group in df.groupby("SID", sort=False):
        poly = per_sid_polygon(group, cfg, sid=str(sid))
        if poly is None or poly.is_empty:
            skipped += 1
            logging.error(f"[Cyclones] storm polygon failed for {sid}: {len(group)} rows")
            # optional artifact dump
            if cfg.fail_dump_dir:
                try:
                    cfg.fail_dump_dir.mkdir(parents=True, exist_ok=True)
                    gj = {"type": "FeatureCollection", "features": []}
                    (cfg.fail_dump_dir / f"{sid}_empty.geojson").write_text(json.dumps(gj), encoding="utf-8")
                except Exception:
                    pass
            continue
        out_rows.append({"SID": sid, "geometry": poly})
        processed += 1

    gdf = gpd.GeoDataFrame(out_rows, geometry="geometry", crs="EPSG:4326")
    _log_processed("Cyclones", processed, skipped)
    return gdf


def write_outputs(gdf: gpd.GeoDataFrame, cfg: Config) -> None:
    if cfg.out_gpkg:
        _quiet_write_gpkg(gdf, str(cfg.out_gpkg), layer_name=cfg.out_layer, overwrite=True)
        logging.info(f"[I/O] wrote {cfg.out_layer} → {cfg.out_gpkg}")
    if cfg.out_csv:
        cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
        gdf.drop(columns="geometry").to_csv(cfg.out_csv, index=False, encoding="utf-8")
        logging.info(f"[I/O] wrote attributes → {cfg.out_csv}")


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="storms_refactored",
        description="Unified cyclone & tornado polygon builder (refactored)",
    )
    p.add_argument("--ibtracs-csv", type=Path, required=True, help="Path to IBTrACS CSV (utf-8)")
    p.add_argument("--out-gpkg", type=Path, help="Output GPKG path")
    p.add_argument("--out-layer", type=str, default="cyclones", help="Output GPKG layer name")
    p.add_argument("--out-csv", type=Path, help="Optional CSV of attributes (utf-8)")

    # Filters & behavior
    p.add_argument("--year-ge", type=int, help="Filter to storms with SEASON >= this year")
    p.add_argument("--min-sshs", type=int, help="Filter to storms with blended SSHS >= this value")
    p.add_argument("--oecd-only", action="store_true", help="(placeholder) keep storms impacting OECD countries")

    # Fallbacks & geometry cleaning
    p.add_argument("--fallback-union-nm", type=float, default=25.0)
    p.add_argument("--fallback-linebuf-m", type=float, default=5000.0)
    p.add_argument("--simplify-deg", type=float, default=0.02)
    p.add_argument("--openclose-km", type=float, default=5.0)
    p.add_argument("--min-weighted-cov", type=float, default=0.15)
    p.add_argument("--uncertainty-scale", type=float, default=1.0)
    p.add_argument("--no-climo-fallback", action="store_true")
    p.add_argument("--switch-margin", type=float, default=0.08)

    # Debugging & artifacts
    p.add_argument("--debug-all", action="store_true")
    p.add_argument("--debug-sids", type=str, help="Comma-separated SID list to trace")
    p.add_argument("--trace-poly", action="store_true")
    p.add_argument("--verbose-seam", action="store_true")
    p.add_argument("--fail-dump-dir", type=Path, help="Dump bad cases (GeoJSON) here")

    # Logging
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]) 
    p.add_argument("--log-file", type=Path, help="Optional utf-8 log file path")

    return p


def cfg_from_args(args: argparse.Namespace) -> Config:
    return Config(
        debug_all=args.debug_all,
        debug_sids=set(map(str.strip, args.debug_sids.split(","))) if args.debug_sids else set(),
        trace_poly=args.trace_poly,
        verbose_seam=args.verbose_seam,
        fail_dump_dir=args.fail_dump_dir,
        fallback_union_nm=float(args.fallback_union_nm),
        fallback_linebuf_m=float(args.fallback_linebuf_m),
        min_weighted_cov=float(args.min_weighted_cov),
        uncertainty_scale=float(args.uncertainty_scale),
        simplify_deg=float(args.simplify_deg),
        openclose_km=float(args.openclose_km),
        climo_fallback=not bool(args.no_climo_fallback),
        switch_margin=float(args.switch_margin),
        out_gpkg=args.out_gpkg,
        out_layer=args.out_layer,
        out_csv=args.out_csv,
        year_ge=args.year_ge,
        min_sshs=args.min_sshs,
        oecd_only=bool(args.oecd_only),
        log_level=args.log_level,
        log_file=args.log_file,
        )


def read_ibtracs_csv(path: Path) -> pd.DataFrame:
    logging.info(f"[I/O] reading IBTrACS: {path}")
    return pd.read_csv(path, encoding="utf-8")


def _apply_min_sshs(df: pd.DataFrame, min_sshs: Optional[int]) -> pd.DataFrame:
    if min_sshs is None:
        return df
    s = pd.to_numeric(df.get("__SSHS_BLEND"), errors="coerce")
    m = s.notna() & (s >= int(min_sshs))
    return df[m]


def _quiet_write_gpkg(df: gpd.GeoDataFrame, path: str, layer_name: str, overwrite: bool = True) -> None:
    root = logging.getLogger()
    prev = root.level
    try:
        if prev <= logging.INFO:
            root.setLevel(logging.WARNING)
        write_gpkg(df, path, layer_name=layer_name, overwrite=overwrite)
    finally:
        root.setLevel(prev)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = cfg_from_args(args)
    setup_logging(cfg.log_level, cfg.log_file)

    _log_header("Cyclones", "Processing IBTrACS storm data...")

    df = read_ibtracs_csv(args.ibtracs_csv)
    if cfg.year_ge is not None and "SEASON" in df.columns:
        df = df[pd.to_numeric(df["SEASON"], errors="coerce") >= int(cfg.year_ge)]
    df = add_blended_sshs(df)
    df = _apply_min_sshs(df, cfg.min_sshs)

    gdf = build_storm_polygons(df, cfg)
    write_outputs(gdf, cfg)

    _log_pipeline_done()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
