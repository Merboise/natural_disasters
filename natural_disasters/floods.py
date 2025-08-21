# natural_disasters/floods.py
# -*- coding: utf-8 -*-
"""
Flood hazard ingestion for the unified pipeline.

Sources supported here (no EM-DAT loading; EM-DAT filtering happens in main.py):
- DFO polygons (+ optional attributes)
- HANZE (EU) polygons via regions v2010/v2021 lookups from an events CSV
- USFD (US flood database) point events (kept as points; can be buffered later if desired)

Output:
- GeoDataFrame in canonical schema:
  event_id, event_type, start_time, end_time, band, geom_method, geom_confidence, area_km2, geometry
"""

from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection, box
from shapely.validation import make_valid
from shapely.ops import unary_union

# ---------------- Constants / CRS ----------------
WGS84 = "EPSG:4326"
WORLD_EQ_AREA = "EPSG:6933"  # Equal-area meters (World Cylindrical Equal Area)

# ---------------- Canonical schema (local to floods) ---------------
CANON_COLS = [
    "event_id", "event_type", "start_time", "end_time",
    "band", "geom_method", "geom_confidence", "area_km2", "geometry"
]

# ---------------- Utilities ----------------
def _ensure_cols(gdf: gpd.GeoDataFrame, cols: List[str]) -> gpd.GeoDataFrame:
    for c in cols:
        if c not in gdf.columns:
            gdf[c] = None
    return gdf

def _area_km2(geom) -> Optional[float]:
    try:
        if geom is None or geom.is_empty:
            return None
        return gpd.GeoSeries([geom], crs=WGS84).to_crs(WORLD_EQ_AREA).area.iloc[0] / 1e6
    except Exception:
        return None

def _safe_clean(g):
    """Fix invalids, keep polygonal parts only, robust buffer(0)."""
    if g is None or getattr(g, "is_empty", True):
        return None
    try:
        if not g.is_valid:
            g = make_valid(g)
        g = _keep_surface_parts(g)
        if g is None or g.is_empty:
            return None
        g = g.buffer(0)
        if g is None or g.is_empty:
            return None
        return _keep_surface_parts(g)
    except Exception:
        try:
            gg = g.buffer(0)
            return None if (gg is None or gg.is_empty) else _keep_surface_parts(gg)
        except Exception:
            return None

def _keep_surface_parts(g):
    if g is None or g.is_empty:
        return None
    if isinstance(g, (Polygon, MultiPolygon)):
        return g
    if isinstance(g, GeometryCollection):
        parts = [p for p in g.geoms if isinstance(p, (Polygon, MultiPolygon))]
        if not parts:
            return None
        try:
            return unary_union(parts)
        except Exception:
            u = parts[0]
            for p in parts[1:]:
                try:
                    u = u.union(p)
                except Exception:
                    try:
                        u = u.buffer(0).union(p.buffer(0))
                    except Exception:
                        pass
            return u
    return None  # ignore points/lines

def _safe_unary_union(geoms):
    cleaned = []
    for g in geoms:
        gg = _safe_clean(g)
        if gg is not None and not gg.is_empty:
            cleaned.append(gg)
    if not cleaned:
        return None
    try:
        return unary_union(cleaned)
    except Exception:
        u = cleaned[0]
        for g in cleaned[1:]:
            try:
                u = u.union(g)
            except Exception:
                try:
                    u = u.buffer(0).union(g.buffer(0))
                except Exception:
                    continue
        return u

# ---------------- IO helpers ----------------
def _read_tabular(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    try:
        return pd.read_csv(path, sep=None, engine="python", low_memory=False, encoding="utf-8")
    except Exception:
        pass
    for sep in [",", ";", "\t", "|"]:
        try:
            return pd.read_csv(path, sep=sep, low_memory=False, encoding="utf-8")
        except Exception:
            continue
    for enc in ["utf-8-sig", "latin-1"]:
        try:
            return pd.read_csv(path, sep=None, engine="python", low_memory=False, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False)

# ---------------- Date helpers ----------------
def _scalar_date(y, m, d) -> Optional[pd.Timestamp]:
    try:
        if pd.isna(y):
            return pd.NaT
        y = int(y)
        m = int(m) if pd.notna(m) else 1
        d = int(d) if pd.notna(d) else 1
        m = max(1, min(12, m))
        d = max(1, min(28, d))
        return pd.Timestamp(year=y, month=m, day=d)
    except Exception:
        return pd.NaT

def parse_hanze_date(s: str) -> Optional[pd.Timestamp]:
    if pd.isna(s):
        return None
    return pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")

def parse_usfd_dt(value) -> Optional[pd.Timestamp]:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if "e" in s.lower():
        try:
            s = str(int(float(s)))
        except Exception:
            return None
    try:
        n = len(s)
        if n >= 14:
            return pd.to_datetime(s[:14], format="%Y%m%d%H%M%S", errors="coerce")
        elif n == 8:
            return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        elif n == 6:
            return pd.to_datetime(s, format="%Y%m", errors="coerce")
        elif n == 4:
            return pd.to_datetime(s, format="%Y", errors="coerce")
    except Exception:
        pass
    return None

# ---------------- Source loaders ----------------
def load_dfo(dfo_shp: Path | None, dfo_attr: Path | None = None,
             simplify_tolerance_deg: float | None = 0.0003) -> list[dict]:
    """
    DFO flood polygons with optional attributes join. Expects a polygon layer.
    """
    if not dfo_shp or not Path(dfo_shp).exists():
        return []
    gdf = gpd.read_file(dfo_shp).to_crs(WGS84)
    if simplify_tolerance_deg and simplify_tolerance_deg > 0:
        gdf["geometry"] = gdf.geometry.apply(_safe_clean).simplify(simplify_tolerance_deg, preserve_topology=True)

    attrs = _read_tabular(Path(dfo_attr)) if dfo_attr and Path(dfo_attr).exists() else pd.DataFrame()
    if not attrs.empty:
        id_cols = [c for c in attrs.columns if c.strip().lower() == "id"]
        if id_cols:
            attrs["ID"] = pd.to_numeric(attrs[id_cols[0]], errors="coerce")
            gdf = gdf.merge(attrs, on="ID", how="left")

    out = []
    for _, r in gdf.iterrows():
        ev_id = r.get("ID", None)
        out.append({
            "geometry": r.geometry,
            "event_id": f"DFO_{ev_id}" if pd.notna(ev_id) else None,
            "source": "DFO",
            "start_date": pd.to_datetime(r.get("Began", None), errors="coerce"),
            "end_date": pd.to_datetime(r.get("Ended", None), errors="coerce"),
            "country": r.get("Country", None),
            "area_km2": (None if pd.isna(r.get("Area", None)) else float(r.get("Area"))),
            "fatalities": r.get("Dead", None),
            "people_affected": r.get("Displaced", None),
            "cause": r.get("MainCause", None),
            "footprint_method": "DFO_polygon"
        })
    return out

def _resolve_code_field(gdf: gpd.GeoDataFrame, preferred: str | None,
                        fallback_candidates: list[str], expected_codes: list[str]) -> str:
    cols = set(gdf.columns)
    if preferred and preferred in cols:
        return preferred
    for name in fallback_candidates:
        if name in cols:
            return name
    exp = {c.strip() for c in expected_codes if c and isinstance(c, str)}
    best_col, best_hits = None, -1
    for c in gdf.columns:
        s = gdf[c]
        if s.dtype == object or str(s.dtype).startswith("string"):
            vals = s.astype(str).str.strip()
            hits = vals.isin(exp).sum()
            if hits > best_hits:
                best_hits = hits
                best_col = c
    if best_col and best_hits > 0:
        return best_col
    raise ValueError(f"Could not find a suitable region code column among: {list(gdf.columns)}")

def load_hanze_csv_to_polys(
    events_csv: Path | None,
    regions_v2010_path: Path | None,
    regions_v2021_path: Path | None,
    regions_layer_2010: Optional[str] = None,
    regions_layer_2021: Optional[str] = None,
    code_field_2010: Optional[str] = None,
    code_field_2021: Optional[str] = None,
    simplify_tolerance_deg: float | None = 0.0003,
) -> list[dict]:
    if not events_csv or not Path(events_csv).exists():
        return []
    df = pd.read_csv(events_csv, low_memory=False)

    gdf2010 = gpd.read_file(regions_v2010_path, layer=regions_layer_2010).to_crs(WGS84) if regions_v2010_path else None
    gdf2021 = gpd.read_file(regions_v2021_path, layer=regions_layer_2021).to_crs(WGS84) if regions_v2021_path else None

    if gdf2010 is not None and simplify_tolerance_deg:
        gdf2010["geometry"] = gdf2010.geometry.apply(_safe_clean).simplify(simplify_tolerance_deg, preserve_topology=True)
    if gdf2021 is not None and simplify_tolerance_deg:
        gdf2021["geometry"] = gdf2021.geometry.apply(_safe_clean).simplify(simplify_tolerance_deg, preserve_topology=True)

    out: list[dict] = []
    for _, r in df.iterrows():
        sd = parse_hanze_date(r.get("Start date"))
        ed = parse_hanze_date(r.get("End date"))
        country = r.get("Country name")
        area = r.get("Area flooded")
        fat = r.get("Fatalities")
        aff = r.get("Persons affected")
        cause = r.get("Cause")
        evid = r.get("ID")

        codes_2021 = r.get("Regions affected (v2021)")
        codes_2010 = r.get("Regions affected (v2010)")

        geom = None
        if isinstance(codes_2021, str) and codes_2021.strip() and gdf2021 is not None:
            codes = [c.strip() for c in codes_2021.split(";") if c.strip()]
            candidates_2021 = ["REG_CODE", "HZ2021_ID", "HZ_2021", "REG_ID", "CODE", "NUTS_ID", "ID", "GID_1", "GID_2"]
            fld_2021 = _resolve_code_field(gdf2021, code_field_2021, candidates_2021, codes)
            sel = gdf2021[gdf2021[fld_2021].astype(str).str.strip().isin(codes)]
            if len(sel):
                geom = _safe_unary_union(sel.geometry)

        if (geom is None or geom.is_empty) and isinstance(codes_2010, str) and codes_2010.strip() and gdf2010 is not None:
            codes = [c.strip() for c in codes_2010.split(";") if c.strip()]
            candidates_2010 = ["REG_CODE", "HZ2010_ID", "HZ_2010", "REG_ID", "CODE", "NUTS_ID", "ID", "GID_1", "GID_2"]
            fld_2010 = _resolve_code_field(gdf2010, code_field_2010, candidates_2010, codes)
            sel = gdf2010[gdf2010[fld_2010].astype(str).str.strip().isin(codes)]
            if len(sel):
                geom = _safe_unary_union(sel.geometry)

        out.append({
            "geometry": geom,
            "event_id": f"HANZE_{evid}",
            "source": "HANZE",
            "start_date": sd, "end_date": ed,
            "country": country,
            "area_km2": None if pd.isna(area) else float(area),
            "fatalities": fat,
            "people_affected": aff,
            "cause": cause,
            "footprint_method": "HANZE_regions"
        })
    return out

def load_usfd_v11(path_csv: Path | None) -> list[dict]:
    if not path_csv or not Path(path_csv).exists():
        return []
    df = pd.read_csv(path_csv, low_memory=False)
    recs: list[dict] = []
    for idx, r in df.iterrows():
        sd = parse_usfd_dt(r.get("DATE_BEGIN"))
        ed = parse_usfd_dt(r.get("DATE_END"))
        lon = r.get("LON"); lat = r.get("LAT")
        geom = Point(float(lon), float(lat)) if pd.notna(lon) and pd.notna(lat) else None
        area = r.get("AREA", None)
        recs.append({
            "geometry": geom,
            "event_id": f"USFD_{idx}",
            "source": "USFD",
            "start_date": sd, "end_date": ed,
            "country": r.get("COUNTRY"),
            "area_km2": None if pd.isna(area) else float(area),
            "fatalities": r.get("FATALITY"),
            "people_affected": None,
            "cause": r.get("CAUSE"),
            "footprint_method": "USFD_point",
            "state": r.get("STATE"),
            "location": r.get("LOCATION"),
            "severity": r.get("SEVERITY"),
            "source_db": r.get("SOURCE_DB"),
            "source_id": r.get("SOURCE_ID"),
            "description": r.get("DESCRIPTION"),
        })
    return recs

# ---------------- Canonicalizer for floods ----------------
def unify_schema_floods(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gdf if isinstance(gdf, gpd.GeoDataFrame) else gpd.GeoDataFrame(columns=CANON_COLS, geometry=[])
    out = gdf.copy()
    out["event_type"] = "flood"

    # standardize start/end time columns
    st = out.get("start_time")
    en = out.get("end_time")
    if st is None:
        out["start_time"] = pd.to_datetime(out.get("start_date"), errors="coerce")
    if en is None:
        out["end_time"]   = pd.to_datetime(out.get("end_date"), errors="coerce")
    mask = out["end_time"].isna() & out["start_time"].notna()
    out.loc[mask, "end_time"] = out.loc[mask, "start_time"]

    # geom_method from footprint_method if present
    if "footprint_method" in out.columns and "geom_method" not in out.columns:
        out["geom_method"] = out["footprint_method"]

    # defaults
    out["band"] = None
    if "geom_confidence" not in out.columns:
        out["geom_confidence"] = None

    # area (recompute if missing)
    if "area_km2" not in out.columns or out["area_km2"].isna().all():
        out["area_km2"] = out.geometry.apply(_area_km2)
    else:
        # ensure numeric
        out["area_km2"] = pd.to_numeric(out["area_km2"], errors="coerce")

    out = _ensure_cols(out, CANON_COLS)
    return out[CANON_COLS]

# ---------------- Public entry point ----------------
def process_flood_data(
    dfo_shp: str | Path | None = None,
    dfo_attr: str | Path | None = None,
    hanze_csv: str | Path | None = None,
    hanze_regions_2010: str | Path | None = None,
    hanze_regions_2021: str | Path | None = None,
    hanze_regions_2010_layer: Optional[str] = None,
    hanze_regions_2021_layer: Optional[str] = None,
    hanze_code_field_2010: Optional[str] = None,
    hanze_code_field_2021: Optional[str] = None,
    usfd_csv: str | Path | None = None,
    simplify_tolerance_deg: float | None = 0.0003,
) -> gpd.GeoDataFrame:
    """
    Load/normalize flood sources (no EM-DAT here), return canonical GeoDataFrame.
    Main controls EM-DAT filtering and combined writing.

    Returns: GeoDataFrame with CANON_COLS schema; CRS=EPSG:4326
    """
    # Collect records
    recs: list[dict] = []
    recs += load_dfo(Path(dfo_shp) if dfo_shp else None,
                     Path(dfo_attr) if dfo_attr else None,
                     simplify_tolerance_deg=simplify_tolerance_deg)
    recs += load_hanze_csv_to_polys(
        Path(hanze_csv) if hanze_csv else None,
        Path(hanze_regions_2010) if hanze_regions_2010 else None,
        Path(hanze_regions_2021) if hanze_regions_2021 else None,
        regions_layer_2010=hanze_regions_2010_layer,
        regions_layer_2021=hanze_regions_2021_layer,
        code_field_2010=hanze_code_field_2010,
        code_field_2021=hanze_code_field_2021,
        simplify_tolerance_deg=simplify_tolerance_deg,
        )
    recs += load_usfd_v11(Path(usfd_csv) if usfd_csv else None)

    if not recs:
        return gpd.GeoDataFrame(columns=CANON_COLS, geometry=[], crs=WGS84)

    gdf = gpd.GeoDataFrame(recs, crs=WGS84)

    # Clean any NaT handling for times
    gdf["start_date"] = pd.to_datetime(gdf.get("start_date"), errors="coerce")
    gdf["end_date"]   = pd.to_datetime(gdf.get("end_date"), errors="coerce")

    # Canonicalize schema
    out = unify_schema_floods(gdf)

    return out
