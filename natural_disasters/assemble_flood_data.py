#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
assemble_flood_data.py

Build a consolidated flood-event layer by combining:
- DFO (polygons + attributes)
- HANZE (EU subnational polygons from regions v2010/v2021)
- USFD (US events; point/circle if enabled)
- EM-DAT (global; admin polygons or points)

Priority for consolidation (final footprint):
  HANZE/USFD > DFO > EMDAT

Pipeline:
  1) Load all sources -> GeoDataFrame
  2) (Optional) Enrich with DEM + GHSL population density
  3) Cluster into master events (time + space) -> master_id
  4) Mark clusters with emdat_anchor (True if any EMDAT in cluster)
  5) Filter by --emdat-scope {both|only|non} at cluster level
  6) Write:
       - Master CSV (all events with master_id + emdat_anchor)
       - Consolidated GPKG (overrides applied; keeps master_id + emdat_anchor)
"""

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import re

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection, box
from shapely.validation import make_valid
from shapely.ops import unary_union

# ---------- optional path helpers ----------
try:
    from .helpers import data_path, output_path
except Exception:
    def data_path(*parts: str) -> Path:
        p = Path("data")
        for q in parts:
            p /= q
        return p.resolve()

    def output_path(*parts: str) -> Path:
        p = Path("disaster_output")
        p.mkdir(parents=True, exist_ok=True)
        for q in parts:
            p /= q
        return p.resolve()

# ---- CRS / constants ----
WGS84 = "EPSG:4326"
WORLD_EQ_AREA = "EPSG:6933"  # World Cylindrical Equal Area (meters)
GHSL_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
_EPOCH_RE = re.compile(r"_E(1975|1980|1985|1990|1995|2000|2005|2010|2015|2020)_")

#
# Logging helpers
#

def vlog(enabled: bool, *msg):
    if enabled:
        print(*msg, flush=True)


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def append_csv(path: Path, df: pd.DataFrame, header_if_new: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    df.to_csv(path, mode="a", index=False, header=(header_if_new and is_new))

def append_gpkg(path: Path, layer: str, gdf: gpd.GeoDataFrame, create_crs: str = "EPSG:4326"):
    """
    Append to a GeoPackage layer. Creates it if missing.
    Uses pyogrio for robust appends.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # create new
        pyogrio.write_dataframe(gdf, path, layer=layer, driver="GPKG")
    else:
        pyogrio.write_dataframe(gdf, path, layer=layer, driver="GPKG", append=True)

def flush_snapshot(raw_events: Optional[gpd.GeoDataFrame],
                   mastered: Optional[gpd.GeoDataFrame],
                   consolidated: Optional[gpd.GeoDataFrame],
                   out_dir: Path,
                   tag: str = "snapshot"):
    """
    Write what we have so far. Safe to call often.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if raw_events is not None and len(raw_events):
        # raw CSV (no geometry) + raw GPKG
        raw_csv = out_dir / f"raw_events_{tag}.csv"
        df = pd.DataFrame(raw_events.drop(columns="geometry", errors="ignore"))
        append_csv(raw_csv, df)
        append_gpkg(out_dir / f"raw_events_{tag}.gpkg", "raw_events", raw_events)

    if mastered is not None and len(mastered):
        # master CSV
        m_csv = out_dir / f"master_events_{tag}.csv"
        dfm = pd.DataFrame(mastered.drop(columns="geometry", errors="ignore"))
        append_csv(m_csv, dfm)

    if consolidated is not None and len(consolidated):
        # consolidated GPKG (single layer, append)
        c_gpkg = out_dir / f"consolidated_{tag}.gpkg"
        append_gpkg(c_gpkg, "flood_events", consolidated)

# =========================
# Date helpers
# =========================
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

def parse_hanze_date(s: str) -> Optional[pd.Timestamp]:
    if pd.isna(s):
        return None
    return pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")

def _scalar_date(y, m, d) -> pd.Timestamp:
    """Build a single Timestamp from scalar year/month/day (tolerant of NaNs/strings)."""
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

def parse_emdat_dates(row) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    sd = _scalar_date(
        row["Start Year"] if "Start Year" in row else None,
        row["Start Month"] if "Start Month" in row else None,
        row["Start Day"] if "Start Day" in row else None,
    )
    ed = _scalar_date(
        row["End Year"] if "End Year" in row else None,
        row["End Month"] if "End Month" in row else None,
        row["End Day"] if "End Day" in row else None,
    )
    if pd.isna(ed) and pd.notna(sd):
        ed = sd
    return sd, ed

# =========================
# Geometry helpers
# =========================
def reproject_buffer_km(geom, radius_km: float):
    if geom is None or geom.is_empty or radius_km is None or radius_km <= 0:
        return geom
    g = gpd.GeoSeries([geom], crs=WGS84).to_crs(WORLD_EQ_AREA)
    g = g.buffer(radius_km * 1000.0)
    return g.to_crs(WGS84).iloc[0]

def polygon_area_km2(geom) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    g = gpd.GeoSeries([geom], crs=WGS84).to_crs(WORLD_EQ_AREA)
    return float(g.area.iloc[0] / 1e6)

def circle_radius_from_area_km2(area_km2: float) -> float:
    if area_km2 is None or area_km2 <= 0:
        return 0.0
    return math.sqrt(area_km2 / math.pi)

def _keep_surface_parts(g):
    """From a (possibly) GeometryCollection keep only polygonal parts."""
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
            # last resort: chain union
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

def safe_clean(g):
    """Fix invalids using make_valid, then strip non-surface parts and buffer(0)."""
    if g is None or g.is_empty:
        return None
    try:
        if not g.is_valid:
            g = make_valid(g)
        g = _keep_surface_parts(g)
        if g is None or g.is_empty:
            return None
        # buffer(0) often fixes minor artifacts; keep tiny tolerance
        g = g.buffer(0)
        return None if (g is None or g.is_empty) else g
    except Exception:
        # last resort: try buffer(0) only
        try:
            gg = g.buffer(0)
            return None if (gg is None or gg.is_empty) else _keep_surface_parts(gg)
        except Exception:
            return None

def safe_unary_union(geoms_iterable):
    """Union a collection robustly; cleans each geom, then union with fallbacks."""
    cleaned = []
    for g in geoms_iterable:
        gg = safe_clean(g)
        if gg is not None and not gg.is_empty:
            cleaned.append(gg)
    if not cleaned:
        return None
    try:
        return unary_union(cleaned)
    except Exception:
        # fallback: incremental union with buffers if needed
        u = cleaned[0]
        for g in cleaned[1:]:
            try:
                u = u.union(g)
            except Exception:
                try:
                    u = u.buffer(0).union(g.buffer(0))
                except Exception:
                    # skip irreparable piece
                    continue
        return u

# =========================
# Tabular & admin helpers
# =========================
def read_tabular(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)

    # CSV with robust delimiter/encoding handling
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

def prep_admin(adm_path: Path, adm_layer: Optional[str], name_fields: List[str]) -> gpd.GeoDataFrame:
    if adm_path.suffix.lower() in (".gpkg", ".gdb"):
        gdf = gpd.read_file(adm_path, layer=adm_layer) if adm_layer else gpd.read_file(adm_path)
    else:
        gdf = gpd.read_file(adm_path)
    for f in name_fields:
        if f in gdf.columns:
            gdf[f"{f}__lower"] = gdf[f].astype(str).str.lower()
    return gdf.to_crs(WGS84)

def match_admin_polys(admin_gdf: gpd.GeoDataFrame, location_str: str,
                      name_fields: List[str]) -> Optional[MultiPolygon]:
    if not location_str or not isinstance(location_str, str):
        return None
    tokens = [t.strip() for t in location_str.replace("&", ",").replace("/", ",").split(",") if t.strip()]
    matches = []
    for tok in tokens:
        tok_low = tok.lower()
        mask = None
        for f in name_fields:
            col = f"{f}__lower"
            if col in admin_gdf.columns:
                m = admin_gdf[col].str.contains(tok_low, na=False)
                mask = m if mask is None else (mask | m)
        if mask is not None and mask.any():
            matches.append(unary_union(admin_gdf.loc[mask, "geometry"]))
    if not matches:
        return None
    return unary_union(matches)

# =========================
# GHSL (population) sampler
# =========================
def nearest_ghsl_epoch(year: Optional[int]) -> int:
    if year is None or pd.isna(year):
        return 2020
    return min(GHSL_EPOCHS, key=lambda e: abs(int(year) - e))

class GHSLSampler:
    """Scans --ghsl-base recursively, indexes *.tif by epoch (_EYYYY_) and bounds."""
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self._index: Dict[int, List[tuple]] = {}
        self._built = False

    def _epoch_from_name(self, path: Path) -> Optional[int]:
        for name in (path.name, path.parent.name):
            m = _EPOCH_RE.search(name)
            if m:
                return int(m.group(1))
        return None

    def _build(self):
        buckets: dict[int, list] = {}
        for tif in self.base_dir.rglob("*.tif"):
            ep = self._epoch_from_name(tif)
            if ep is None:
                continue
            try:
                with rasterio.open(tif) as src:
                    b = src.bounds
                    buckets.setdefault(ep, []).append((box(b.left, b.bottom, b.right, b.top), tif))
            except Exception:
                continue
        self._index = buckets
        self._built = True

    def sample(self, lon: float, lat: float, year: Optional[int]) -> Optional[float]:
        if not self._built:
            self._build()
        target_epoch = nearest_ghsl_epoch(year)
        if target_epoch not in self._index:
            return None
        pt = Point(lon, lat)
        for poly, path in self._index[target_epoch]:
            if poly.contains(pt):
                try:
                    with rasterio.open(path) as src:
                        for val in src.sample([(lon, lat)]):
                            v = float(val[0])
                            return None if np.isnan(v) else v
                except Exception:
                    return None
        return None

# =========================
# DEM samplers (elevation, slope)
# =========================
def find_dem_tile(base_dem_dir: Path, lon: float, lat: float) -> Optional[Path]:
    for tif in Path(base_dem_dir).rglob("*.tif"):
        try:
            with rasterio.open(tif) as src:
                b = src.bounds
                if (b.left <= lon <= b.right) and (b.bottom <= lat <= b.top):
                    return tif
        except Exception:
            continue
    return None

def sample_dem_and_slope(dem_dir: Path, lon: float, lat: float) -> Tuple[Optional[float], Optional[float]]:
    tif = find_dem_tile(dem_dir, lon, lat)
    if tif is None:
        return None, None
    with rasterio.open(tif) as src:
        row, col = src.index(lon, lat)
        w = Window(col_off=max(col - 1, 0), row_off=max(row - 1, 0), width=3, height=3)
        arr = src.read(1, window=w, boundless=True, fill_value=np.nan)
        dem = float(arr[1, 1]) if arr.shape[0] >= 2 and arr.shape[1] >= 2 else np.nan
        if np.isnan(arr).any():
            return (None if np.isnan(dem) else dem, None)
        transform = src.transform
        px_deg_x, px_deg_y = abs(transform.a), abs(transform.e)
        meters_x = 111320.0 * math.cos(math.radians(lat)) * px_deg_x
        meters_y = 110540.0 * px_deg_y
        dzdx = ((arr[1, 2] - arr[1, 0]) / (2.0 * meters_x))
        dzdy = ((arr[2, 1] - arr[0, 1]) / (2.0 * meters_y))
        slope_rad = math.atan(np.hypot(dzdx, dzdy))
        slope_deg = float(np.degrees(slope_rad))
        return (None if np.isnan(dem) else dem, slope_deg)

# =========================
# Event model & consolidation
# =========================
def normalize_event_row(geom, event_id, source, start_date, end_date,
                        country=None, area_km2=None, fatalities=None,
                        affected=None, cause=None, extra=None):
    return {
        "geometry": geom,
        "event_id": str(event_id) if event_id is not None else None,
        "source": source,
        "start_date": pd.to_datetime(start_date) if start_date is not None else None,
        "end_date": pd.to_datetime(end_date) if end_date is not None else None,
        "country": country,
        "area_km2": area_km2 if area_km2 is not None else (polygon_area_km2(geom) if geom is not None else None),
        "fatalities": fatalities,
        "people_affected": affected,
        "cause": cause,
        **(extra or {})
    }

def time_overlap(a_start, a_end, b_start, b_end, pad_days=7) -> bool:
    if a_start is None and b_start is None:
        return True
    a0 = a_start or b_start
    a1 = a_end or a0
    b0 = b_start or a_start or a1
    b1 = b_end or b0
    a0p = a0 - pd.Timedelta(days=pad_days)
    a1p = a1 + pd.Timedelta(days=pad_days)
    return not (a1p < b0 or b1 < a0p)

def override_priority(src: str) -> int:
    return {"HANZE": 4, "USFD": 4, "DFO": 2, "EMDAT": 1}.get(src.upper(), 0)

def consolidate_events(events: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    events = events.copy()
    events = events[events.geometry.notnull() & ~events.geometry.is_empty]
    events = events.reset_index(drop=True)
    keep = [True] * len(events)

    for i in range(len(events)):
        if not keep[i]:
            continue
        gi = events.geometry.iloc[i]
        si = events.source.iloc[i]
        ti0, ti1 = events.start_date.iloc[i], events.end_date.iloc[i]
        for j in range(i + 1, len(events)):
            if not keep[j]:
                continue
            gj = events.geometry.iloc[j]
            if not gi.intersects(gj):
                continue
            if not time_overlap(ti0, ti1, events.start_date.iloc[j], events.end_date.iloc[j]):
                continue
            sj = events.source.iloc[j]
            if override_priority(sj) > override_priority(si):
                keep[i] = False
                break
            elif override_priority(sj) < override_priority(si):
                keep[j] = False
            else:
                ai = events.area_km2.iloc[i] or 0
                aj = events.area_km2.iloc[j] or 0
                if aj > ai:
                    keep[i] = False
                    break
                else:
                    keep[j] = False
    return events.loc[keep].reset_index(drop=True)

# =========================
# Source-specific loaders
# =========================
def load_dfo(dfo_shp: Path, dfo_attr: Optional[Path]) -> List[dict]:
    if not dfo_shp or not dfo_shp.exists():
        return []
    gdf = gpd.read_file(dfo_shp).to_crs(WGS84)
    attrs = read_tabular(dfo_attr) if dfo_attr and dfo_attr.exists() else pd.DataFrame()
    id_cols = [c for c in attrs.columns if c.strip().lower() == "id"]
    if id_cols:
        attrs["ID"] = pd.to_numeric(attrs[id_cols[0]], errors="coerce")
        gdf = gdf.merge(attrs, on="ID", how="left")
    out: List[dict] = []
    for _, r in gdf.iterrows():
        ev_id = r.get("ID", None)
        out.append(normalize_event_row(
            geom=r.geometry,
            event_id=f"DFO_{ev_id}" if pd.notna(ev_id) else None,
            source="DFO",
            start_date=r.get("Began", None),
            end_date=r.get("Ended", None),
            country=r.get("Country", None),
            area_km2=r.get("Area", None),
            fatalities=r.get("Dead", None),
            affected=r.get("Displaced", None),
            cause=r.get("MainCause", None),
            extra={"footprint_method": "DFO_polygon"}
        ))
    return out

def _resolve_code_field(gdf: gpd.GeoDataFrame, preferred: str | None, fallback_candidates: list[str], expected_codes: list[str]) -> str:
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
    raise ValueError(
        f"Could not find a region code field in regions layer. "
        f"Tried candidates {fallback_candidates}; no overlap with event codes. "
        f"Available columns: {list(gdf.columns)}"
    )

def load_hanze_csv_to_polys(
    events_csv: Path,
    regions_v2010_path: Optional[Path],
    regions_v2021_path: Optional[Path],
    regions_layer_2010: Optional[str] = None,
    regions_layer_2021: Optional[str] = None,
    code_field_2010: Optional[str] = None,
    code_field_2021: Optional[str] = None,
) -> List[dict]:
    df = pd.read_csv(events_csv, low_memory=False)
    gdf2010 = gpd.read_file(regions_v2010_path, layer=regions_layer_2010).to_crs(WGS84) if regions_v2010_path else None
    gdf2021 = gpd.read_file(regions_v2021_path, layer=regions_layer_2021).to_crs(WGS84) if regions_v2021_path else None

    if gdf2010 is not None:
        gdf2010 = gdf2010.to_crs(WGS84)
        gdf2010["geometry"] = gdf2010.geometry.apply(safe_clean).simplify(0.0003, preserve_topology=True)

    if gdf2021 is not None:
        gdf2021 = gdf2021.to_crs(WGS84)
        gdf2021["geometry"] = gdf2021.geometry.apply(safe_clean).simplify(0.0003, preserve_topology=True)


    out: List[dict] = []
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
                geom = safe_unary_union(sel.geometry)

        if (geom is None or geom.is_empty) and isinstance(codes_2010, str) and codes_2010.strip() and gdf2010 is not None:
            codes = [c.strip() for c in codes_2010.split(";") if c.strip()]
            candidates_2010 = ["REG_CODE", "HZ2010_ID", "HZ_2010", "REG_ID", "CODE", "NUTS_ID", "ID", "GID_1", "GID_2"]
            fld_2010 = _resolve_code_field(gdf2010, code_field_2010, candidates_2010, codes)
            sel = gdf2010[gdf2010[fld_2010].astype(str).str.strip().isin(codes)]
            if len(sel):
                geom = safe_unary_union(sel.geometry)

        out.append(normalize_event_row(
            geom=geom,
            event_id=f"HANZE_{evid}",
            source="HANZE",
            start_date=sd, end_date=ed,
            country=country,
            area_km2=None if pd.isna(area) else float(area),
            fatalities=fat,
            affected=aff,
            cause=cause,
            extra={
                "footprint_method": "HANZE_regions",
                "notes": r.get("Notes"),
                "flood_source": r.get("Flood source"),
                "type": r.get("Type"),
            }
        ))
    return out

def load_usfd_v11(path_csv: Path) -> List[dict]:
    df = pd.read_csv(path_csv, low_memory=False)
    recs: List[dict] = []
    for idx, r in df.iterrows():
        sd = parse_usfd_dt(r.get("DATE_BEGIN"))
        ed = parse_usfd_dt(r.get("DATE_END"))
        lon = r.get("LON"); lat = r.get("LAT")
        geom = Point(float(lon), float(lat)) if pd.notna(lon) and pd.notna(lat) else None
        area = r.get("AREA", None)
        method = "USFD_point"
        recs.append(normalize_event_row(
            geom=geom,
            event_id=f"USFD_{idx}",
            source="USFD",
            start_date=sd, end_date=ed,
            country=r.get("COUNTRY"),
            area_km2=None if pd.isna(area) else float(area),
            fatalities=r.get("FATALITY"),
            affected=None,
            cause=r.get("CAUSE"),
            extra={
                "state": r.get("STATE"),
                "location": r.get("LOCATION"),
                "severity": r.get("SEVERITY"),
                "source_db": r.get("SOURCE_DB"),
                "source_id": r.get("SOURCE_ID"),
                "description": r.get("DESCRIPTION"),
                "footprint_method": method
            }
        ))
    return recs

def load_emdat_records(path_tabular: Path, admin_gdf: gpd.GeoDataFrame) -> List[dict]:
    """
    Read EMDAT (CSV/XLSX), keep floods, build geometry from admin names or (lon,lat),
    and return normalized event dicts.
    """
    df = read_tabular(path_tabular)
    # filter floods
    if "Disaster Type" in df.columns:
        df = df[df["Disaster Type"].astype(str).str.contains("Flood", case=False, na=False)]

    recs: List[dict] = []
    for _, row in df.iterrows():
        sd, ed = parse_emdat_dates(row)

        lon = row["Longitude"] if "Longitude" in row else None
        lat = row["Latitude"] if "Latitude" in row else None
        geom = None

        loc = row["Location"] if "Location" in row else (row["Admin Units"] if "Admin Units" in row else None)
        if pd.notna(loc):
            g = match_admin_polys(admin_gdf, str(loc), ["NAME_1", "NAME_2", "NAME_ENGLI"])
            if g and not g.is_empty:
                geom = g

        if (geom is None or geom.is_empty) and pd.notna(lon) and pd.notna(lat):
            try:
                geom = Point(float(lon), float(lat))
            except Exception:
                pass

        disno = None
        for k in ("DisNo.", "Dis No", "Dis No.", "Disaster No."):
            if k in df.columns:
                disno = row.get(k)
                break

        method = "EMDAT_admin" if (geom is not None and not isinstance(geom, Point)) else "EMDAT_point" if geom is not None else "EMDAT_missing"

        recs.append(normalize_event_row(
            geom=geom,
            event_id=f"EMDAT_{disno}" if disno is not None else None,
            source="EMDAT",
            start_date=sd, end_date=ed,
            country=row.get("Country") if "Country" in row else None,
            area_km2=None,
            fatalities=row.get("Total Deaths") if "Total Deaths" in row else None,
            affected=row.get("Total Affected") if "Total Affected" in row else None,
            cause=row.get("Disaster Subtype") if "Disaster Subtype" in row else None,
            extra={
                "river_basin": row.get("River Basin") if "River Basin" in row else None,
                "magnitude": row.get("Magnitude") if "Magnitude" in row else None,
                "footprint_method": method
            }
        ))
    return recs

# =========================
# Master IDs & EM-DAT anchor
# =========================
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0088
    from math import radians, sin, cos, asin, sqrt
    dlon = radians(lon2 - lon1); dlat = radians(lat2 - lat1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def assign_master_ids(
    gdf: gpd.GeoDataFrame,
    days_tol: int = 14,
    dist_km: float = 100.0,
    country_must_match: bool = False,
) -> gpd.GeoDataFrame:
    g = gdf.copy()

    reps = g.geometry.apply(lambda ge: ge.representative_point() if ge is not None and not ge.is_empty else None)
    g["__lon"] = reps.apply(lambda p: float(p.x) if p else np.nan)
    g["__lat"] = reps.apply(lambda p: float(p.y) if p else np.nan)

    t0 = pd.to_datetime(g["start_date"], errors="coerce")
    t1 = pd.to_datetime(g["end_date"], errors="coerce").fillna(t0)
    g["__tcenter"] = t0 + (t1 - t0) / 2

    g = g.sort_values(["__tcenter", "source"], kind="mergesort").reset_index(drop=True)

    master_ids = [-1] * len(g)
    current = 0

    for i in range(len(g)):
        if master_ids[i] != -1:
            continue
        master_ids[i] = current
        ti = g["__tcenter"].iloc[i]
        loni = g["__lon"].iloc[i]
        lati = g["__lat"].iloc[i]
        ci = str(g["country"].iloc[i]).strip().lower() if pd.notna(g["country"].iloc[i]) else None

        for j in range(i + 1, len(g)):
            if master_ids[j] != -1:
                continue
            tj = g["__tcenter"].iloc[j]
            if pd.isna(ti) or pd.isna(tj):
                continue
            if abs((tj - ti).days) > days_tol:
                if tj > ti + pd.Timedelta(days=days_tol):
                    break
                else:
                    continue
            lonj = g["__lon"].iloc[j]
            latj = g["__lat"].iloc[j]
            if np.isnan(loni) or np.isnan(lonj) or np.isnan(lati) or np.isnan(latj):
                continue
            if country_must_match:
                cj = str(g["country"].iloc[j]).strip().lower() if pd.notna(g["country"].iloc[j]) else None
                if ci and cj and ci != cj:
                    continue
            d = haversine_km(loni, lati, lonj, latj)
            if d <= dist_km:
                master_ids[j] = current
        current += 1

    g["master_id"] = master_ids

    first_times = (
        g.groupby("master_id", dropna=False)["__tcenter"]
         .min()
         .rename("tcenter_min")
         .reset_index()
         .sort_values("tcenter_min", kind="mergesort")
         .reset_index(drop=True)
    )
    first_times["rank"] = first_times.index
    g = g.merge(first_times[["master_id", "rank"]], on="master_id", how="left")
    g["master_id"] = g["rank"].astype(int)

    g = g.drop(columns=["rank", "__tcenter", "__lon", "__lat", "tcenter_min"], errors="ignore")
    return g

def add_emdat_anchor(gdf_master: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    has_emdat = gdf_master.groupby("master_id")["source"].apply(lambda s: (s == "EMDAT").any())
    has_emdat = has_emdat.rename("emdat_anchor").reset_index()
    out = gdf_master.merge(has_emdat, on="master_id", how="left")
    out["emdat_anchor"] = out["emdat_anchor"].fillna(False).astype(bool)
    return out

def filter_by_emdat_scope(gdf_master: gpd.GeoDataFrame, scope: str) -> gpd.GeoDataFrame:
    if scope == "both":
        return gdf_master
    mask = gdf_master["emdat_anchor"].astype(bool)
    if scope == "only":
        return gdf_master.loc[mask].reset_index(drop=True)
    else:  # 'non'
        return gdf_master.loc[~mask].reset_index(drop=True)

# =========================
# Enrichment
# =========================
def enrich_with_dem_ghsl(gdf: gpd.GeoDataFrame,
                         dem_dir: Optional[Path],
                         ghsl_base_dir: Optional[Path]) -> gpd.GeoDataFrame:
    ghsl = GHSLSampler(ghsl_base_dir) if ghsl_base_dir else None
    vals_dem, vals_slope, vals_pop = [], [], []
    for _, r in gdf.iterrows():
        lon, lat = (None, None)
        if r.geometry and not r.geometry.is_empty:
            p = r.geometry.representative_point()
            lon, lat = float(p.x), float(p.y)
        year = None
        if pd.notna(r.get("start_date")):
            try:
                year = int(pd.to_datetime(r["start_date"]).year)
            except Exception:
                year = None
        dem, slope, pop = (None, None, None)
        if lon is not None:
            if dem_dir:
                dem, slope = sample_dem_and_slope(dem_dir, lon, lat)
            if ghsl and year:
                pop = ghsl.sample(lon, lat, year)
        vals_dem.append(dem); vals_slope.append(slope); vals_pop.append(pop)
    gdf = gdf.copy()
    gdf["dem"] = vals_dem; gdf["slope"] = vals_slope; gdf["pop_density"] = vals_pop
    return gdf

# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="Assemble flood datasets into a consolidated GeoPackage.")

    # Inputs
    ap.add_argument("--dfo-shp", type=Path, required=False, help="DFO FloodArchive_region.shp")
    ap.add_argument("--dfo-attr", type=Path, required=False, help="DFO attributes (xlsx/csv)")

    ap.add_argument("--hanze-csv", type=Path, required=False, help="HANZE events table (CSV)")
    ap.add_argument("--hanze-regions-2010", type=Path, required=False, help="HANZE Regions v2010 (shp/gpkg)")
    ap.add_argument("--hanze-regions-2010-layer", type=str, required=False, help="Layer name if gpkg/gdb")
    ap.add_argument("--hanze-regions-2010-code", type=str, default="REG_CODE", help="Region code field v2010")

    ap.add_argument("--hanze-regions-2021", type=Path, required=False, help="HANZE Regions v2021 (shp/gpkg)")
    ap.add_argument("--hanze-regions-2021-layer", type=str, required=False, help="Layer name if gpkg/gdb")
    ap.add_argument("--hanze-regions-2021-code", type=str, default="REG_CODE", help="Region code field v2021")

    ap.add_argument("--usfd-csv", type=Path, required=False, help="USFD_v1.1.csv")
    ap.add_argument("--emdat", type=Path, required=False, help="EM-DAT export (xlsx/csv)")
    ap.add_argument("--adm-shp", type=Path, required=True, help="Admin boundaries (GADM gpkg/gdb/shp)")
    ap.add_argument("--adm-layer", type=str, required=False, help="Layer name if using GPKG/GDB")
    ap.add_argument("--adm-name-fields", type=str, nargs="+",
                    default=["NAME_1", "NAME_2", "NAME_ENGLI"], help="Admin name fields for matching")
    ap.add_argument("--ghsl-base", type=Path, required=False, help="GHSL base dir (scanned recursively)")
    ap.add_argument("--dem-dir", type=Path, required=False, help="DEM root folder")
    ap.add_argument("--stage-dir", type=Path, required=False,
                help="Folder to write intermediate outputs (sources, enriched, master-scoped).")
    ap.add_argument("--verbose", action="store_true",
                help="Print extra progress messages.")

    ap.add_argument("--log-every", type=int, default=1000, help="Print progress every N rows (per source)")
    ap.add_argument("--flush-every", type=int, default=0, help="Snapshot every N total rows (0=disabled)")
    ap.add_argument("--snapshot-dir", type=Path, default=output_path("_snapshots"), help="Where to write snapshots")
    ap.add_argument("--checkpoint", type=Path, help="Resume/record simple counters (json)")


    # Outputs
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output GeoPackage path")
    ap.add_argument("--master-csv", type=Path, required=False,
                    help="External CSV of all source events with master_id (default: disaster_output/master_events.csv)")

    # Master-ID clustering params
    ap.add_argument("--master-days", type=int, default=14, help="Time tolerance in days for clustering")
    ap.add_argument("--master-dist-km", type=float, default=100.0, help="Distance tolerance (km) for clustering")
    ap.add_argument("--master-country-match", action="store_true", help="Require same country to cluster")

    # EM-DAT scoping
    ap.add_argument("--emdat-scope", choices=["both", "only", "non"], default="both",
                    help="Cluster-level filter: 'only' keep clusters with EM-DAT, "
                         "'non' keep clusters without EM-DAT, 'both' keep all (default)")

    args = ap.parse_args()

    # Admin boundaries
    admin_gdf = prep_admin(args.adm_shp, args.adm_layer, args.adm_name_fields)

    # Load sources
    records: List[dict] = []

    if args.dfo_shp:
        records += load_dfo(args.dfo_shp, args.dfo_attr)
    if args.usfd_csv and args.usfd_csv.exists():
        records += load_usfd_v11(args.usfd_csv)
    if args.hanze_csv and args.hanze_csv.exists():
        records += load_hanze_csv_to_polys(
            events_csv=args.hanze_csv,
            regions_v2010_path=args.hanze_regions_2010,
            regions_v2021_path=args.hanze_regions_2021,
            regions_layer_2010=args.hanze_regions_2010_layer,
            regions_layer_2021=args.hanze_regions_2021_layer,
            code_field_2010=args.hanze_regions_2010_code,
            code_field_2021=args.hanze_regions_2021_code,
        )
    if args.emdat and args.emdat.exists():
        records += load_emdat_records(args.emdat, admin_gdf)

    if not records:
        raise SystemExit("No records ingested. Check your input paths/filters.")

    gdf = gpd.GeoDataFrame(records, crs=WGS84)

    vlog(args.verbose, f"[STAGE] Loaded records: {len(gdf)}  (DFO={ (gdf.source=='DFO').sum() }," 
                   f" HANZE={ (gdf.source=='HANZE').sum() }, USFD={ (gdf.source=='USFD').sum() },"
                   f" EMDAT={ (gdf.source=='EMDAT').sum() })")

    if args.stage_dir:
        args.stage_dir.mkdir(parents=True, exist_ok=True)

        # All sources in one file
        (args.stage_dir / "sources_loaded.gpkg").unlink(missing_ok=True)
        gdf.to_file(args.stage_dir / "sources_loaded.gpkg", layer="all_sources", driver="GPKG")

        # One layer per source for quick peeking
        staged = args.stage_dir / "sources_by_source.gpkg"
        if staged.exists():
            staged.unlink()
        for src in ["DFO", "HANZE", "USFD", "EMDAT"]:
            sub = gdf[gdf["source"] == src]
            if len(sub):
                sub.to_file(staged, layer=src.lower(), driver="GPKG",
                            mode="w" if not staged.exists() else "a")
        vlog(args.verbose, f"[WRITE] Stage: sources -> {args.stage_dir}")

    # Enrich (optional)
    gdf = enrich_with_dem_ghsl(gdf,
                               dem_dir=args.dem_dir if args.dem_dir else None,
                               ghsl_base_dir=args.ghsl_base if args.ghsl_base else None)
    
    vlog(args.verbose, "[STAGE] Enriched with DEM/GHSL")
    if args.stage_dir:
        (args.stage_dir / "enriched.gpkg").unlink(missing_ok=True)
        gdf.to_file(args.stage_dir / "enriched.gpkg", layer="enriched", driver="GPKG")

    # ---------- MASTER (all events) ----------
    gdf_master = assign_master_ids(
        gdf,
        days_tol=args.master_days,
        dist_km=args.master_dist_km,
        country_must_match=args.master_country_match
    )

    # Mark emdat_anchor (cluster has any EMDAT)
    gdf_master = add_emdat_anchor(gdf_master)

    # Apply EM-DAT scoping at cluster level
    gdf_master = filter_by_emdat_scope(gdf_master, args.emdat_scope)
    if gdf_master.empty:
        print(f"[WARN] No events remain after --emdat-scope={args.emdat_scope}.")

    vlog(args.verbose, f"[STAGE] Master rows after scope={args.emdat_scope}: {len(gdf_master)}")
    if args.stage_dir and len(gdf_master):
        (args.stage_dir / "master_scoped.gpkg").unlink(missing_ok=True)
        gdf_master.to_file(args.stage_dir / "master_scoped.gpkg", layer="master_scoped", driver="GPKG")

    # Write MASTER CSV (flat; no geometry) with centroids
    reps = gdf_master.geometry.apply(lambda ge: ge.representative_point() if ge is not None and not ge.is_empty else None)
    gdf_master["centroid_lon"] = reps.apply(lambda p: float(p.x) if p else np.nan)
    gdf_master["centroid_lat"] = reps.apply(lambda p: float(p.y) if p else np.nan)

    master_csv_path = args.master_csv or output_path("master_events.csv")
    master_csv_path.parent.mkdir(parents=True, exist_ok=True)

    cols_for_csv = [
        "master_id", "emdat_anchor",
        "event_id", "source", "start_date", "end_date", "country",
        "area_km2", "fatalities", "people_affected", "cause",
        "footprint_method", "dem", "slope", "pop_density",
        "centroid_lon", "centroid_lat"
    ]
    cols_for_csv = [c for c in cols_for_csv if c in gdf_master.columns]
    gdf_master[cols_for_csv].to_csv(master_csv_path, index=False)
    print(f"Wrote master events CSV with {len(gdf_master)} rows -> {master_csv_path}")

    # ---------- CONSOLIDATED (apply overrides on the scoped set) ----------
    consolidated = consolidate_events(gdf_master)

    # Keep master_id and emdat_anchor in the output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    consolidated.to_file(args.output, layer="flood_events", driver="GPKG")
    print(f"Wrote {len(consolidated)} consolidated events to {args.output} (layer=flood_events)")

if __name__ == "__main__":
    main()
