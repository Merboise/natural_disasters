# natural_disasters/floods.py
# -*- coding: utf-8 -*-
"""
Flood hazard ingestion for the unified pipeline.

Sources supported here (no EM-DAT loading; EM-DAT filtering and canonical schema mapping happen in main.py):
- DFO polygons (+ optional attributes)
- HANZE (EU) polygons via regions v2010/v2021 lookups from an events CSV
- USFD (US flood database) point events (kept as points; can be buffered later if desired)

This module returns a GeoDataFrame of raw flood records; main.py handles canonicalization.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid
from shapely.ops import unary_union
from .helpers import (haversine_km, _centroid_latlon, _safe_clean, _safe_unary_union, data_path, unify_schema_floods)
import warnings

# ---------------- Constants / CRS ----------------
WGS84 = "EPSG:4326"
WORLD_EQ_AREA = "EPSG:6933"  # Equal-area meters (World Cylindrical Equal Area)

# Merge/primary selection priority: USFD > HANZE > DFO (EM-DAT enrich only)
FLOOD_SOURCE_PRIORITY = ["USFD", "HANZE", "DFO"]
DFO_SEVERITY_DESC  = "DFO severity (source-defined ordinal impact indicator; larger value indicates greater severity per DFO metadata)."
USFD_SEVERITY_DESC = "USFD severity (source-defined report severity code per NOAA report metadata)."

# ---------------- Geometry presence mask (no GeoSeries.notna() alone) ----------------
try:
    # Prefer the shared helper if available (uses: (~s.is_empty) & s.notna())
    from .helpers import _geom_present_mask  # type: ignore
except Exception:
    def _geom_present_mask(obj) -> pd.Series:
        """
        Recommended mask for geometry presence under newer GeoPandas:
        (~is_empty) & notna() — avoids relying on notna() semantics alone.
        Accepts GeoSeries or GeoDataFrame.
        """
        s = obj.geometry if isinstance(obj, gpd.GeoDataFrame) else obj
        return (~s.is_empty) & s.notna()

# ---------------- Utilities ----------------
def _get_ci(r, *names):
    for n in names:
        if n in r:
            return r.get(n)
    # try case variants
    for n in names:
        ln = n.lower()
        for c in r.index:
            if str(c).lower() == ln:
                return r.get(c)
    return None

def enrich_floods(
    floods_raw: gpd.GeoDataFrame,
    emdat_flood_df: Optional[pd.DataFrame] = None,
    time_tolerance_days: int = 14,
    proximity_km: float = 200.0,
) -> gpd.GeoDataFrame:
    """
    Merge overlapping flood mentions across USFD/HANZE/DFO using priority
    USFD > HANZE > DFO. EM-DAT is used only to enrich attributes, never to
    override geometry or severity. Polygons are unioned when possible.
    Also preserves *all* source-prefixed extras and the exact raw columns.
    """
    if floods_raw is None or floods_raw.empty:
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)

    # Non-destructive canonicalization (adds start_time/end_time if missing; preserves extras)
    g = unify_schema_floods(floods_raw.copy())

    # cluster helpers
    clat, clon = _centroid_latlon(g)
    g["__lat"] = clat
    g["__lon"] = clon
    g = g.sort_values("start_time").reset_index(drop=True)

    def _pick_primary(df_cluster: pd.DataFrame) -> pd.Series:
        for src in FLOOD_SOURCE_PRIORITY:  # expected: ["USFD","HANZE","DFO"]
            m = (df_cluster["source"] == src)
            if m.any():
                return df_cluster[m].iloc[0]
        return df_cluster.iloc[0]

    def _union_polys(geoms):
        polys = []
        for gg in geoms:
            if gg is None or getattr(gg, "is_empty", True):
                continue
            if getattr(gg, "geom_type", "") in ("Polygon", "MultiPolygon"):
                polys.append(_safe_clean(gg))
        return _safe_unary_union(polys) if polys else None

    # ----- greedy time+space clustering -----
    clusters, meta = [], []  # meta: (start_min, end_max, mlat, mlon, n)
    for i, r in g.iterrows():
        st, et, lat, lon = r["start_time"], r["end_time"], r["__lat"], r["__lon"]
        placed = False
        for cid, (smin, smax, mlat, mlon, n) in enumerate(meta):
            if pd.isna(st) or pd.isna(et):
                continue
            # temporal overlap ± tolerance
            if max(st, smin - pd.Timedelta(days=time_tolerance_days)) <= min(et, smax + pd.Timedelta(days=time_tolerance_days)):
                # spatial proximity
                d = haversine_km(lat, lon, mlat, mlon)
                if np.isfinite(d) and d <= proximity_km:
                    clusters[cid].append(i)
                    new_n = n + 1
                    meta[cid] = (
                        min(smin, st), max(smax, et),
                        (mlat * n + (lat if np.isfinite(lat) else mlat)) / new_n,
                        (mlon * n + (lon if np.isfinite(lon) else mlon)) / new_n,
                        new_n,
                    )
                    placed = True
                    break
        if not placed:
            clusters.append([i])
            meta.append((st, et, (lat if np.isfinite(lat) else np.nan), (lon if np.isfinite(lon) else np.nan), 1))

    # ----- merge per cluster -----
    rows = []
    for cid, idxs in enumerate(clusters):
        part = g.loc[idxs].copy()
        primary = _pick_primary(part)
        sources = part["source"].dropna().astype(str).unique().tolist()

        # geometry: union polygons if present, else primary geometry
        union_geom = _union_polys(part.geometry.tolist())
        geom = union_geom if union_geom is not None else primary.geometry

        # cluster time window
        start_min = pd.to_datetime(part["start_time"]).min()
        end_max   = pd.to_datetime(part["end_time"]).max()
        if pd.isna(end_max) and pd.notna(start_min):
            end_max = start_min

        # area from union if polygonal
        area = _area_km2(geom)

        merged = {
            "event_id": primary.get("event_id") or f"FCL_{cid}",
            "event_type": "flood",
            "start_time": start_min,
            "end_time": end_max,
            "band": None,
            "geom_method": ("merged_union" if union_geom is not None else (primary.get("geom_method") or primary.get("footprint_method"))),
            "geom_confidence": None,
            "area_km2": area,
            "geometry": geom,
            "sources": "|".join(sorted(sources)),
            "source_event_ids": "|".join(
                f"{r.source}:{r.event_id}"
                for _, r in part.iterrows()
                if pd.notna(r.get("source")) and pd.notna(r.get("event_id"))
            ),
        }

        # ---------- carry through ALL source-prefixed extras (first non-null) ----------
        for prefix in ("usfd_", "hanze_", "dfo_"):
            src_cols = [c for c in part.columns if isinstance(c, str) and c.startswith(prefix)]
            for c in src_cols:
                if c not in merged or pd.isna(merged.get(c)):
                    vals = part[c].dropna()
                    if not vals.empty:
                        merged[c] = vals.iloc[0]

        # ---------- also carry the exact raw column names you asked to keep ----------
        raw_like = [
            # DFO raw names
            "ID","GlideNumber","Country","OtherCountry","long","lat","Area","Began","Ended",
            "Validation","Dead","Displaced","MainCause","Severity",
            # HANZE raw names
            "Country code","Year","Country name","Start date","End date","Type","Flood source",
            "Regions affected (v2010)","Regions affected (v2021)","Area flooded","Fatalities",
            "Persons affected","Losses (nominal value)","Losses (original currency)","Losses (2020 euro)",
            "Cause","Notes","References","Changes",
            # USFD raw names
            "LOCATION","AREA","FATALITY","DAMAGE","SEVERITY","SOURCE","CAUSE","SOURCE_DB","SOURCE_ID",
            "DESCRIPTION","slope","dem","LULC","DISTANT_RIVER","CONT_AREA","DEPTH","year",
        ]
        for c in raw_like:
            if c in part.columns and (c not in merged or pd.isna(merged.get(c))):
                vals = part[c].dropna()
                if not vals.empty:
                    merged[c] = vals.iloc[0]

        # ---------- severity by priority: USFD → DFO (HANZE typically none) ----------
        sev_val, sev_note = None, None
        if pd.notna(merged.get("usfd_severity")):
            sev_val = merged.get("usfd_severity"); sev_note = USFD_SEVERITY_DESC
        elif pd.notna(merged.get("dfo_severity")):
            sev_val = merged.get("dfo_severity");  sev_note = DFO_SEVERITY_DESC
        merged["severity"] = sev_val
        merged["severity_notes"] = sev_note

        # harmonized human-impact helpers (keep source-prefixed originals too)
        def _pick_first(*names):
            for nm in names:
                v = merged.get(nm)
                if pd.notna(v):
                    return v
            return None
        merged["fatalities_any"]      = _pick_first("fatalities", "usfd_fatality", "dfo_dead", "hanze_fatalities")
        merged["people_affected_any"] = _pick_first("people_affected", "dfo_displaced", "hanze_persons_affected")

        # ---------- optional EM-DAT enrichment (no override of geometry/severity) ----------
        if emdat_flood_df is not None and not emdat_flood_df.empty:
            em = emdat_flood_df.copy()
            em["start_date"] = pd.to_datetime(em.get("start_date"), errors="coerce")
            em["end_date"]   = pd.to_datetime(em.get("end_date"), errors="coerce")
            temporal = (em["start_date"] <= end_max) & (em["end_date"] >= start_min)
            cand = em.loc[temporal].copy()

            if not cand.empty:
                # centroid of the cluster
                latc = np.nanmean(part["__lat"])
                lonc = np.nanmean(part["__lon"])

                def _best_within(sub: pd.DataFrame) -> Optional[pd.Series]:
                    if "em_lat" in sub.columns and "em_lon" in sub.columns:
                        sub2 = sub.dropna(subset=["em_lat","em_lon"])
                        if not sub2.empty and np.isfinite(latc) and np.isfinite(lonc):
                            d = haversine_km(latc, lonc, sub2["em_lat"].values, sub2["em_lon"].values)
                            j = int(np.nanargmin(d)) if len(d) else None
                            return sub2.iloc[j] if j is not None else None
                    # Fallback: just pick first temporal match
                    return sub.iloc[0]

                closest = _best_within(cand)
                if closest is not None:
                    # normalized convenience fields
                    merged["emdat_event_id"] = closest.get("event_id")
                    merged["emdat_iso3"]     = closest.get("iso3")
                    merged["emdat_start"]    = closest.get("start_date")
                    merged["emdat_end"]      = closest.get("end_date")
                    merged["emdat_lat"]      = closest.get("em_lat") if "em_lat" in closest.index else closest.get("Latitude")
                    merged["emdat_lon"]      = closest.get("em_lon") if "em_lon" in closest.index else closest.get("Longitude")

                    # helper to pull any of several alternative column spellings
                    def _gcol(row, *opts):
                        for o in opts:
                            if o in row.index:
                                return row.get(o)
                        return None

                    # preferred numeric aggregations (keep originals below as raw)
                    merged["emdat_total_deaths"]   = _gcol(closest, "Total Deaths","total_deaths","deaths")
                    merged["emdat_total_affected"] = _gcol(closest, "Total Affected","No. Affected","affected")
                    merged["emdat_total_damage_usd_000s"] = _gcol(
                        closest, "Total Damage ('000 US$)","total_damage_usd_000s","damage_000_usd"
                    )

                    # also copy the **exact EM-DAT raw columns** you listed
                    emdat_raw_keep = [
                        "Start Year","Start Month","Start Day",
                        "End Year","End Month","End Day",
                        "Total Deaths","No. Injured","No. Affected","No. Homeless","Total Affected",
                        "Reconstruction Costs ('000 US$)","Reconstruction Costs, Adjusted ('000 US$)",
                        "Insured Damage ('000 US$)","Insured Damage, Adjusted ('000 US$)",
                        "Total Damage ('000 US$)","Total Damage, Adjusted ('000 US$)",
                        "CPI","Admin Units","Entry Date","Last Update","Latitude","Longitude","River Basin"
                    ]
                    for col in emdat_raw_keep:
                        if col in closest.index and (col not in merged or pd.isna(merged.get(col))):
                            merged[col] = closest.get(col)

                    # If the EM-DAT record only had Y/M/D parts, ensure emdat_start/end exist
                    # (process_emdat_data already builds start_date/end_date, but we’re defensive)
                    if pd.isna(merged.get("emdat_start")):
                        sy, sm, sd = merged.get("Start Year"), merged.get("Start Month"), merged.get("Start Day")
                        try:
                            if pd.notna(sy):
                                sm = int(sm) if pd.notna(sm) else 1
                                sd = int(sd) if pd.notna(sd) else 1
                                merged["emdat_start"] = pd.Timestamp(int(sy), sm, sd)
                        except Exception:
                            pass
                    if pd.isna(merged.get("emdat_end")):
                        ey, em_, ed_ = merged.get("End Year"), merged.get("End Month"), merged.get("End Day")
                        try:
                            if pd.notna(ey):
                                em_ = int(em_) if pd.notna(em_) else 1
                                ed_ = int(ed_) if pd.notna(ed_) else 1
                                merged["emdat_end"] = pd.Timestamp(int(ey), em_, ed_)
                        except Exception:
                            pass

        rows.append(merged)

    out = gpd.GeoDataFrame(rows, crs=WGS84)

    # recompute any missing area_km2 (equal-area calc) if geometry exists
    if "area_km2" in out.columns:
        m = out["area_km2"].isna()
        if m.any():
            out.loc[m, "area_km2"] = out.loc[m, "geometry"].apply(_area_km2)

    # final canonical order (non-destructive; preserves all extras/raws)
    out = unify_schema_floods(out)
    return out

def _ensure_cols(gdf: gpd.GeoDataFrame, cols: List[str]) -> gpd.GeoDataFrame:
    for c in cols:
        if c not in gdf.columns:
            gdf[c] = None
    return gdf

def _area_km2(geom) -> Optional[float]:
    # Kept for convenience; not used by this loader (canonicalization happens in main/helpers)
    try:
        if geom is None or geom.is_empty:
            return None
        return gpd.GeoSeries([geom], crs=WGS84).to_crs(WORLD_EQ_AREA).area.iloc[0] / 1e6
    except Exception:
        return None

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

    # Handle scientific notation like 1.9961E+11
    if "e" in s.lower():
        try:
            s = str(int(float(s)))   # e.g., 199610000000
        except Exception:
            return None

    try:
        n = len(s)
        if n >= 14:
            # YYYYMMDDHHMMSS (take first 14)
            return pd.to_datetime(s[:14], format="%Y%m%d%H%M%S", errors="coerce")
        elif n in (12, 13):
            # YYYYMMDDHHMM (pad seconds)
            return pd.to_datetime(s[:12] + "00", format="%Y%m%d%H%M%S", errors="coerce")
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
    DFO polygons + optional attributes. Keeps raw DFO columns and adds dfo_* copies.
    """
    if not dfo_shp or not Path(dfo_shp).exists():
        return []
    gdf = gpd.read_file(dfo_shp).to_crs(WGS84)

    # optional simplify (safe)
    if simplify_tolerance_deg and simplify_tolerance_deg > 0:
        gdf["geometry"] = gdf.geometry.apply(_safe_clean).simplify(simplify_tolerance_deg, preserve_topology=True)

    # join attributes if provided
    attrs = _read_tabular(Path(dfo_attr)) if dfo_attr and Path(dfo_attr).exists() else pd.DataFrame()
    if not attrs.empty:
        # robust match on "ID"
        id_cols = [c for c in attrs.columns if str(c).strip().lower() == "id"]
        if id_cols:
            attrs["ID"] = pd.to_numeric(attrs[id_cols[0]], errors="coerce")
            gdf = gdf.merge(attrs, on="ID", how="left")

    out = []
    for _, r in gdf.iterrows():
        # case-insensitive getters
        def ci(*names): return _get_ci(r, *names)

        # ---- raw DFO fields you asked to keep (exact names) ----
        raw = {
            "ID":            ci("ID"),
            "GlideNumber":   ci("GLIDENUMBE","GLIDE_NUMBER","GlideNumber"),
            "Country":       ci("COUNTRY","Country"),
            "OtherCountry":  ci("OTHERCOUNT","OtherCountry"),
            "long":          pd.to_numeric(ci("LONG","Lon","LON","long"), errors="coerce"),
            "lat":           pd.to_numeric(ci("LAT","Lat","lat"), errors="coerce"),
            "Area":          ci("AREA","Area"),
            "Began":         ci("BEGAN","Began"),
            "Ended":         ci("ENDED","Ended"),
            "Validation":    ci("VALIDATION","Validation"),
            "Dead":          ci("DEAD","Dead"),
            "Displaced":     ci("DISPLACED","Displaced"),
            "MainCause":     ci("MAINCAUSE","MainCause"),
            "Severity":      ci("SEVERITY","Severity"),
        }

        # canonical
        began = pd.to_datetime(raw["Began"], errors="coerce")
        ended = pd.to_datetime(raw["Ended"], errors="coerce")
        area  = pd.to_numeric(raw["Area"], errors="coerce")

        row = {
            "geometry": r.geometry,
            "event_id": f"DFO_{raw['ID']}" if pd.notna(raw["ID"]) else None,
            "source": "DFO",
            "start_date": began,
            "end_date":   ended,
            "country": raw["Country"],
            "area_km2": None if pd.isna(area) else float(area),
            "fatalities": raw["Dead"],
            "people_affected": raw["Displaced"],
            "cause": raw["MainCause"],
            "footprint_method": "DFO_polygon",
            # prefixed duplicates, safe for merged rows
            "dfo_id": raw["ID"],
            "dfo_glide": raw["GlideNumber"],
            "dfo_country": raw["Country"],
            "dfo_othercountry": raw["OtherCountry"],
            "dfo_lon": raw["long"],
            "dfo_lat": raw["lat"],
            "dfo_area": raw["Area"],
            "dfo_began": raw["Began"],
            "dfo_ended": raw["Ended"],
            "dfo_validation": raw["Validation"],
            "dfo_dead": raw["Dead"],
            "dfo_displaced": raw["Displaced"],
            "dfo_maincause": raw["MainCause"],
            "dfo_severity": raw["Severity"],
        }

        # also put the exact raw names on the record (so your CSV has them verbatim)
        row.update(raw)
        out.append(row)
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
    for g in (gdf2010, gdf2021):
        if g is not None and simplify_tolerance_deg:
            g["geometry"] = g.geometry.apply(_safe_clean).simplify(simplify_tolerance_deg, preserve_topology=True)

    out: list[dict] = []
    for _, r in df.iterrows():
        # raw names exactly as in your list
        raw = {
            "ID": r.get("ID"),
            "Country code": r.get("Country code"),
            "Year": r.get("Year"),
            "Country name": r.get("Country name"),
            "Start date": r.get("Start date"),
            "End date": r.get("End date"),
            "Type": r.get("Type"),
            "Flood source": r.get("Flood source"),
            "Regions affected (v2010)": r.get("Regions affected (v2010)"),
            "Regions affected (v2021)": r.get("Regions affected (v2021)"),
            "Area flooded": r.get("Area flooded"),
            "Fatalities": r.get("Fatalities"),
            "Persons affected": r.get("Persons affected"),
            "Losses (nominal value)": r.get("Losses (nominal value)"),
            "Losses (original currency)": r.get("Losses (original currency)"),
            "Losses (2020 euro)": r.get("Losses (2020 euro)"),
            "Cause": r.get("Cause"),
            "Notes": r.get("Notes"),
            "References": r.get("References"),
            "Changes": r.get("Changes"),
        }

        # geometry via region codes
        geom = None
        reg2021 = raw["Regions affected (v2021)"]
        reg2010 = raw["Regions affected (v2010)"]

        if isinstance(reg2021, str) and reg2021.strip() and gdf2021 is not None:
            codes = [c.strip() for c in reg2021.split(";") if c.strip()]
            fld_2021 = _resolve_code_field(gdf2021, code_field_2021,
                                           ["REG_CODE","HZ2021_ID","HZ_2021","REG_ID","CODE","NUTS_ID","ID","GID_1","GID_2"],
                                           codes)
            sel = gdf2021[gdf2021[fld_2021].astype(str).str.strip().isin(codes)]
            if len(sel):
                geom = _safe_unary_union(sel.geometry)

        if (geom is None or getattr(geom, "is_empty", True)) and isinstance(reg2010, str) and reg2010.strip() and gdf2010 is not None:
            codes = [c.strip() for c in reg2010.split(";") if c.strip()]
            fld_2010 = _resolve_code_field(gdf2010, code_field_2010,
                                           ["REG_CODE","HZ2010_ID","HZ_2010","REG_ID","CODE","NUTS_ID","ID","GID_1","GID_2"],
                                           codes)
            sel = gdf2010[gdf2010[fld_2010].astype(str).str.strip().isin(codes)]
            if len(sel):
                geom = _safe_unary_union(sel.geometry)

        # canonical + prefixed copies
        sd = parse_hanze_date(raw["Start date"])
        ed = parse_hanze_date(raw["End date"])
        area = pd.to_numeric(raw["Area flooded"], errors="coerce")

        row = {
            "geometry": geom,
            "event_id": f"HANZE_{raw['ID']}",
            "source": "HANZE",
            "start_date": sd,
            "end_date": ed,
            "country": raw["Country name"],
            "area_km2": None if pd.isna(area) else float(area),
            "fatalities": raw["Fatalities"],
            "people_affected": raw["Persons affected"],
            "cause": raw["Cause"],
            "footprint_method": "HANZE_regions",
        }

        # include raw names exactly + prefixed
        row.update(raw)
        for k, v in raw.items():
            row["hanze_" + k.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "").replace("'", "")] = v

        out.append(row)
    return out

def load_usfd_v11(path_csv: Path | None) -> list[dict]:
    if not path_csv or not Path(path_csv).exists():
        return []
    df = pd.read_csv(path_csv, low_memory=False)
    recs: list[dict] = []

    for idx, r in df.iterrows():
        # raw names exactly as you provided
        raw = {
            "ID": r.get("ID"),
            "DATE_BEGIN": r.get("DATE_BEGIN"),
            "DATE_END": r.get("DATE_END"),
            "DURATION": r.get("DURATION"),
            "LON": pd.to_numeric(r.get("LON"), errors="coerce"),
            "LAT": pd.to_numeric(r.get("LAT"), errors="coerce"),
            "COUNTRY": r.get("COUNTRY"),
            "STATE": r.get("STATE"),
            "LOCATION": r.get("LOCATION"),
            "AREA": r.get("AREA"),
            "FATALITY": r.get("FATALITY"),
            "DAMAGE": r.get("DAMAGE"),
            "SEVERITY": r.get("SEVERITY"),
            "SOURCE": r.get("SOURCE"),
            "CAUSE": r.get("CAUSE"),
            "SOURCE_DB": r.get("SOURCE_DB"),
            "SOURCE_ID": r.get("SOURCE_ID"),
            "DESCRIPTION": r.get("DESCRIPTION"),
            "slope": r.get("slope"),
            "dem": r.get("dem"),
            "LULC": r.get("LULC"),
            "DISTANT_RIVER": r.get("DISTANT_RIVER"),
            "CONT_AREA": r.get("CONT_AREA"),
            "DEPTH": r.get("DEPTH"),
            "year": r.get("year"),
        }

        sd = parse_usfd_dt(raw["DATE_BEGIN"])
        ed = parse_usfd_dt(raw["DATE_END"])
        lon, lat = raw["LON"], raw["LAT"]
        geom = Point(float(lon), float(lat)) if np.isfinite(lon) and np.isfinite(lat) and (-180 <= lon <= 180) and (-90 <= lat <= 90) else None
        area = pd.to_numeric(raw["AREA"], errors="coerce")

        row = {
            "geometry": geom,
            "event_id": f"USFD_{idx}",
            "source": "USFD",
            "start_date": sd,
            "end_date": ed,
            "country": raw["COUNTRY"],
            "area_km2": None if pd.isna(area) else float(area),
            "fatalities": raw["FATALITY"],
            "people_affected": None,
            "cause": raw["CAUSE"],
            "footprint_method": "USFD_point",
        }

        # include raw + prefixed
        row.update(raw)
        for k, v in raw.items():
            row["usfd_" + k.lower()] = v

        recs.append(row)
    return recs


# ---------------- Public entry point ----------------
def process_flood_data(
    dfo_shp: str | None = data_path("DFO/FloodArchive_region.shp"),
    dfo_attr: str | None = data_path("DFO/FloodArchive_attributes.csv"),
    hanze_csv: str | None = data_path("HANZE/HANZE_events.csv"),
    hanze_regions_2010: str | Path | None = data_path("HANZE/Regions_v2010_simplified/Regions_v2010_simplified.shp"),
    hanze_regions_2021: str | Path | None = data_path("HANZE/Regions_v2021_simplified/Regions_v2021_simplified.shp"),
    hanze_regions_2010_layer: Optional[str] = None,
    hanze_regions_2021_layer: Optional[str] = None,
    hanze_code_field_2010: Optional[str] = None,
    hanze_code_field_2021: Optional[str] = None,
    usfd_csv: str | Path | None = data_path("USFD/USFD_v1.1.csv"),
    simplify_tolerance_deg: float | None = 0.0003,
) -> gpd.GeoDataFrame:
    """
    Load flood sources (DFO + HANZE + USFD) and return a raw GeoDataFrame.
    Canonicalization (event_id, event_type, start_time, end_time, band, geom_method,
    geom_confidence, area_km2) is applied in main.py via helpers.unify_schema_floods().

    Returns: GeoDataFrame; CRS=EPSG:4326
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
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)

    gdf = gpd.GeoDataFrame(recs, crs=WGS84)

    # Normalize raw date fields (keep as *_date; canonicalization happens in main/helpers)
    gdf["start_date"] = pd.to_datetime(gdf.get("start_date"), errors="coerce")
    gdf["end_date"]   = pd.to_datetime(gdf.get("end_date"),   errors="coerce")

    # Optional: keep only rows with present geometry; comment out if you prefer to retain geometry-less records
    # gdf = gdf[_geom_present_mask(gdf)].copy()

    return gdf

# ---- CLI (drop-in) ---------------------------------------------------------
if __name__ == "__main__":
    import argparse, logging
    import warnings
    from pathlib import Path
    from .helpers import setup_logging, output_path, write_single_hazard_gdb

    # Silence the noisy notna() warning coming from third-party writers
    warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)

    ap = argparse.ArgumentParser(
        description="Build a raw flood dataset (DFO + HANZE + USFD) and write CSV/GPKG."
    )
    # Inputs
    ap.add_argument("--dfo-shp", type=Path, help="DFO FloodArchive_region.shp")
    ap.add_argument("--dfo-attr", type=Path, help="DFO attributes (xlsx/csv)")
    ap.add_argument("--hanze-csv", type=Path, help="HANZE events table (CSV)")
    ap.add_argument("--hanze-regions-2010", type=Path, help="HANZE Regions v2010 (shp/gpkg)")
    ap.add_argument("--hanze-regions-2010-layer", type=str, help="Layer name if gpkg/gdb")
    ap.add_argument("--hanze-regions-2010-code", type=str, help="Region code field v2010 (optional)")
    ap.add_argument("--hanze-regions-2021", type=Path, help="HANZE Regions v2021 (shp/gpkg)")
    ap.add_argument("--hanze-regions-2021-layer", type=str, help="Layer name if gpkg/gdb")
    ap.add_argument("--hanze-regions-2021-code", type=str, help="Region code field v2021 (optional)")
    ap.add_argument("--usfd-csv", type=Path, help="USFD_v1.1.csv")

    ap.add_argument("--simplify-deg", type=float, default=0.0003,
                    help="Polygon simplification tolerance in degrees (0.0003 default; 0 to disable).")

    # Outputs
    ap.add_argument("--out-gpkg", type=Path, default=output_path("floods.gpkg"),
                    help="Output GeoPackage path (default: disaster_output/floods.gpkg)")
    ap.add_argument("--out-layer", type=str, default="floods",
                    help="Output layer name inside the GeoPackage (default: floods)")
    ap.add_argument("--out-csv", type=Path, default=output_path("floods.csv"),
                    help="Attributes-only CSV path (default: disaster_output/floods.csv)")

    # Logging
    ap.add_argument("--output-folder", type=Path, default=output_path(""),
                    help="Base folder for logs/aux outputs (default: disaster_output)")
    ap.add_argument("--log-level", type=str, choices=["DEBUG","INFO","WARNING","ERROR"], default="INFO")

    args = ap.parse_args()

    setup_logging(str(args.output_folder), level=getattr(logging, args.log_level))
    logging.info("[BEGIN] floods.py CLI")
    logging.info(f"Args: {vars(args)}")

    gdf = process_flood_data(
        dfo_shp=args.dfo_shp,
        dfo_attr=args.dfo_attr,
        hanze_csv=args.hanze_csv,
        hanze_regions_2010=args.hanze_regions_2010,
        hanze_regions_2021=args.hanze_regions_2021,
        hanze_regions_2010_layer=args.hanze_regions_2010_layer,
        hanze_regions_2021_layer=args.hanze_regions_2021_layer,
        hanze_code_field_2010=args.hanze_regions_2010_code,
        hanze_code_field_2021=args.hanze_regions_2021_code,
        usfd_csv=args.usfd_csv,
        simplify_tolerance_deg=(args.simplify_deg if args.simplify_deg and args.simplify_deg > 0 else None),
    )

    if gdf is None or gdf.empty:
        logging.warning("No flood features produced; nothing to write.")
        raise SystemExit(0)

    # Write CSV (attributes only)
    try:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        cols = [c for c in gdf.columns if c != "geometry"]
        gdf[cols].to_csv(args.out_csv, index=False)
        logging.info(f"Wrote CSV: {args.out_csv}  ({len(gdf)} rows)")
    except Exception as e:
        logging.error(f"Failed writing CSV: {e}")

    # Write GPKG (geometry)
    try:
        args.out_gpkg.parent.mkdir(parents=True, exist_ok=True)
        write_single_hazard_gdb(
            gdf=gdf,
            output_path=str(args.out_gpkg),
            layer_name=args.out_layer,
            fix_mode="wrap",
            precision=1e-7,
            verbose_geometry_logging=False,
        )
        logging.info(f"Wrote GPKG layer '{args.out_layer}' to {args.out_gpkg}  ({len(gdf)} features)")
    except Exception as e:
        logging.error(f"Failed writing GPKG: {e}")

    logging.info("[END] floods.py CLI")
