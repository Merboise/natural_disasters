#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
assemble_flood_data.py

Build a consolidated flood-event layer (OECD-ready) by combining:
- DFO (polygons + attributes)
- HANZE (EU subnational polygons + impacts)
- USFD (US events + area; geometry derived from area if needed)
- EM-DAT (global table; polygons approximated from admin names or buffers)

Overrides:
- If a HANZE or USFD event overlaps a DFO/EM-DAT event in space & time, prefer HANZE/USFD.
- DFO polygons are used when available; EM-DAT fills gaps elsewhere.

Optional refinement:
- If EM-DAT has only a point, estimate a buffer radius using GHSL population raster:
    radius_km ~ sqrt( (TotalAffected / local_density_people_per_km2) / pi )

Outputs:
- GeoPackage with one layer: `flood_events`
- Common attributes across sources for downstream analysis

Usage (example):
python assemble_flood_data.py \
  --dfo-shp /data/DFO/FloodArchive_region.shp \
  --dfo-attr /data/DFO/FloodArchive.xlsx \
  --hanze-shp /data/HANZE/HANZE_floods_regions_2021.shp \
  --hanze-csv /data/HANZE/HANZE_events.csv \
  --usfd-csv /data/USFD/USFD_v1.0.csv \
  --emdat /data/EMDAT/emdat_public_floods.xlsx \
  --adm-shp /data/GADM/gadm41_levels.gpkg \
  --adm-layer level1 \
  --pop-tif /data/GHSL/GHS_POP_2020.tif \
  --o /data/out/consolidated_floods.gpkg
"""

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple, List

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import xy as transform_xy
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union

# ---- CRS helpers ----
WGS84 = "EPSG:4326"
WORLD_EQ_AREA = "EPSG:6933"  # World Cylindrical Equal Area (meters)

# ---- Geometry helpers ----
def reproject_buffer_km(geom, radius_km: float):
    """Buffer in km using an equal-area CRS, return in WGS84."""
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

def pixel_area_km2_from_raster(src: rasterio.io.DatasetReader) -> float:
    # Assumes projected CRS in meters or geographic degrees; we handle both.
    # If geographic (degrees), we approximate area near the sample point later.
    res_x = abs(src.transform.a)
    res_y = abs(src.transform.e)
    if src.crs and src.crs.is_projected:
        return (res_x * res_y) / 1e6  # m^2 -> km^2
    # Fallback: rough global mean for 30 arcsec / 1km-ish grids not reliable
    return None

def sample_population_density(src: rasterio.io.DatasetReader,
                              lon: float, lat: float) -> Optional[float]:
    """
    Return people per km^2 at (lon,lat) from a GHSL population *count* raster by
    dividing the pixel count by pixel area in km^2. If CRS is geographic, we
    compute local pixel area by projecting a 1x1 pixel polygon around the point
    to WORLD_EQ_AREA.
    """
    try:
        vals = list(src.sample([(lon, lat)]))
        if not vals:
            return None
        pop_count = float(vals[0][0])
        if np.isnan(pop_count) or pop_count <= 0:
            return None

        if src.crs and src.crs.is_projected:
            px_area_km2 = pixel_area_km2_from_raster(src)
            if not px_area_km2 or px_area_km2 <= 0:
                return None
            return pop_count / px_area_km2

        # Geographic CRS: estimate local pixel polygon and reproject to equal-area
        row, col = src.index(lon, lat)
        # pixel corners in raster space -> bounds -> four corners in lon/lat
        left, top = src.transform * (col, row)
        right, bottom = src.transform * (col + 1, row + 1)
        px_poly = Polygon([(left, top), (right, top), (right, bottom), (left, bottom)])
        px_area_km2 = polygon_area_km2(px_poly)
        if px_area_km2 <= 0:
            return None
        return pop_count / px_area_km2
    except Exception:
        return None

# ---- Data readers ----
def read_tabular(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)

def safe_datetime(y, m, d) -> Optional[pd.Timestamp]:
    try:
        yi = int(y)
        mi = int(m) if pd.notna(m) else 1
        di = int(d) if pd.notna(d) else 1
        return pd.Timestamp(year=yi, month=max(1,mi), day=max(1,di))
    except Exception:
        return None

# ---- Admin matching ----
def prep_admin(adm_path: Path, adm_layer: Optional[str], name_fields: List[str]) -> gpd.GeoDataFrame:
    if adm_path.suffix.lower() in (".gpkg", ".gdb"):
        gdf = gpd.read_file(adm_path, layer=adm_layer) if adm_layer else gpd.read_file(adm_path)
    else:
        gdf = gpd.read_file(adm_path)
    for f in name_fields:
        if f in gdf.columns:
            gdf[f"{f}__lower"] = gdf[f].astype(str).str.lower()
    gdf = gdf.to_crs(WGS84)
    return gdf

def match_admin_polys(admin_gdf: gpd.GeoDataFrame, location_str: str,
                      name_fields: List[str]) -> Optional[MultiPolygon]:
    if not location_str or not isinstance(location_str, str):
        return None
    tokens = [t.strip() for t in location_str.replace("&", ",").replace("/", ",").split(",") if t.strip()]
    matches = []
    for tok in tokens:
        tok_low = tok.lower()
        # try contains on provided fields
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

# ---- Event model & consolidation ----
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
    """Return True if two events overlap within ±pad_days."""
    if a_start is None and b_start is None:
        return True
    # replace None with other for simple overlap logic
    a0 = a_start or b_start
    a1 = a_end or a0
    b0 = b_start or a_start or a1
    b1 = b_end or b0
    a0p = a0 - pd.Timedelta(days=pad_days)
    a1p = a1 + pd.Timedelta(days=pad_days)
    return not (a1p < b0 or b1 < a0p)

def override_priority(src: str) -> int:
    """
    Higher value => higher priority override.
    HANZE/USFD > DFO > EMDAT
    """
    rank = {"HANZE": 3, "USFD": 3, "DFO": 2, "EMDAT": 1}
    return rank.get(src.upper(), 0)

def consolidate_events(events: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Merge/override events: if two events overlap in space/time, keep the higher-priority.
    """
    events = events.copy()
    events = events[~events.geometry.is_empty & events.geometry.notnull()]
    events = events.reset_index(drop=True)
    keep = [True] * len(events)

    # Simple n^2 pass; fine for tens of thousands. For huge volumes, spatial index & sweep line.
    for i in range(len(events)):
        if not keep[i]:
            continue
        gi = events.geometry.iloc[i]
        si = events.source.iloc[i]
        ti0, ti1 = events.start_date.iloc[i], events.end_date.iloc[i]
        for j in range(i + 1, len(events)):
            if not keep[j]:
                continue
            # Quick bbox check
            if not gi.bounds or not events.geometry.iloc[j].bounds:
                continue
            if not gi.intersects(events.geometry.iloc[j]):
                continue
            # Time overlap?
            if not time_overlap(ti0, ti1, events.start_date.iloc[j], events.end_date.iloc[j]):
                continue
            # Decide by priority
            sj = events.source.iloc[j]
            if override_priority(sj) > override_priority(si):
                keep[i] = False
                break
            elif override_priority(sj) < override_priority(si):
                keep[j] = False
            else:
                # Same priority: keep the one with larger/more detailed area
                ai = events.area_km2.iloc[i] or 0
                aj = events.area_km2.iloc[j] or 0
                if aj > ai:
                    keep[i] = False
                    break
                else:
                    keep[j] = False

    return events.loc[keep].reset_index(drop=True)

# ---- Source-specific loaders ----
def load_dfo(dfo_shp: Path, dfo_attr: Path) -> List[dict]:
    if not dfo_shp or not dfo_shp.exists():
        return []
    gdf = gpd.read_file(dfo_shp).to_crs(WGS84)

    attrs = read_tabular(dfo_attr) if dfo_attr and dfo_attr.exists() else pd.DataFrame()
    # try to find ID column
    id_cols = [c for c in attrs.columns if c.strip().lower() == "id"]
    if id_cols:
        attrs["ID"] = pd.to_numeric(attrs[id_cols[0]], errors="coerce")
        gdf = gdf.merge(attrs, on="ID", how="left")

    out = []
    for _, r in gdf.iterrows():
        ev_id = r.get("ID", None)
        began = r.get("Began", None)
        ended = r.get("Ended", None)
        out.append(normalize_event_row(
            geom=r.geometry,
            event_id=f"DFO_{ev_id}" if pd.notna(ev_id) else None,
            source="DFO",
            start_date=began,
            end_date=ended,
            country=r.get("Country", None),
            area_km2=r.get("Area", None),
            fatalities=r.get("Dead", None),
            affected=r.get("Displaced", None),
            cause=r.get("MainCause", None),
        ))
    return out

def load_hanze(hanze_shp: Path, hanze_csv: Path,
               id_field_shp: str = "event_id",
               id_field_csv: str = "event_id") -> List[dict]:
    if not hanze_shp or not hanze_shp.exists():
        return []
    gdf = gpd.read_file(hanze_shp).to_crs(WGS84)
    attrs = read_tabular(hanze_csv) if hanze_csv and hanze_csv.exists() else pd.DataFrame()
    if id_field_shp in gdf.columns and id_field_csv in attrs.columns:
        gdf = gdf.merge(attrs, left_on=id_field_shp, right_on=id_field_csv, how="left")

    out = []
    for _, r in gdf.iterrows():
        # Column name guesses per HANZE v2.1
        ev_id = r.get(id_field_shp, None)
        sd = r.get("start_date", r.get("StartDate", r.get("startDate", None)))
        ed = r.get("end_date", r.get("EndDate", r.get("endDate", None)))
        area = r.get("area_km2", None)
        deaths = r.get("deaths", r.get("fatalities", None))
        ppl = r.get("people_affected", r.get("affected", None))
        cause = r.get("flood_type", r.get("type", None))
        country = r.get("country", r.get("Country", None))

        out.append(normalize_event_row(
            geom=r.geometry,
            event_id=f"HANZE_{ev_id}" if ev_id is not None else None,
            source="HANZE",
            start_date=sd,
            end_date=ed,
            country=country,
            area_km2=area,
            fatalities=deaths,
            affected=ppl,
            cause=cause
        ))
    return out

def load_usfd(usfd_csv: Path) -> List[dict]:
    if not usfd_csv or not usfd_csv.exists():
        return []
    df = read_tabular(usfd_csv)
    # expected fields per Zenodo: DATE_BEGIN, DATE_END, LON, LAT, AREA (km^2)
    out = []
    for idx, r in df.iterrows():
        lon = r.get("LON", None)
        lat = r.get("LAT", None)
        area = r.get("AREA", None)
        area = float(area) if pd.notna(area) else None

        geom = None
        if pd.notna(lon) and pd.notna(lat):
            # If AREA given, make a circle buffer with equivalent area.
            if area and area > 0:
                radius_km = circle_radius_from_area_km2(area)
                geom = reproject_buffer_km(Point(float(lon), float(lat)), radius_km)
            else:
                # minimal placeholder geometry
                geom = reproject_buffer_km(Point(float(lon), float(lat)), 5.0)

        out.append(normalize_event_row(
            geom=geom,
            event_id=f"USFD_{idx}",
            source="USFD",
            start_date=r.get("DATE_BEGIN", None),
            end_date=r.get("DATE_END", None),
            country=r.get("COUNTRY", "United States"),
            area_km2=area,
            fatalities=r.get("FATALITY", None),
            affected=None,
            cause=r.get("CAUSE", None)
        ))
    return out

def load_emdat(emdat_path: Path, admin_gdf: gpd.GeoDataFrame,
               admin_name_fields: List[str],
               pop_src: Optional[rasterio.io.DatasetReader],
               default_riverine_km=25.0, default_flash_km=10.0) -> List[dict]:
    if not emdat_path or not emdat_path.exists():
        return []
    df = read_tabular(emdat_path)

    # EM-DAT public table typical columns:
    # 'Dis No', 'Country', 'Location', 'Latitude', 'Longitude',
    # 'Start Year','Start Month','Start Day','End Year','End Month','End Day',
    # 'Total Affected','Total Death','Disaster Subtype', 'Disaster Type'
    # Filter to flood events if not pre-filtered:
    if "Disaster Type" in df.columns:
        df = df[df["Disaster Type"].astype(str).str.upper().str.contains("FLOOD")]

    out = []
    for _, r in df.iterrows():
        ev_id = r.get("Dis No", None)

        sd = safe_datetime(r.get("Start Year"), r.get("Start Month"), r.get("Start Day"))
        ed = safe_datetime(r.get("End Year"), r.get("End Month"), r.get("End Day"))

        country = r.get("Country", None)
        location = r.get("Location", None)
        lat = r.get("Latitude", None)
        lon = r.get("Longitude", None)
        affected = r.get("Total Affected", None)
        deaths = r.get("Total Death", None)
        subtype = r.get("Disaster Subtype", r.get("Disaster Subtype ", None))

        geom = None

        # 1) Try location -> admin polygons union
        if pd.notna(location):
            unioned = match_admin_polys(admin_gdf, str(location), admin_name_fields)
            if unioned and not unioned.is_empty:
                geom = unioned

        # 2) If not matched and we have a point, create a buffer
        if (geom is None or geom.is_empty) and pd.notna(lat) and pd.notna(lon):
            # Estimate radius using GHSL if available and affected given
            radius_km = default_riverine_km
            if pop_src is not None and pd.notna(affected):
                dens = sample_population_density(pop_src, float(lon), float(lat))
                if dens and dens > 0:
                    est_area_km2 = float(affected) / float(dens)
                    if est_area_km2 > 1.0:
                        radius_km = max(default_flash_km, circle_radius_from_area_km2(est_area_km2))
            geom = reproject_buffer_km(Point(float(lon), float(lat)), radius_km)

        # 3) If still nothing, try the whole country polygon
        if (geom is None or geom.is_empty) and country:
            # match by common country name fields
            for f in ["ADMIN", "NAME_0", "GID_0", "CNTR_NAME", "COUNTRY", "NAME_ENGLI", "SOVEREIGNT"]:
                if f in admin_gdf.columns:
                    m = admin_gdf[f].astype(str).str.lower() == str(country).lower()
                    if m.any():
                        geom = unary_union(admin_gdf.loc[m, "geometry"])
                        break

        if geom is None or geom.is_empty:
            continue

        out.append(normalize_event_row(
            geom=geom,
            event_id=f"EMDAT_{ev_id}" if ev_id is not None else None,
            source="EMDAT",
            start_date=sd,
            end_date=ed,
            country=country,
            area_km2=None,
            fatalities=deaths,
            affected=affected,
            cause=subtype
        ))
    return out

# ---- CLI ----
def main():
    ap = argparse.ArgumentParser(description="Assemble flood datasets into a consolidated GeoPackage.")
    ap.add_argument("--dfo-shp", type=Path, required=False)
    ap.add_argument("--dfo-attr", type=Path, required=False)
    ap.add_argument("--hanze-shp", type=Path, required=False)
    ap.add_argument("--hanze-csv", type=Path, required=False)
    ap.add_argument("--usfd-csv", type=Path, required=False)
    ap.add_argument("--emdat", type=Path, required=False, help="EM-DAT flood export (xlsx/csv)")
    ap.add_argument("--adm-shp", type=Path, required=True, help="Admin boundaries (GADM gpkg/gdb/shp)")
    ap.add_argument("--adm-layer", type=str, required=False, help="Layer name if using GPKG/GDB")
    ap.add_argument("--adm-name-fields", type=str, nargs="+",
                    default=["NAME_1", "NAME_2", "NAME_ENGLI"], help="Admin name fields to use for matching")
    ap.add_argument("--pop-tif", type=Path, required=False, help="GHSL population raster (optional)")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output GeoPackage path")
    args = ap.parse_args()

    # Load admin boundaries
    admin_gdf = prep_admin(args.adm_shp, args.adm_layer, args.adm_name_fields)

    # Optional population raster
    pop_src = rasterio.open(args.pop_tif) if args.pop_tif and args.pop_tif.exists() else None

    # Load sources
    records: List[dict] = []
    records += load_dfo(args.dfo_shp, args.dfo_attr) if args.dfo_shp else []
    records += load_hanze(args.hanze_shp, args.hanze_csv) if args.hanze_shp else []
    records += load_usfd(args.usfd_csv) if args.usfd_csv else []
    records += load_emdat(args.emdat, admin_gdf, args.adm_name_fields, pop_src) if args.emdat else []

    if pop_src is not None:
        pop_src.close()

    if not records:
        raise SystemExit("No records ingested. Check your input paths/filters.")

    gdf = gpd.GeoDataFrame(records, crs=WGS84)
    # Basic cleanups
    gdf["start_date"] = pd.to_datetime(gdf["start_date"], errors="coerce")
    gdf["end_date"] = pd.to_datetime(gdf["end_date"], errors="coerce")

    # Consolidate with override logic
    consolidated = consolidate_events(gdf)

    # Write
    args.output.parent.mkdir(parents=True, exist_ok=True)
    consolidated.to_file(args.output, layer="flood_events", driver="GPKG")
    print(f"Wrote {len(consolidated)} events to {args.output} (layer=flood_events)")

if __name__ == "__main__":
    main()
