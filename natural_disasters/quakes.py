# -*- coding: utf-8 -*-
"""
Quick earthquake impact footprints (MMI VI/VII ellipses) from ISC-GEM-like CSV.

Exposes:
- process_earthquake_data(csv_path: str, ...) -> GeoDataFrame
  Columns: event_id, event_type, source, eq_date/start/end (+ *_iso),
           latitude, longitude, mw, depth_km, regime, mech, geom_method,
           band (VI/VII), area_km2, geometry (EPSG:4326), plus r/a/b/strike.

CLI:
python -m natural_disasters.quakes --csv <path> [--out-gpkg ...] [--out-csv ...]
"""

from __future__ import annotations

# Must be first: configure GDAL/PROJ before geospatial imports
try:
    from .bootstrap_gdal import verify_gdal_ready
    gd, pj = verify_gdal_ready()
except Exception:
    pass

import re, os, time, math, logging, argparse
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.affinity import scale as shp_scale, rotate as shp_rotate, translate as shp_translate
from shapely.ops import unary_union
from shapely.validation import make_valid
try:
    from shapely.ops import clip_by_rect  # Shapely 2.x
except Exception:
    def clip_by_rect(geom, minx, miny, maxx, maxy):
        return geom.intersection(box(minx, miny, maxx, maxy))
from pyproj import CRS, Transformer

WGS84 = "EPSG:4326"
WORLD_EQ_AREA = "EPSG:6933"  # equal-area, meters

# ---- Atkinson & Wald (2007) coeffs ----
COEFFS = {
    "active": {"C1":12.27,"C2":2.270,"C3":0.1304,"C4":-1.30,"C5":-7.07e-4,"C6":1.95,"C7":-0.577,"h":14.0,"Rt":30.0},
    "stable": {"C1":11.72,"C2":2.36,"C3":0.1155,"C4":-0.44,"C5":-0.002044,"C6":2.31,"C7":-0.479,"h":17.0,"Rt":80.0},
}

# ---------- utilities ----------
def _norm_name(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def _find_col(df: pd.DataFrame, *names: str) -> str | None:
    targets = {_norm_name(n) for n in names}
    for c in df.columns:
        if _norm_name(c) in targets:
            return c
    for c in df.columns:
        nc = _norm_name(c)
        if any(t in nc for t in targets):
            return c
    return None

def _as_int(x) -> int | None:
    try:
        v = pd.to_numeric(x, errors="coerce")
        if pd.isna(v): return None
        return int(v)
    except Exception:
        return None

def _assemble_datetime(df: pd.DataFrame) -> pd.Series:
    """Build a UTC series from:
       (date + time) | direct datetime-ish | single 'date' | parts | autodetect."""
    def to_utc(s: pd.Series) -> pd.Series:
        return pd.to_datetime(s.astype(str).str.strip(), errors="coerce", utc=True)

    cdate = _find_col(df, "date")
    ctime = _find_col(df, "time", "otime")
    if cdate is not None and ctime is not None:
        s = to_utc(df[cdate] + " " + df[ctime])
        if s.notna().any():
            logging.info(f"[time] using combined columns: {cdate!r} + {ctime!r}")
            return s

    for cname in ("datetime","isotime","time","origintime","otime"):
        c = _find_col(df, cname)
        if c is not None:
            s = to_utc(df[c])
            if s.notna().any():
                logging.info(f"[time] using direct datetime column: {c!r}")
                return s

    if cdate is not None:
        s = to_utc(df[cdate])
        if s.notna().any():
            logging.info(f"[time] using single date column: {cdate!r}")
            return s

    cy = _find_col(df, "year","yr","iyear","yyyy")
    cm = _find_col(df, "month","mon","mo","imonth")
    cd = _find_col(df, "day","dy","iday")
    ch = _find_col(df, "hour","hr","ihour","h")
    cmin = _find_col(df, "minute","min","imin","mn")
    csec = _find_col(df, "second","sec","isecond","s")

    if cy is not None:
        years  = df[cy]
        months = df[cm]   if cm   else None
        days   = df[cd]   if cd   else None
        hours  = df[ch]   if ch   else None
        mins   = df[cmin] if cmin else None
        secs   = df[csec] if csec else None

        out = []
        for i in range(len(df)):
            Y = _as_int(years.iat[i])  if years  is not None else None
            M = _as_int(months.iat[i]) if months is not None else 1
            D = _as_int(days.iat[i])   if days   is not None else 1
            hh = _as_int(hours.iat[i]) if hours  is not None else 0
            mm = _as_int(mins.iat[i])  if mins   is not None else 0
            ss = _as_int(secs.iat[i])  if secs   is not None else 0
            try:
                if Y is None:
                    out.append(pd.NaT); continue
                M = min(12, max(1, M or 1))
                D = min(28, max(1, D or 1))
                out.append(pd.Timestamp(Y, M, D, hh or 0, mm or 0, ss or 0, tz="UTC"))
            except Exception:
                out.append(pd.NaT)
        s = pd.Series(out, dtype="datetime64[ns, UTC]")
        if s.notna().any():
            logging.info("[time] assembled from parts (Y/M/D/H/M/S)")
            return s

    best_col, best_rate = None, 0.0
    for c in df.columns:
        s = to_utc(df[c]); rate = s.notna().mean()
        if rate > best_rate and rate >= 0.5:
            best_rate, best_col = rate, c
    if best_col is not None:
        logging.info(f"[time] autodetected datetime column: {best_col!r} (parse rate {best_rate:.2%})")
        return to_utc(df[best_col])

    logging.warning("[time] failed to parse any datetime; NaT will be used")
    return pd.Series([pd.NaT] * len(df), dtype="datetime64[ns, UTC]")

def _normalize_times(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    s = pd.to_datetime(series, errors="coerce", utc=True)
    naive = s.dt.tz_convert("UTC").dt.tz_localize(None)
    iso = s.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str.rstrip("0").str.rstrip(".")
    iso = iso.where(s.notna(), None)
    return naive, iso

# ---- core math & geometry helpers ----
def _crosses_dateline(lons: list[float]) -> bool:
    return any(abs(lons[i+1] - lons[i]) > 180 for i in range(len(lons)-1))

def _unwrap_lons(lons: list[float]) -> list[float]:
    out, shift, prev = [], 0.0, None
    for x in lons:
        if prev is not None:
            dx = x - prev
            if dx > 180:  shift -= 360
            elif dx < -180: shift += 360
        out.append(x + shift); prev = x
    return out

def _split_antimeridian(poly_wgs84: Polygon | MultiPolygon):
    if poly_wgs84.is_empty: return poly_wgs84
    def _split_one(p: Polygon):
        xs, ys = zip(*list(p.exterior.coords))
        if not _crosses_dateline(list(xs)): return p
        xs_u = _unwrap_lons(list(xs)); ring_u = list(zip(xs_u, ys))
        p_u = Polygon(ring_u)
        parts = []
        for offset in (-360, 0, 360):
            clipped = clip_by_rect(p_u, -180 + offset, -90, 180 + offset, 90)
            if not clipped.is_empty:
                if offset != 0: clipped = shp_translate(clipped, xoff=-offset, yoff=0.0)
                parts.append(clipped)
        if not parts: return p
        return make_valid(unary_union(parts))
    if isinstance(poly_wgs84, MultiPolygon):
        return make_valid(unary_union([_split_one(g) for g in poly_wgs84.geoms]))
    else:
        return _split_one(poly_wgs84)

def _mmi_pred(Mw: float, r_km: float, depth_km: float, cf: dict) -> float:
    h = max(cf["h"], 0.0 if pd.isna(depth_km) else float(depth_km))
    R = math.hypot(r_km, h)
    if R <= 0: R = 1e-6
    B = 0.0 if R <= cf["Rt"] else math.log10(R / cf["Rt"])
    return (cf["C1"]
        + cf["C2"]*(Mw-6.0)
        + cf["C3"]*(Mw-6.0)**2
        + cf["C4"]*math.log10(R)
        + cf["C5"]*R
        + cf["C6"]*B
        + cf["C7"]*Mw*math.log10(R))

def _radius_for_mmi(Mw: float, depth_km: float, target_I: float, cf: dict, rmax_km: float = 1500.0) -> float:
    if _mmi_pred(Mw, 0.0, depth_km, cf) < target_I: return 0.0
    lo, hi = 0.0, rmax_km
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if _mmi_pred(Mw, mid, depth_km, cf) >= target_I: lo = mid
        else: hi = mid
    return lo

def _rupture_area_km2_wc94(Mw: float) -> float:
    return 10 ** (-3.49 + 0.91*Mw)

def _mech_from_rake(rake: Optional[float]) -> str:
    if rake is None or pd.isna(rake): return "U"
    r = float(rake); ar = abs(((r + 180) % 360) - 180)
    if ar <= 30: return "SS"
    if 45 <= r <= 135: return "R"
    if -135 <= r <= -45: return "N"
    return "SS"

def _rupture_length_from_area(A_km2: float, mech: str) -> Tuple[float,float]:
    AR = 3.0 if mech == "SS" else 2.0
    if A_km2 <= 0: return 0.0, 0.0
    L = (A_km2 * AR) ** 0.5
    W = A_km2 / L
    return float(L), float(W)

def _pick_regime(lat: float, lon: float, user: str = "active") -> str:
    if user in ("active","stable"): return user
    return "active"

def _ellipse_polygon_wgs84(lat: float, lon: float, a_km: float, b_km: float,
                           strike_deg: float, segments: int = 180):
    a_m = max(0.0, a_km) * 1000.0
    b_m = max(0.0, b_km) * 1000.0
    if a_m == 0.0 and b_m == 0.0: return Point(lon, lat).buffer(0)
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
    inv = Transformer.from_crs(aeqd, 4326, always_xy=True).transform
    circ = Point(0, 0).buffer(1.0, resolution=segments)
    ell  = shp_scale(circ, xfact=a_m, yfact=b_m, origin=(0, 0))
    angle_ccw = 90.0 - (0.0 if pd.isna(strike_deg) else float(strike_deg))
    ell_rot = shp_rotate(ell, angle_ccw, origin=(0, 0), use_radians=False)
    xs, ys = zip(*list(ell_rot.exterior.coords))
    lonlat = [inv(x, y) for x, y in zip(xs, ys)]
    return _split_antimeridian(Polygon(lonlat))

def _area_km2(poly) -> Optional[float]:
    try:
        if poly is None or poly.is_empty: return None
        return gpd.GeoSeries([poly], crs=WGS84).to_crs(WORLD_EQ_AREA).area.iloc[0] / 1e6
    except Exception:
        return None

# ---------- EM-DAT filtering ----------
def _load_emdat_country_set(emdat_csv: str) -> set[str]:
    """Return a normalized set of country names present in the EM-DAT CSV."""
    em = pd.read_csv(emdat_csv, low_memory=False)
    # Try common columns (EM-DAT exports vary)
    for cname in ("country", "Country", "COUNTRY", "Country name", "country_name"):
        if cname in em.columns:
            s = em[cname].astype(str)
            return {_norm_name(x) for x in s.dropna().unique()}
    # fallback: try ISO if that is what’s present
    for cname in ("iso", "ISO", "iso3", "ISO3"):
        if cname in em.columns:
            s = em[cname].astype(str)
            return {_norm_name(x) for x in s.dropna().unique()}
    # empty set -> no filtering
    return set()

def _country_name_from_row(row) -> str | None:
    """Pick a reasonable country name field from a countries layer row."""
    for f in ("ADMIN","NAME_EN","NAME","SOVEREIGNT","COUNTRY","CNTRY_NAME","GID_0","NAME_0"):
        if f in row and pd.notna(row[f]): return str(row[f])
    # ISO fallback
    for f in ("ISO_A3","ISO3","ADM0_A3"):
        if f in row and pd.notna(row[f]): return str(row[f])
    return None

# ---- public entry point used by main.py ----
def process_earthquake_data(csv_path: str,
                            output_folder: Optional[str] = None,
                            regime: str = "active",
                            segments: int = 180,
                            simplify_deg: float | None = None,
                            # EM-DAT filtering (optional; mirrors main.py wiring)
                            emdat_csv: Optional[str] = None,
                            countries: Optional[str] = None,
                            countries_layer: Optional[str] = None) -> gpd.GeoDataFrame:

    t0 = time.time()
    df = pd.read_csv(Path(csv_path), low_memory=False)
    logging.info(f"[read] rows={len(df):,} | elapsed={time.time()-t0:.2f}s")

    # ---- resolve required columns ----
    lat_col = _find_col(df, "lat", "latitude")
    lon_col = _find_col(df, "lon", "longitude")
    mw_col     = _find_col(df, "mw", "mwc", "mag", "magnitude")
    depth_col  = _find_col(df, "depth", "depth_km", "dep_km")
    eid_col    = _find_col(df, "eventid", "event_id", "evid", "id")
    if lat_col is None or lon_col is None:
        missing = []
        if lat_col is None: missing.append("lat/latitude")
        if lon_col is None: missing.append("lon/longitude")
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    need_missing = [nm for nm, col in {"mw": mw_col, "depth": depth_col}.items() if col is None]
    if need_missing:
        raise ValueError(f"Missing required columns in {csv_path}: {need_missing}")

    # ---- robust datetime + normalization ----
    eq_date_raw = _assemble_datetime(df)
    eq_date_dt, eq_date_iso = _normalize_times(eq_date_raw)

    # ---- numeric series ----
    ev_id = (df[eid_col].astype(str) if eid_col else df.index.astype(str))
    lat   = pd.to_numeric(df[lat_col],   errors="coerce")
    lon   = pd.to_numeric(df[lon_col],   errors="coerce")
    mw    = pd.to_numeric(df[mw_col],    errors="coerce")
    depth = pd.to_numeric(df[depth_col], errors="coerce")

    # ---- optional EM-DAT country filtering BEFORE building ellipses ----
    if emdat_csv and countries:
        try:
            em_cset = _load_emdat_country_set(emdat_csv)
            if em_cset:
                pts = gpd.GeoDataFrame({"event_id": ev_id, "mw": mw, "depth_km": depth},
                                       geometry=gpd.points_from_xy(lon, lat, crs=WGS84))
                world = gpd.read_file(countries, layer=countries_layer) if countries_layer else gpd.read_file(countries)
                if world.crs is None or str(world.crs).upper() != WGS84:
                    world = world.to_crs(WGS84)
                # name column to compare
                world["_nm_"] = world.apply(_country_name_from_row, axis=1)
                world["_nm_norm_"] = world["_nm_"].fillna("").map(_norm_name)
                keep_ids = set()
                # spatial join epicenters -> country name -> filter by EM-DAT set
                j = gpd.sjoin(pts, world[["_nm_norm_", "geometry"]], how="left", predicate="within")
                mask = j["_nm_norm_"].isin(em_cset)
                keep_ids = set(j.loc[mask, "event_id"].astype(str).tolist())
                keep_mask = ev_id.astype(str).isin(keep_ids)
                kept = int(keep_mask.sum())
                logging.info(f"[emdat] filtered by countries in EM-DAT: kept {kept:,}/{len(df):,} events")
                df = df.loc[keep_mask].reset_index(drop=True)
                # re-slice series to the filtered df
                ev_id = ev_id.loc[keep_mask].reset_index(drop=True)
                lat   = lat.loc[keep_mask].reset_index(drop=True)
                lon   = lon.loc[keep_mask].reset_index(drop=True)
                mw    = mw.loc[keep_mask].reset_index(drop=True)
                depth = depth.loc[keep_mask].reset_index(drop=True)
                eq_date_dt  = eq_date_dt.loc[keep_mask].reset_index(drop=True)
                eq_date_iso = eq_date_iso.loc[keep_mask].reset_index(drop=True)
            else:
                logging.info("[emdat] CSV provided but no recognizable country column; skipping EM-DAT filter")
        except Exception as e:
            logging.error(f"[emdat] filtering failed: {e}")

    # ---- optional extras (radius & mechanism/strike hints) ----
    def _nan_series(n: int) -> pd.Series:
        return pd.Series([float("nan")] * n)

    earthqk_radius_col = _find_col(df, "earthqk_radius", "radius_mil", "radius_mi", "eq_radius")
    radius_mi = (pd.to_numeric(df[earthqk_radius_col], errors="coerce")
                 if earthqk_radius_col else pd.Series([pd.NA]*len(df)))

    smaj_col   = _find_col(df, "smajax", "semi_major_km", "semi_major", "smaj", "smjr")
    smin_col   = _find_col(df, "sminax", "semi_minor_km", "semi_minor", "smin", "smnr")
    strike_col = _find_col(df, "strike", "strike_deg", "azimuth", "azi")
    str1_col   = _find_col(df, "str1", "np1_strike", "strike1", "strike_np1")
    rake1_col  = _find_col(df, "rake1", "np1_rake", "rake_np1")

    N = len(df)
    smajax     = (pd.to_numeric(df[smaj_col], errors="coerce")   if smaj_col   else _nan_series(N))
    sminax     = (pd.to_numeric(df[smin_col], errors="coerce")   if smin_col   else _nan_series(N))
    strike_unc = (pd.to_numeric(df[strike_col], errors="coerce") if strike_col else _nan_series(N))
    str1       = (pd.to_numeric(df[str1_col], errors="coerce")   if str1_col   else _nan_series(N))
    rake1      = (pd.to_numeric(df[rake1_col], errors="coerce")  if rake1_col  else _nan_series(N))

    # ---- build features ----
    t_build = time.time()
    recs = []
    for i in range(N):
        la, lo, M, z = lat.iat[i], lon.iat[i], mw.iat[i], depth.iat[i]
        if pd.isna(la) or pd.isna(lo) or pd.isna(M):
            continue

        reg = _pick_regime(la, lo, regime)
        cf = COEFFS[reg]

        r6 = _radius_for_mmi(M, z, 6.0, cf)
        r7 = _radius_for_mmi(M, z, 7.0, cf)

        mech = _mech_from_rake(rake1.iat[i] if i < len(rake1) else None)
        A = _rupture_area_km2_wc94(M)
        L, W = _rupture_length_from_area(A, mech)

        a6, b6 = r6 + 0.5*L, r6
        a7, b7 = r7 + 0.5*L, r7
        if mech == "SS":
            a6 *= 1.15; a7 *= 1.15

        # Inflate by epicentral uncertainty (km) if available
        smj = smajax.iat[i] if i < len(smajax) else float("nan")
        smn = sminax.iat[i] if i < len(sminax) else float("nan")
        if not pd.isna(smj):
            a6 = (a6*a6 + smj*smj) ** 0.5
            a7 = (a7*a7 + smj*smj) ** 0.5
        if not pd.isna(smn):
            b6 = (b6*b6 + smn*smn) ** 0.5
            b7 = (b7*b7 + smn*smn) ** 0.5

        # orientation (prefer nodal plane 1 strike; else generic strike; else 0)
        if i < len(str1) and not pd.isna(str1.iat[i]):
            strike_deg = float(str1.iat[i])
        elif i < len(strike_unc) and not pd.isna(strike_unc.iat[i]):
            strike_deg = float(strike_unc.iat[i])
        else:
            strike_deg = 0.0

        # geometries
        g6 = _ellipse_polygon_wgs84(la, lo, a6, b6, strike_deg, segments=segments)
        g7 = _ellipse_polygon_wgs84(la, lo, a7, b7, strike_deg, segments=segments)

        base = {
            "event_id": ev_id.iat[i],
            "event_type": "earthquake",
            "source": "ISC-GEM",
            "eq_date":   (eq_date_dt.iat[i]  if i < len(eq_date_dt)  else pd.NaT),
            "start_time":(eq_date_dt.iat[i]  if i < len(eq_date_dt)  else pd.NaT),
            "end_time":  (eq_date_dt.iat[i]  if i < len(eq_date_dt)  else pd.NaT),
            "eq_date_iso":   (eq_date_iso.iat[i] if i < len(eq_date_iso) else None),
            "start_time_iso":(eq_date_iso.iat[i] if i < len(eq_date_iso) else None),
            "end_time_iso":  (eq_date_iso.iat[i] if i < len(eq_date_iso) else None),
            "latitude": la, "longitude": lo, "mw": M, "depth_km": z,
            "regime": reg, "mech": mech, "geom_method": "ellipse_aw07+wc94",
            "r6_km": r6, "r7_km": r7, "rupture_L_km": L, "rupture_A_km2": A,
            "a6_km": a6, "b6_km": b6, "a7_km": a7, "b7_km": b7,
            "strike_deg": strike_deg,
            "radius_mil": (float(radius_mi.iat[i]) if i < len(radius_mi) and pd.notna(radius_mi.iat[i]) else None),
        }
        recs.append({**base, "band": "VI",  "geometry": g6, "area_km2": _area_km2(g6)})
        recs.append({**base, "band": "VII", "geometry": g7, "area_km2": _area_km2(g7)})

    gdf = gpd.GeoDataFrame(recs, crs=WGS84)
    logging.info(f"[build] done: features={len(gdf):,} | elapsed={time.time()-t0:.2f}s")

    # optional simplify before write
    if simplify_deg and simplify_deg > 0:
        logging.info(f"[export] simplifying geometries: tolerance={simplify_deg}° (preserve_topology=True)")
        gdf["geometry"] = gdf.geometry.simplify(simplify_deg, preserve_topology=True)

    return gdf

# ---- CLI ----
if __name__ == "__main__":
    from .helpers import setup_logging, output_path, write_single_hazard_gdb
    ap = argparse.ArgumentParser(description="Build quick EQ impact ellipses (MMI VI/VII) from CSV")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-gpkg", type=Path, default=output_path("earthquakes.gpkg"))
    ap.add_argument("--out-layer", type=str, default="earthquakes")
    ap.add_argument("--out-csv", type=Path, default=output_path("earthquakes.csv"))
    ap.add_argument("--regime", choices=["active","stable","auto"], default="active")
    ap.add_argument("--segments", type=int, default=180)
    ap.add_argument("--simplify-deg", type=float, default=0)
    # EM-DAT filtering (optional)
    ap.add_argument("--emdat-csv", type=str, default=None,
                    help="EM-DAT CSV to restrict to its countries (Top 5% file etc.)")
    ap.add_argument("--countries", type=str, default=None,
                    help="Countries dataset for locating epicenters (same used elsewhere)")
    ap.add_argument("--countries-layer", type=str, default=None)
    ap.add_argument("--log-level", choices=["DEBUG","INFO","WARNING","ERROR"], default="INFO")
    args = ap.parse_args()

    setup_logging(output_path(""))
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logging.info("[BEGIN] quakes.py CLI")
    logging.info(f"Args: {{'csv': '{args.csv}', 'out_gpkg': '{args.out_gpkg}', 'out_layer': '{args.out_layer}', "
                 f"'out_csv': '{args.out_csv}', 'regime': '{args.regime}', 'segments': {args.segments}, "
                 f"'simplify_deg': {args.simplify_deg}, 'emdat_csv': '{args.emdat_csv}', "
                 f"'countries': '{args.countries}', 'countries_layer': '{args.countries_layer}', "
                 f"'log_level': '{args.log_level}'}}")

    os.environ.setdefault("OGR_SQLITE_SYNCHRONOUS", "OFF")
    os.environ.setdefault("OGR_SQLITE_CACHE", "512")

    g = process_earthquake_data(
        args.csv,
        output_folder=str(output_path("")),
        regime=args.regime,
        segments=args.segments,
        simplify_deg=(args.simplify_deg if args.simplify_deg > 0 else None),
        emdat_csv=args.emdat_csv,
        countries=args.countries,
        countries_layer=args.countries_layer,
    )

    # Write CSV (attrs only)
    try:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["geometry"]).to_csv(args.out_csv, index=False)
        logging.info(f"Wrote CSV: {args.out_csv} ({len(g):,} rows)")
    except Exception as e:
        logging.error(f"Failed writing CSV: {e}")

    # Write GPKG
    try:
        args.out_gpkg.parent.mkdir(parents=True, exist_ok=True)
        write_single_hazard_gdb(
            gdf=g, output_path=str(args.out_gpkg), layer_name=args.out_layer,
            fix_mode="wrap", precision=1e-7, verbose_geometry_logging=False,
        )
        logging.info(f"Wrote GPKG layer '{args.out_layer}' to {args.out_gpkg} ({len(g):,} features)")
    except Exception as e:
        logging.error(f"Failed writing GPKG: {e}")

    logging.info("[END] quakes.py CLI")
