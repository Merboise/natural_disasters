# ----
# main.py
# ----
import os, sys, logging
from os.path import join

# Must be first: configure env before any geospatial import happens
from .bootstrap_gdal import verify_gdal_ready
gd, pj = verify_gdal_ready()   # you can log these if you want

import argparse
import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv

from .helpers import (
    setup_logging, write_single_hazard_gdb, combine_disasters_to_gdb, data_path, output_path,
    unify_schema_storms, unify_schema_quakes, unify_schema_tsunamis, spatial_temporal_filter,
    audit_emdat_matches, write_exclusions, unify_schema_floods,
    write_gpkg,   # <-- add this
)

import calendar

from .storms import process_ibtracs_data
from .quakes import process_earthquake_data
from ..archive.tsunamis import process_tsunami_data
from .floods import process_flood_data, enrich_floods

load_dotenv()
DEM_LOCAL_ROOT = os.getenv("DEM_LOCAL_ROOT", None)
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(__file__))

# --- OECD membership (as of 2025-08) ---
_OECD_ISO3 = {
    "AUS","AUT","BEL","CAN","CHL","COL","CZE","DNK","EST","FIN","FRA","DEU",
    "GRC","HUN","ISL","IRL","ISR","ITA","JPN","KOR","LVA","LTU","LUX","MEX",
    "NLD","NZL","NOR","POL","PRT","SVK","SVN","ESP","SWE","CHE","TUR","GBR","USA"
}
def is_oecd(iso3: object) -> bool:
    return isinstance(iso3, str) and iso3.strip().upper() in _OECD_ISO3

def _norm_event_id(x: object) -> str:
    s = "" if x is None else str(x)
    # normalize Unicode dashes to ASCII hyphen-minus
    return (s.replace("\u2010","-").replace("\u2011","-").replace("\u2012","-")
             .replace("\u2013","-").replace("\u2014","-").replace("\u2212","-")).strip()

def _haz_norm(s: object) -> str:
    t = (str(s) if s is not None else "").strip().lower()
    if any(k in t for k in ["storm","cyclone","typhoon","hurricane","severe storm"]): return "storm"
    if "earth" in t: return "earthquake"
    if "tsunami" in t: return "tsunami"
    if "flood" in t: return "flood"
    return "other"

# ---------- EM-DAT loader ----------
def process_emdat_data(emdat_csv_path: str) -> pd.DataFrame:
    logging.info("Processing EM-DAT data...")
    try:
        df = pd.read_csv(emdat_csv_path, low_memory=False)

        # standardize IDs and ISO3
        cand_iso = [c for c in df.columns if c.lower() in ("iso3","iso")]
        df["iso3"] = df[cand_iso[0]].astype(str).str.upper() if cand_iso else ""
        cand_id = [c for c in df.columns if c.lower() in ("disno","disasterno","eventid","event_id")]
        if cand_id: df["event_id"] = df[cand_id[0]].map(_norm_event_id)
        else:       df["event_id"] = df.index.astype(str)

        def mk_date(row, prefix):
            y = _coerce_int(row.get(f"{prefix}year"))
            if y is None:
                return pd.NaT

            m = _coerce_int(row.get(f"{prefix}month"))
            d = _coerce_int(row.get(f"{prefix}day"))

            # defaults if missing
            m = 1 if m is None else m
            d = 1 if d is None else d

            # clamp month and day
            m = min(12, max(1, m))
            last_day = calendar.monthrange(y, m)[1]
            d = min(last_day, max(1, d))

            try:
                return pd.Timestamp(year=y, month=m, day=d)
            except Exception:
                # extremely defensive fallback
                return pd.to_datetime(f"{y}-{m}-{d}", errors="coerce")


        df["start_date"] = df.apply(lambda r: mk_date(r, "start"), axis=1)
        df["end_date"]   = df.apply(lambda r: mk_date(r, "end"), axis=1)
        df = df.dropna(subset=["start_date"])

        # hazard type (normalize)
        haz_cols = [c for c in df.columns if c.lower() in ("disaster_type","disastertype","type","haz_type","classification")]
        df["haz_type_raw"] = df[haz_cols[0]] if haz_cols else ""
        df["haz_norm"] = df["haz_type_raw"].map(_haz_norm)

        # coordinates if present
        lat_cand = [c for c in df.columns if c.lower() in ("lat","latitude","em_lat")]
        lon_cand = [c for c in df.columns if c.lower() in ("lon","longitude","long","em_lon")]
        if lat_cand: df["em_lat"] = pd.to_numeric(df[lat_cand[0]], errors="coerce")
        if lon_cand: df["em_lon"] = pd.to_numeric(df[lon_cand[0]], errors="coerce")

        # OECD flag
        df["oecd_ok"] = df["iso3"].map(is_oecd)

        logging.info(f"Loaded {len(df)} EM-DAT records with valid start dates.")
        return df
    except Exception as e:
        logging.error(f"EM-DAT processing failed: {e}")
        return pd.DataFrame()

# small helper
def _nonempty(g):
    return hasattr(g, "empty") and not g.empty

def _coerce_int(x):
    """Return int(x) if possible (handling strings/floats like '3', '3.0', '1e1'); else None."""
    try:
        if pd.isna(x):
            return None
        v = pd.to_numeric(x, errors="coerce")
        if pd.isna(v):
            return None
        return int(v)
    except Exception:
        return None
    
def main(
    use_emdat_filtering: bool = True,
    do_spatial_iso: bool = False,
    do_temporal_bucketing: bool = False,
    run_storms: bool = False,
    run_earthquakes: bool = True,
    run_tsunamis: bool = False,
    run_floods: bool = False,
    oecd_filter: str = "oecd",   # "oecd" | "non_oecd" | "all"
    # inputs
    ibtracs_file: str = data_path("ibtracs.ALL.list.v04r01.csv"),
    earthquake_file: str = data_path("isc-gem/isc-gem-cat.csv"),
    tsunami_events_file: str = data_path("tsunami_events_filtered.csv"),
    tsunami_runups_file: str = data_path("tsunami_runups_filtered.csv"),
    # floods inputs (adjust names if different)
    dfo_shp: str | None = None,
    dfo_attr: str | None = None,
    hanze_csv: str | None = None,
    hanze_regions_2010: str | None = None,
    hanze_regions_2021: str | None = None,
    hanze_regions_2010_layer: str | None = None,
    hanze_regions_2021_layer: str | None = None,
    hanze_code_field_2010: str | None = None,
    hanze_code_field_2021: str | None = None,
    usfd_csv: str | None = None,
    # shared
    dem_dir: str = "dem_by_iso",
    emdat_file: str = data_path("Top 5 Percent EMDAT.csv"),
    countries_path: str = os.path.join("ne_10m_admin_0_countries", "ne_10m_admin_0_countries.shp"),
    output_folder: str = "disaster_output",
    fallback_path: str = "disaster_output/fallback_layers",
    verbose_geometry_logging: bool = False,
):
    os.makedirs(output_folder, exist_ok=True)
    setup_logging(output_folder)
    logging.info(f"GDAL_DATA: {gd}")
    logging.info(f"PROJ_LIB:  {pj}")
    logging.info("Starting natural disaster pipeline...")

    # always initialize "before" frames for audits
    storms_before = gpd.GeoDataFrame()
    quakes_before = gpd.GeoDataFrame()
    tsu_before    = gpd.GeoDataFrame()
    floods_before = gpd.GeoDataFrame()

    # --- optional EM-DAT filter & which hazards to run ---
    emdat_df = pd.DataFrame()
    emdat_all = pd.DataFrame()
    emdat_in_scope = pd.DataFrame()
    if use_emdat_filtering:
        emdat_df = process_emdat_data(emdat_file)
        emdat_all = emdat_df.copy()
        # OECD scoping
        if oecd_filter == "oecd":
            emdat_in_scope = emdat_df[emdat_df["oecd_ok"] == True].copy()
        elif oecd_filter == "non_oecd":
            emdat_in_scope = emdat_df[emdat_df["oecd_ok"] == False].copy()
        else:
            emdat_in_scope = emdat_df.copy()

        # Auto-enable hazards if none explicitly requested
        haz_present = set(emdat_in_scope["haz_norm"].dropna().unique().tolist()) if not emdat_in_scope.empty else set()
        if not (run_storms or run_earthquakes or run_tsunamis or run_floods):
            run_storms      = ("storm"      in haz_present)
            run_earthquakes = ("earthquake" in haz_present)
            run_tsunamis    = ("tsunami"    in haz_present)
            run_floods      = ("flood"      in haz_present)

        if emdat_df.empty:
            logging.warning("EM-DAT empty; disabling filter.")
            use_emdat_filtering = False
    else:
        # still define these for downstream logic
        emdat_all = pd.DataFrame()
        emdat_in_scope = pd.DataFrame()

    # --- STORMS ---
    storm_gdf = gpd.GeoDataFrame(); storms_status = {"status":"empty","n_before":0,"n_after":0,"error":None}
    if run_storms:
        try:
            storm_gdf = process_ibtracs_data(ibtracs_file, output_folder)
            storms_before = storm_gdf.copy()
            storms_status["n_before"] = len(storm_gdf)

            storms_filtered = False
            if use_emdat_filtering and _nonempty(storm_gdf) and _nonempty(emdat_in_scope):
                storm_gdf = spatial_temporal_filter(
                    storm_gdf, emdat_in_scope,
                    gdf_start="start_time", gdf_end="end_time",
                    gdf_lat=None, gdf_lon=None,  # centroid of polygons
                    max_km=250.0
                )
                storms_filtered = True

            storms_status["n_after"] = len(storm_gdf)
            storms_status["status"] = "ok" if storms_status["n_before"] > 0 else "empty"
            logging.info(f"Storms {'after EM-DAT spatio-temporal filtering' if storms_filtered else 'loaded (no EM-DAT filtering)'}: {len(storm_gdf)}")
        except Exception as e:
            storms_status.update({"status":"error","error":str(e)})
            logging.exception("Storms pipeline failed.")

    # --- EARTHQUAKES ---
    earthquake_gdf = gpd.GeoDataFrame(); quakes_status = {"status":"empty","n_before":0,"n_after":0,"error":None}
    if run_earthquakes:
        try:
            earthquake_gdf = process_earthquake_data(earthquake_file, output_folder)
            quakes_before = earthquake_gdf.copy()
            quakes_status["n_before"] = len(earthquake_gdf)

            quakes_filtered = False
            if use_emdat_filtering and _nonempty(earthquake_gdf) and _nonempty(emdat_in_scope):
                earthquake_gdf = spatial_temporal_filter(
                    earthquake_gdf, emdat_in_scope,
                    gdf_start="eq_date", gdf_end="eq_date",
                    gdf_lat=("latitude" if "latitude" in earthquake_gdf.columns else None),
                    gdf_lon=("longitude" if "longitude" in earthquake_gdf.columns else None),
                    max_km=200.0
                )
                quakes_filtered = True

            quakes_status["n_after"] = len(earthquake_gdf)
            quakes_status["status"] = "ok" if quakes_status["n_before"] > 0 else "empty"
            logging.info(f"Earthquakes {'after EM-DAT spatio-temporal filtering' if quakes_filtered else 'loaded (no EM-DAT filtering)'}: {len(earthquake_gdf)}")
        except Exception as e:
            quakes_status.update({"status":"error","error":str(e)})
            logging.exception("Quake pipeline failed.")

    # --- TSUNAMIS ---
    tsunami_gdf = gpd.GeoDataFrame(); tsu_status = {"status":"empty","n_before":0,"n_after":0,"error":None}
    if run_tsunamis:
        try:
            tsunami_gdf = process_tsunami_data(
                tsunami_events_file, tsunami_runups_file,
                countries_path=countries_path,
                dem_dir=None, use_dem=True, dem_local_root=DEM_LOCAL_ROOT,
                dem_tile_size_deg=5, output_folder=output_folder, tmp_dir=output_folder,
                write_per_event=True, per_event_format="gpkg",
                write_aggregate=True, aggregate_path=os.path.join(output_folder, "tsunamis_all.gpkg"),
            )
            tsu_before = tsunami_gdf.copy()
            tsu_status["n_before"] = len(tsunami_gdf)

            tsu_filtered = False
            if use_emdat_filtering and _nonempty(tsunami_gdf) and _nonempty(emdat_in_scope):
                st_col = "start_time" if "start_time" in tsunami_gdf.columns else "date"
                en_col = "end_time"   if "end_time"   in tsunami_gdf.columns else "date"
                tsunami_gdf = spatial_temporal_filter(
                    tsunami_gdf, emdat_in_scope,
                    gdf_start=st_col, gdf_end=en_col,
                    gdf_lat=None, gdf_lon=None,  # centroid
                    max_km=250.0
                )
                tsu_filtered = True

            tsu_status["n_after"] = len(tsunami_gdf)
            tsu_status["status"] = "ok" if tsu_status["n_before"] > 0 else "empty"
            logging.info(f"Tsunamis {'after EM-DAT spatio-temporal filtering' if tsu_filtered else 'loaded (no EM-DAT filtering)'}: {len(tsunami_gdf)}")
        except Exception as e:
            tsu_status.update({"status":"error","error":str(e)})
            logging.exception("Tsunami pipeline failed.")

    # --- FLOODS ---
    flood_gdf = gpd.GeoDataFrame(); floods_status = {"status":"empty","n_before":0,"n_after":0,"error":None}
    if run_floods:
        try:
            # 1) load raw floods (DFO + HANZE + USFD), dates are start_date/end_date here
            flood_gdf = process_flood_data(
                dfo_shp=dfo_shp, dfo_attr=dfo_attr,
                hanze_csv=hanze_csv,
                hanze_regions_2010=hanze_regions_2010,
                hanze_regions_2021=hanze_regions_2021,
                hanze_regions_2010_layer=hanze_regions_2010_layer,
                hanze_regions_2021_layer=hanze_regions_2021_layer,
                hanze_code_field_2010=hanze_code_field_2010,
                hanze_code_field_2021=hanze_code_field_2021,
                usfd_csv=usfd_csv,
                simplify_tolerance_deg=0.0003,
            )
            floods_before = flood_gdf.copy()
            floods_status["n_before"] = len(flood_gdf)

            # 2) optional EM-DAT spatio-temporal filtering (use raw columns)
            floods_filtered = False
            if use_emdat_filtering and _nonempty(flood_gdf) and _nonempty(emdat_in_scope):
                flood_gdf = spatial_temporal_filter(
                    flood_gdf, emdat_in_scope,
                    gdf_start="start_date", gdf_end="end_date",   # <-- use raw date fields here
                    gdf_lat=None, gdf_lon=None,                  # centroid of polygons/points
                    max_km=250.0
                )
                floods_filtered = True

            logging.info(
                f"Floods {'after EM-DAT spatio-temporal filtering' if floods_filtered else 'loaded (no EM-DAT filtering)'}: {len(flood_gdf)}"
            )

            # 3) merge USFD+HANZE+DFO (priority) and enrich with EM-DAT (attributes only)
            em_flood = None
            if not emdat_df.empty:
                # use all EM-DAT floods for enrichment regardless of scoping/filters
                em_flood = emdat_df[emdat_df.get('haz_norm', '') == 'flood'].copy()

            flood_gdf = enrich_floods(flood_gdf, emdat_flood_df=em_flood)

            floods_status["n_after"] = len(flood_gdf)
            floods_status["status"] = "ok" if floods_status["n_before"] > 0 else "empty"
            logging.info(f"Floods merged across sources (USFD→HANZE→DFO) and enriched: {len(flood_gdf)}")

        except Exception as e:
            floods_status.update({"status":"error","error":str(e)})
            logging.exception("Flood pipeline failed.")

    # --- EM-DAT matched/unmatched CSVs + excluded features GPKG (per-hazard) ---
    if use_emdat_filtering and not emdat_df.empty:
        matched_idx = set()
        if _nonempty(storm_gdf):
            matched_idx |= audit_emdat_matches(storm_gdf, emdat_df, "start_time", "end_time", max_km=250.0)
        if _nonempty(earthquake_gdf):
            latc = "latitude" if "latitude" in earthquake_gdf.columns else None
            lonc = "longitude" if "longitude" in earthquake_gdf.columns else None
            matched_idx |= audit_emdat_matches(earthquake_gdf, emdat_df, "eq_date", "eq_date",
                                               gdf_lat=latc, gdf_lon=lonc, max_km=200.0)
        if _nonempty(tsunami_gdf):
            st_col = "start_time" if "start_time" in tsunami_gdf.columns else "date"
            en_col = "end_time"   if "end_time"   in tsunami_gdf.columns else "date"
            matched_idx |= audit_emdat_matches(tsunami_gdf, emdat_df, st_col, en_col, max_km=250.0)
        if _nonempty(flood_gdf):
            matched_idx |= audit_emdat_matches(flood_gdf, emdat_df, "start_time", "end_time", max_km=250.0)

        write_exclusions(
            output_folder, emdat_df, matched_idx,
            storms_before, storm_gdf,
            quakes_before, earthquake_gdf,
            tsu_before,    tsunami_gdf,
            floods_before, flood_gdf,
            )

    # --- EM-DAT coverage audit + splits (single block) ---
    def _matset(gdf, s_col, e_col, latc=None, lonc=None, km=250.0):
        if not _nonempty(gdf) or not _nonempty(emdat_in_scope):
            return set()
        return audit_emdat_matches(
            gdf, emdat_in_scope,  # OECD-scoped set
            s_col, e_col,
            gdf_lat=latc, gdf_lon=lonc, max_km=km
        )

    matched_storms = _matset(storm_gdf, "start_time", "end_time", km=250.0)
    matched_quakes = _matset(
        earthquake_gdf, "eq_date", "eq_date",
        latc=("latitude" if "latitude" in earthquake_gdf.columns else None),
        lonc=("longitude" if "longitude" in earthquake_gdf.columns else None),
        km=200.0
    )
    st_col = "start_time" if "start_time" in tsunami_gdf.columns else "date"
    en_col = "end_time"   if "end_time"   in tsunami_gdf.columns else "date"
    matched_tsunami = _matset(tsunami_gdf, st_col, en_col, km=250.0)
    matched_floods  = _matset(flood_gdf,  "start_time", "end_time", km=250.0)

    status_map = {
        "storm":      storms_status,
        "earthquake": quakes_status,
        "tsunami":    tsu_status,
        "flood":      floods_status,
    }
    enabled_map = {
        "storm": run_storms, "earthquake": run_earthquakes,
        "tsunami": run_tsunamis, "flood": run_floods
    }
    match_sets = {
        "storm": matched_storms, "earthquake": matched_quakes,
        "tsunami": matched_tsunami, "flood": matched_floods
    }

    rows = []
    for r in emdat_all.itertuples(index=False):
        eid = getattr(r, "event_id")
        haz = getattr(r, "haz_norm") or "other"
        iso = getattr(r, "iso3")
        start = getattr(r, "start_date")
        end   = getattr(r, "end_date")
        oecd_ok = bool(getattr(r, "oecd_ok"))

        in_scope = (oecd_filter == "all") or (oecd_filter == "oecd" and oecd_ok) or (oecd_filter == "non_oecd" and not oecd_ok)

        reason = None; detail = None; matched = False
        if not in_scope:
            reason = "oecd_excluded"
        elif haz not in status_map:
            reason = "hazard_unrecognized"
        elif not enabled_map.get(haz, False):
            reason = "hazard_disabled"
        else:
            st = status_map[haz]
            if st["status"] == "error":
                reason, detail = "pipeline_error", st["error"]
            elif st["n_before"] == 0:
                reason = "pipeline_empty"
            else:
                matched = (eid in match_sets.get(haz, set()))
                reason  = "matched" if matched else "not_matched"

        rows.append({
            "event_id": eid,
            "iso3": iso,
            "hazard": haz,
            "start_date": start,
            "end_date": end,
            "oecd_ok": oecd_ok,
            "in_scope": in_scope,
            "matched": matched,
            "reason": reason,
            "detail": detail,
            "pipeline_status": (status_map.get(haz, {}).get("status") if haz in status_map else None),
            "pipeline_n_before": (status_map.get(haz, {}).get("n_before") if haz in status_map else None),
            "pipeline_n_after": (status_map.get(haz, {}).get("n_after") if haz in status_map else None),
        })

    audit_df = pd.DataFrame(rows)
    audit_csv = os.path.join(output_folder, "emdat_audit.csv")
    audit_df.to_csv(audit_csv, index=False)
    logging.info(f"Wrote EM-DAT audit: {audit_csv}")

    # SAFE summary creation even when audit_df is empty
    if audit_df.empty:
        summary = pd.DataFrame(columns=["hazard","reason","count"])
    else:
        summary = (audit_df.groupby(["hazard","reason"]).size()
                .reset_index(name="count")
                .sort_values(["hazard","reason"]))
    summary_csv = os.path.join(output_folder, "emdat_audit_summary.csv")
    summary.to_csv(summary_csv, index=False)
    logging.info(f"Wrote EM-DAT audit summary: {summary_csv}")

    # --- Splits (already guarded, keep but fix FutureWarnings) ---
    processed_csv        = os.path.join(output_folder, "emdat_processed.csv")
    not_processed_csv    = os.path.join(output_folder, "emdat_not_processed.csv")
    out_of_scope_csv     = os.path.join(output_folder, "emdat_out_of_scope.csv")
    splits_xlsx          = os.path.join(output_folder, "emdat_audit_splits.xlsx")

    if audit_df.empty:
        base_cols = ["event_id","iso3","hazard","start_date","end_date","oecd_ok","in_scope","matched","reason","detail"]
        pd.DataFrame(columns=base_cols).to_csv(processed_csv, index=False)
        pd.DataFrame(columns=base_cols).to_csv(not_processed_csv, index=False)
        pd.DataFrame(columns=base_cols).to_csv(out_of_scope_csv, index=False)
        with pd.ExcelWriter(splits_xlsx) as xw:
            pd.DataFrame(columns=base_cols).to_excel(xw, sheet_name="processed", index=False)
            pd.DataFrame(columns=base_cols).to_excel(xw, sheet_name="not_processed", index=False)
            pd.DataFrame(columns=base_cols).to_excel(xw, sheet_name="out_of_scope", index=False)
    else:
        processed = audit_df[(audit_df["in_scope"]) & (audit_df["reason"] == "matched")].copy()
        not_proc  = audit_df[(audit_df["in_scope"]) & (audit_df["reason"] != "matched")].copy()
        out_scope = audit_df[~audit_df["in_scope"]].copy()

        processed.to_csv(processed_csv, index=False)
        not_proc.to_csv(not_processed_csv, index=False)
        out_scope.to_csv(out_of_scope_csv, index=False)

        with pd.ExcelWriter(splits_xlsx) as xw:
            processed.to_excel(xw, sheet_name="processed", index=False)
            not_proc.to_excel(xw, sheet_name="not_processed", index=False)
            out_scope.to_excel(xw, sheet_name="out_of_scope", index=False)


    # --- NORMALIZE SCHEMAS (storms/eq/tsu) ---
    storm_out = unify_schema_storms(storm_gdf)         if _nonempty(storm_gdf)        else storm_gdf
    eq_out    = unify_schema_quakes(earthquake_gdf)    if _nonempty(earthquake_gdf)   else earthquake_gdf
    tsu_out   = unify_schema_tsunamis(tsunami_gdf)     if _nonempty(tsunami_gdf)      else tsunami_gdf
    flood_gdf = unify_schema_floods(flood_gdf)         if _nonempty(flood_gdf)        else flood_gdf

    # --- WRITE PER-HAZARD outputs (normalized) ---
    if run_storms and _nonempty(storm_out):
        write_single_hazard_gdb(
            storm_out,
            output_path=join(output_folder, "storms.gdb"),
            layer_name="Storms",
            verbose_geometry_logging=verbose_geometry_logging,
        )
    if run_earthquakes and _nonempty(eq_out):
        write_single_hazard_gdb(
            eq_out,
            output_path=join(output_folder, "earthquakes.gdb"),
            layer_name="Earthquakes",
            verbose_geometry_logging=verbose_geometry_logging,
        )
    if run_tsunamis and _nonempty(tsu_out):
        write_single_hazard_gdb(
            tsu_out,
            output_path=join(output_folder, "tsunamis.gdb"),
            layer_name="Tsunamis",
            verbose_geometry_logging=verbose_geometry_logging,
        )
    if run_floods and _nonempty(flood_gdf):
        # ESRI GDB
        write_single_hazard_gdb(
            flood_gdf,
            output_path=join(output_folder, "floods.gdb"),
            layer_name="Floods",
            verbose_geometry_logging=verbose_geometry_logging,
        )

       # --- COMBINED UNIFIED LAYER (multi-layer GPKG) ---
    # We now write a single GeoPackage with separate layers per hazard
    multi_gpkg = os.path.join(output_folder, "disasters.gpkg")

    # in the _write_layer helper inside the multi-layer GPKG section:
    def _write_layer(gdf: gpd.GeoDataFrame, layer_name: str):
        if _nonempty(gdf):
            try:
                write_gpkg(gdf, multi_gpkg, layer_name, overwrite=True)  # <— overwrite layer safely
                logging.info(f"Wrote layer '{layer_name}' to {multi_gpkg} ({len(gdf)} features)")
            except Exception as e:
                logging.warning(f"Failed writing '{layer_name}' to {multi_gpkg}: {e}")


    # Use the normalized frames for storms/eq/tsu; floods already canonical from floods.py
    wrote_any = False
    if _nonempty(storm_out):
        _write_layer(storm_out, "Storms"); wrote_any = True
    if _nonempty(eq_out):
        _write_layer(eq_out, "Earthquakes"); wrote_any = True
    if _nonempty(tsu_out):
        _write_layer(tsu_out, "Tsunamis"); wrote_any = True
    if _nonempty(flood_gdf):
        _write_layer(flood_gdf, "Floods"); wrote_any = True

    if wrote_any:
        logging.info(f"Wrote multi-layer GeoPackage: {multi_gpkg}")
    else:
        logging.warning("No hazards to write to multi-layer GeoPackage.")

    # (Optional) If you still want a single-file fallback GDB when GPKG writing fails entirely,
    # keep the combiner below. It will only kick in if nothing got written above.
    if not wrote_any:
        try:
            combined_gdb = os.path.join(output_folder, "all_disasters.gdb")
            s = storm_out if _nonempty(storm_out) else gpd.GeoDataFrame()
            q = eq_out    if _nonempty(eq_out)    else gpd.GeoDataFrame()
            t = tsu_out   if _nonempty(tsu_out)   else gpd.GeoDataFrame()
            f = flood_gdf if _nonempty(flood_gdf) else gpd.GeoDataFrame()
            combine_disasters_to_gdb(
                s, q, countries_path, combined_gdb,
                tsunami_gdf=t, output_folder=output_folder,
                do_spatial_iso=do_spatial_iso,
                do_temporal_bucketing=do_temporal_bucketing,
            )
        except Exception as e:
            logging.error(f"Combined FileGDB fallback failed: {e}")


    logging.info("Pipeline completed.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Natural disasters unified pipeline")

    # ---- file/path options (accept dash OR underscore) ----
    ap.add_argument("--earthquake-file", "--earthquake_file", dest="earthquake_file")
    ap.add_argument("--ibtracs-file", "--ibtracs_file", dest="ibtracs_file")
    ap.add_argument("--tsunami-events-file", "--tsunami_events_file", dest="tsunami_events_file")
    ap.add_argument("--tsunami-runups-file", "--tsunami_runups_file", dest="tsunami_runups_file")
    ap.add_argument("--emdat-file", "--emdat_file", dest="emdat_file")
    ap.add_argument("--countries-path", "--countries_path", dest="countries_path")
    ap.add_argument("--output-folder", "--output_folder", dest="output_folder")

    # ---- OECD scope ----
    ap.add_argument("--oecd-filter", choices=["oecd","non_oecd","all"], dest="oecd_filter")

    # ---- hazard toggles (tri-state: None = don't override main() default) ----
    ap.add_argument("--run-storms", dest="run_storms", action="store_true")
    ap.add_argument("--no-run-storms", dest="run_storms", action="store_false")
    ap.add_argument("--run-earthquakes", dest="run_earthquakes", action="store_true")
    ap.add_argument("--no-run-earthquakes", dest="run_earthquakes", action="store_false")
    ap.add_argument("--run-tsunamis", dest="run_tsunamis", action="store_true")
    ap.add_argument("--no-run-tsunamis", dest="run_tsunamis", action="store_false")
    ap.add_argument("--run-floods", dest="run_floods", action="store_true")
    ap.add_argument("--no-run-floods", dest="run_floods", action="store_false")

    # ---- EM-DAT filter (tri-state) ----
    ap.add_argument("--use-emdat-filtering", dest="use_emdat_filtering", action="store_true")
    ap.add_argument("--no-use-emdat-filtering", dest="use_emdat_filtering", action="store_false")

    # defaults: None so CLI only overrides when explicitly passed
    ap.set_defaults(
        run_storms=None, run_earthquakes=None, run_tsunamis=None, run_floods=None,
        use_emdat_filtering=None, oecd_filter=None,
        earthquake_file=None, ibtracs_file=None, tsunami_events_file=None,
        tsunami_runups_file=None, emdat_file=None, countries_path=None,
        output_folder=None,
    )

    args = ap.parse_args()
    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    main(**kwargs)
