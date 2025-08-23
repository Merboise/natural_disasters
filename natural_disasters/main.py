# ----
# main.py
# ----
import os, sys, logging

# Must be first: configure env before any geospatial import happens
from .bootstrap_gdal import verify_gdal_ready
gd, pj = verify_gdal_ready()   # you can log these if you want


import geopandas as gpd
import pyogrio
import pandas as pd
from dotenv import load_dotenv

from .helpers import (
    setup_logging, write_single_hazard_gdb, combine_disasters_to_gdb, data_path, output_path,
    unify_schema_storms, unify_schema_quakes, unify_schema_tsunamis, spatial_temporal_filter,
    audit_emdat_matches, write_exclusions,
)
from .storms import process_ibtracs_data
from .quakes import process_earthquake_data
from .tsunamis import process_tsunami_data
from .floods import process_flood_data  # <- new

load_dotenv()
DEM_LOCAL_ROOT = os.getenv("DEM_LOCAL_ROOT", None)
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(__file__))

# ---------- EM-DAT loader ----------
def process_emdat_data(emdat_csv_path: str) -> pd.DataFrame:
    logging.info("Processing EM-DAT data...")
    try:
        df = pd.read_csv(emdat_csv_path)

        def mk_date(row, prefix):
            y = row.get(f"{prefix}year")
            m = row.get(f"{prefix}month", 1)
            d = row.get(f"{prefix}day", 1)
            if pd.isna(y):
                return pd.NaT
            m = int(m) if pd.notna(m) else 1
            d = int(d) if pd.notna(d) else 1
            return pd.to_datetime(f"{int(y)}-{m}-{d}", errors="coerce")

        df["start_date"] = df.apply(lambda r: mk_date(r, "start"), axis=1)
        df["end_date"]   = df.apply(lambda r: mk_date(r, "end"), axis=1)
        df = df.dropna(subset=["start_date"])

        # hazard type (best effort)
        for c in ["disaster_type","Disaster Type","type","TYPE"]:
            if c in df.columns:
                df["haz_type"] = df[c].astype(str).str.lower()
                break
        if "haz_type" not in df.columns:
            df["haz_type"] = None

        # lat/lon (best effort)
        lat_cand = [c for c in df.columns if c.lower() in ("lat","latitude","latitud","em_lat")]
        lon_cand = [c for c in df.columns if c.lower() in ("lon","longitude","longitud","em_lon")]
        if lat_cand: df["em_lat"] = pd.to_numeric(df[lat_cand[0]], errors="coerce")
        if lon_cand: df["em_lon"] = pd.to_numeric(df[lon_cand[0]], errors="coerce")

        logging.info(f"Loaded {len(df)} EM-DAT records with valid start dates.")
        return df
    except Exception as e:
        logging.error(f"EM-DAT processing failed: {e}")
        return pd.DataFrame()

# small helper
def _nonempty(g):
    return hasattr(g, "empty") and not g.empty

def main(
    use_emdat_filtering: bool = False,
    do_spatial_iso: bool = False,
    do_temporal_bucketing: bool = False,
    run_storms: bool = False,
    run_earthquakes: bool = False,
    run_tsunamis: bool = True,
    run_floods: bool = False,
    # inputs
    ibtracs_file: str = data_path("ibtracs.ALL.list.v04r01.csv"),
    earthquake_file: str = data_path("earthquakes.csv"),
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
    if use_emdat_filtering:
        emdat_df = process_emdat_data(emdat_file)
        if emdat_df.empty:
            logging.warning("EM-DAT empty; disabling filter.")
            use_emdat_filtering = False
        else:
            haz = set((emdat_df["haz_type"].dropna().str.lower() if "haz_type" in emdat_df else []))
            # auto-enable hazards if none explicitly requested
            if not (run_storms or run_earthquakes or run_tsunamis or run_floods):
                run_storms      = any(h in haz for h in ["storm","storm/flood","tropical cyclone","tropical storm","hurricane","typhoon","severe storm"])
                run_earthquakes = any(h in haz for h in ["earthquake"])
                run_tsunamis    = any(h in haz for h in ["tsunami"])
                run_floods      = any(h in haz for h in ["flood","flash flood","coastal flood","riverine flood"])

    # --- STORMS ---
    storm_gdf = gpd.GeoDataFrame()
    if run_storms:
        storm_gdf = process_ibtracs_data(ibtracs_file, output_folder)
        storms_before = storm_gdf.copy()
        if use_emdat_filtering and _nonempty(storm_gdf):
            storm_gdf = spatial_temporal_filter(
                storm_gdf, emdat_df,
                gdf_start="start_time", gdf_end="end_time",
                gdf_lat=None, gdf_lon=None,  # centroid of polygons
                max_km=250.0
            )
            logging.info(f"Storms after EM-DAT spatio-temporal filtering: {len(storm_gdf)}")

    # --- EARTHQUAKES ---
    earthquake_gdf = gpd.GeoDataFrame()
    if run_earthquakes:
        earthquake_gdf = process_earthquake_data(earthquake_file, output_folder)
        quakes_before = earthquake_gdf.copy()
        if use_emdat_filtering and _nonempty(earthquake_gdf):
            earthquake_gdf = spatial_temporal_filter(
                earthquake_gdf, emdat_df,
                gdf_start="eq_date", gdf_end="eq_date",
                gdf_lat=("latitude" if "latitude" in earthquake_gdf.columns else None),
                gdf_lon=("longitude" if "longitude" in earthquake_gdf.columns else None),
                max_km=200.0
            )
            logging.info(f"Earthquakes after EM-DAT spatio-temporal filtering: {len(earthquake_gdf)}")

    # --- TSUNAMIS ---
    tsunami_gdf = gpd.GeoDataFrame()
    if run_tsunamis:
        tsunami_gdf = process_tsunami_data(
            tsunami_events_file,
            tsunami_runups_file,
            countries_path=countries_path,
            dem_dir=None,
            use_dem=True,
            dem_local_root=DEM_LOCAL_ROOT,
            dem_tile_size_deg=5,
            output_folder=output_folder,
            tmp_dir=output_folder,
            write_per_event=True,
            per_event_format="gpkg",  # or "gdb"
            write_aggregate=True,
            aggregate_path=os.path.join(output_folder, "tsunamis_all.gpkg"),
        )
        tsu_before = tsunami_gdf.copy()
        if use_emdat_filtering and _nonempty(tsunami_gdf):
            st_col = "start_time" if "start_time" in tsunami_gdf.columns else "date"
            en_col = "end_time"   if "end_time"   in tsunami_gdf.columns else "date"
            tsunami_gdf = spatial_temporal_filter(
                tsunami_gdf, emdat_df,
                gdf_start=st_col, gdf_end=en_col,
                gdf_lat=None, gdf_lon=None,  # centroid
                max_km=250.0
            )
            logging.info(f"Tsunamis after EM-DAT spatio-temporal filtering: {len(tsunami_gdf)}")

    # --- FLOODS ---
    flood_gdf = gpd.GeoDataFrame()
    if run_floods:
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
        if use_emdat_filtering and _nonempty(flood_gdf):
            flood_gdf = spatial_temporal_filter(
                flood_gdf, emdat_df,
                gdf_start="start_time", gdf_end="end_time",
                gdf_lat=None, gdf_lon=None,  # centroid of polygons/points
                max_km=250.0
            )
            logging.info(f"Floods after EM-DAT spatio-temporal filtering: {len(flood_gdf)}")

    # --- EM-DAT audit: which EM-DAT rows matched any kept features? ---
    matched = set()
    if use_emdat_filtering and not emdat_df.empty:
        if _nonempty(storm_gdf):
            matched |= audit_emdat_matches(storm_gdf, emdat_df, "start_time", "end_time", max_km=250.0)
        if _nonempty(earthquake_gdf):
            latc = "latitude" if "latitude" in earthquake_gdf.columns else None
            lonc = "longitude" if "longitude" in earthquake_gdf.columns else None
            matched |= audit_emdat_matches(earthquake_gdf, emdat_df, "eq_date", "eq_date",
                                           gdf_lat=latc, gdf_lon=lonc, max_km=200.0)
        if _nonempty(tsunami_gdf):
            st_col = "start_time" if "start_time" in tsunami_gdf.columns else "date"
            en_col = "end_time"   if "end_time"   in tsunami_gdf.columns else "date"
            matched |= audit_emdat_matches(tsunami_gdf, emdat_df, st_col, en_col, max_km=250.0)
        if _nonempty(flood_gdf):
            matched |= audit_emdat_matches(flood_gdf, emdat_df, "start_time", "end_time", max_km=250.0)

        # write excluded features & EM-DAT matched/unmatched
        write_exclusions(
            output_folder, emdat_df, matched,
            storms_before, storm_gdf,
            quakes_before, earthquake_gdf,
            tsu_before,    tsunami_gdf,
            floods_before=floods_before, floods_after=flood_gdf,  # <-- helpers supports these kwargs
        )

    # --- NORMALIZE SCHEMAS (storms/eq/tsu) ---
    storm_out = unify_schema_storms(storm_gdf)         if _nonempty(storm_gdf)        else storm_gdf
    eq_out    = unify_schema_quakes(earthquake_gdf)    if _nonempty(earthquake_gdf)   else earthquake_gdf
    tsu_out   = unify_schema_tsunamis(tsunami_gdf)     if _nonempty(tsunami_gdf)      else tsunami_gdf
    # flood_gdf already canonical (floods.py produces start_time/end_time/etc.)

    # --- WRITE PER-HAZARD GDBs (normalized) ---
    if run_storms and _nonempty(storm_out):
        write_single_hazard_gdb(
            storm_out,
            countries_path=countries_path,
            output_gdb=os.path.join(output_folder, "storms.gdb"),
            label="Storms",
            date_col="start_time",
            verbose_geometry_logging=verbose_geometry_logging,
        )

    if run_earthquakes and _nonempty(eq_out):
        write_single_hazard_gdb(
            eq_out,
            countries_path=countries_path,
            output_gdb=os.path.join(output_folder, "earthquakes.gdb"),
            label="Earthquakes",
            date_col="start_time",
            verbose_geometry_logging=verbose_geometry_logging,
        )

    if run_tsunamis and _nonempty(tsu_out):
        write_single_hazard_gdb(
            tsu_out,
            countries_path=countries_path,
            output_gdb=os.path.join(output_folder, "tsunamis.gdb"),
            label="Tsunamis",
            date_col="start_time",
            verbose_geometry_logging=verbose_geometry_logging,
        )

    if run_floods and _nonempty(flood_gdf):
        write_single_hazard_gdb(
            flood_gdf,
            countries_path=countries_path,
            output_gdb=os.path.join(output_folder, "floods.gdb"),
            label="Floods",
            date_col="start_time",
            verbose_geometry_logging=verbose_geometry_logging,
        )

    # --- COMBINED UNIFIED LAYER (normalized frames) ---
    parts = [g for g in [storm_out, eq_out, tsu_out, flood_gdf] if _nonempty(g)]
    if parts:
        combined = pd.concat(parts, ignore_index=True)
        combined_gpkg = os.path.join(output_folder, "all_disasters.gpkg")
        try:
            combined.to_file(combined_gpkg, driver="GPKG", layer="all_disasters")
            logging.info(f"Wrote unified combined layer: {combined_gpkg}")
        except Exception as e:
            logging.warning(f"GPKG write failed ({e}); falling back to existing combiner.")
            combined_gdb = os.path.join(output_folder, "all_disasters.gdb")
            # pass normalized frames when possible
            s = storm_out if _nonempty(storm_out) else gpd.GeoDataFrame()
            q = eq_out    if _nonempty(eq_out)    else gpd.GeoDataFrame()
            t = tsu_out   if _nonempty(tsu_out)   else gpd.GeoDataFrame()
            f = flood_gdf if _nonempty(flood_gdf) else gpd.GeoDataFrame()
            # combine_disasters_to_gdb signature expects storms/eq/tsunami; if you extend it to floods, wire it there
            combine_disasters_to_gdb(
                s, q, countries_path, combined_gdb,
                tsunami_gdf=t, output_folder=output_folder
            )

    logging.info("Pipeline completed.")

if __name__ == "__main__":
    main()
