# ----
# main.py
# ----
import os, sys, logging

os.environ.setdefault("GDAL_DATA", r"C:\OSGeo4W\share\gdal")
os.environ.setdefault("PROJ_LIB",  r"C:\OSGeo4W\share\proj")

import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv
from .helpers import (
    setup_logging, write_single_hazard_gdb, combine_disasters_to_gdb, data_path, output_path,
    unify_schema_storms, unify_schema_quakes, unify_schema_tsunamis, spatial_temporal_filter,
    audit_emdat_matches, write_exclusions
)

from .storms import process_ibtracs_data
from .quakes import process_earthquake_data
from .tsunamis import process_tsunami_data
from .floods import process_flood_data

load_dotenv()
DEM_LOCAL_ROOT = os.getenv("DEM_LOCAL_ROOT", None)
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(__file__))

def has_temporal_overlap(ib_start, ib_end, em_start, em_end, tolerance_days=7):
    if pd.isna(ib_start) or pd.isna(ib_end) or pd.isna(em_start) or pd.isna(em_end):
        return False
    em_start_ext = em_start - pd.Timedelta(days=tolerance_days)
    em_end_ext   = em_end   + pd.Timedelta(days=tolerance_days)
    return max(ib_start, em_start_ext) <= min(ib_end, em_end_ext)

def process_emdat_data(emdat_csv_path: str) -> pd.DataFrame:
    logging.info("Processing EM-DAT data...")
    try:
        df = pd.read_csv(emdat_csv_path)

        def mk_date(row, prefix):
            y = row.get(f'{prefix}year')
            m = row.get(f'{prefix}month', 1)
            d = row.get(f'{prefix}day', 1)
            if pd.isna(y):
                return pd.NaT
            m = int(m) if pd.notna(m) else 1
            d = int(d) if pd.notna(d) else 1
            return pd.to_datetime(f"{int(y)}-{m}-{d}", errors="coerce")

        df["start_date"] = df.apply(lambda r: mk_date(r, "start"), axis=1)
        df["end_date"]   = df.apply(lambda r: mk_date(r, "end"), axis=1)
        df = df.dropna(subset=["start_date"])

        # Try to carry a hazard type if present; otherwise leave None
        for c in ["disaster_type","Disaster Type","type","TYPE"]:
            if c in df.columns:
                df["haz_type"] = df[c].astype(str).str.lower()
                break
        if "haz_type" not in df.columns:
            df["haz_type"] = None

        # Normalise EM lat/lon if present (best effort)
        lat_cand = [c for c in df.columns if c.lower() in ("lat","latitude","latitud","em_lat")]
        lon_cand = [c for c in df.columns if c.lower() in ("lon","longitude","longitud","em_lon")]
        if lat_cand: df["em_lat"] = pd.to_numeric(df[lat_cand[0]], errors="coerce")
        if lon_cand: df["em_lon"] = pd.to_numeric(df[lon_cand[0]], errors="coerce")

        logging.info(f"Loaded {len(df)} EM-DAT records with valid start dates.")
        return df
    except Exception as e:
        logging.error(f"EM-DAT processing failed: {e}")
        return pd.DataFrame()

def main(
    use_emdat_filtering: bool = False,
    do_spatial_iso: bool = False,
    do_temporal_bucketing: bool = False,
    run_storms: bool = False,
    run_earthquakes: bool = False,
    run_tsunamis: bool = True,
    run_floods: bool = False,
    ibtracs_file: str = data_path("ibtracs.ALL.list.v04r01.csv"),
    earthquake_file: str = data_path("earthquakes.csv"),
    tsunami_events_file: str = data_path("tsunami_events_filtered.csv"),
    tsunami_runups_file: str = data_path("tsunami_runups_filtered.csv"),
    dem_dir: str = "dem_by_iso",
    emdat_file: str = data_path("Top 5 Percent EMDAT.csv"),
    countries_path: str = os.path.join("ne_10m_admin_0_countries", "ne_10m_admin_0_countries.shp"),
    output_folder: str = "disaster_output",
    fallback_path: str = "disaster_output/fallback_layers",
    verbose_geometry_logging: bool = False,
):
    os.makedirs(output_folder, exist_ok=True)
    setup_logging(output_folder)
    logging.info("Starting natural disaster pipeline...")

    storms_before = gpd.GeoDataFrame()
    quakes_before = gpd.GeoDataFrame()
    tsu_before    = gpd.GeoDataFrame()

    # --- optional EM-DAT filter & which hazards to run ---
    emdat_df = pd.DataFrame()
    if use_emdat_filtering:
        emdat_df = process_emdat_data(emdat_file)
        if emdat_df.empty:
            logging.warning("EM-DAT empty; disabling filter.")
            use_emdat_filtering = False
        else:
            haz = set((emdat_df["haz_type"].dropna().str.lower() if "haz_type" in emdat_df else []))
            if not (run_storms or run_earthquakes or run_tsunamis):
                run_storms      = any(h in haz for h in ["storm","storm/flood","tropical cyclone","tropical storm","hurricane","typhoon","severe storm"])
                run_earthquakes = any(h in haz for h in ["earthquake"])
                run_tsunamis    = any(h in haz for h in ["tsunami"])


    # --- STORMS ---
    storm_gdf = gpd.GeoDataFrame()
    if run_storms:
        storm_gdf = process_ibtracs_data(ibtracs_file, output_folder)
        storms_before = storm_gdf.copy()
        if use_emdat_filtering and not storm_gdf.empty:
            storm_gdf = spatial_temporal_filter(
                storm_gdf, emdat_df,
                gdf_start="start_time", gdf_end="end_time",
                gdf_lat=None, gdf_lon=None,  # use centroid of polygons
                max_km=250.0
            )
            logging.info(f"Storms after EM-DAT spatio-temporal filtering: {len(storm_gdf)}")

    # --- EARTHQUAKES ---
    earthquake_gdf = gpd.GeoDataFrame()
    if run_earthquakes:
        earthquake_gdf = process_earthquake_data(earthquake_file, output_folder)
        quakes_before = earthquake_gdf.copy()
        if use_emdat_filtering and not earthquake_gdf.empty:
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
            per_event_format="gpkg",              # or "gdb"
            write_aggregate=True,
            aggregate_path=os.path.join(output_folder, "tsunamis_all.gpkg"),
        )
        tsu_before = tsunami_gdf.copy()
        if use_emdat_filtering and not tsunami_gdf.empty:
            # tsunamis now gain start/end from events merge (if available)
            st_col = "start_time" if "start_time" in tsunami_gdf.columns else "date"
            en_col = "end_time"   if "end_time" in tsunami_gdf.columns else "date"
            tsunami_gdf = spatial_temporal_filter(
                tsunami_gdf, emdat_df,
                gdf_start=st_col, gdf_end=en_col,
                gdf_lat=None, gdf_lon=None,  # centroid of polygons
                max_km=250.0
            )
            logging.info(f"Tsunamis after EM-DAT spatio-temporal filtering: {len(tsunami_gdf)}")

    # --- FLOODS ---
    flood_gdf = gpd.GeoDataFrame()
    if run_floods:
        flood_gdf = process_flood_data(
            dfo_shp=data_path("FloodArchive_region.shp"),                 # <- adjust names as in your repo
            dfo_attr=data_path("DFO_attributes.csv"),                     # optional
            hanze_csv=data_path("hanze_events.csv"),                      # optional
            hanze_regions_2010=data_path("hanze_regions_v2010.gpkg"),     # optional
            hanze_regions_2021=data_path("hanze_regions_v2021.gpkg"),     # optional
            hanze_regions_2010_layer=None,
            hanze_regions_2021_layer=None,
            hanze_code_field_2010="REG_CODE",
            hanze_code_field_2021="REG_CODE",
            usfd_csv=data_path("USFD_v1.1.csv"),                          # optional
            simplify_tolerance_deg=0.0003,
        )
        floods_before = flood_gdf.copy()
        if use_emdat_filtering and not flood_gdf.empty:
            # EM-DAT filter is handled exactly like other hazards (time + centroid proximity)
            flood_gdf = spatial_temporal_filter(
                flood_gdf, emdat_df,
                gdf_start="start_time", gdf_end="end_time",
                gdf_lat=None, gdf_lon=None,  # centroid of polygons/points
                max_km=250.0
            )

    # Build a set of EM-DAT indices matched by any kept features
    matched = set()
    if use_emdat_filtering and not emdat_df.empty:
        if run_storms and not storm_gdf.empty:
            matched |= audit_emdat_matches(storm_gdf, emdat_df, "start_time", "end_time", max_km=250.0)
        if run_earthquakes and not earthquake_gdf.empty:
            latc = "latitude" if "latitude" in earthquake_gdf.columns else None
            lonc = "longitude" if "longitude" in earthquake_gdf.columns else None
            matched |= audit_emdat_matches(earthquake_gdf, emdat_df, "eq_date", "eq_date",
                                        gdf_lat=latc, gdf_lon=lonc, max_km=200.0)
        if run_tsunamis and not tsunami_gdf.empty:
            st_col = "start_time" if "start_time" in tsunami_gdf.columns else "date"
            en_col = "end_time"   if "end_time" in tsunami_gdf.columns else "date"
            matched |= audit_emdat_matches(tsunami_gdf, emdat_df, st_col, en_col, max_km=250.0)

        # write excluded features and EM-DAT matched/unmatched CSVs
        write_exclusions(
            output_folder, emdat_df, matched,
            storms_before, storm_gdf,
            quakes_before, earthquake_gdf,
            tsu_before,    tsunami_gdf
        )


    # --- NORMALIZE SCHEMAS ---
    storm_out = unify_schema_storms(storm_gdf) if not storm_gdf.empty else storm_gdf
    eq_out    = unify_schema_quakes(earthquake_gdf) if not earthquake_gdf.empty else earthquake_gdf
    tsu_out   = unify_schema_tsunamis(tsunami_gdf) if not tsunami_gdf.empty else tsunami_gdf

    # --- WRITE PER-HAZARD GDBs (normalized) ---
    if run_storms and not storm_out.empty:
        write_single_hazard_gdb(
            storm_out,
            countries_path=countries_path,
            output_gdb=os.path.join(output_folder, "storms.gdb"),
            label="Storms",
            date_col="start_time",
            verbose_geometry_logging=verbose_geometry_logging,
        )

    if run_earthquakes and not eq_out.empty:
        write_single_hazard_gdb(
            eq_out,
            countries_path=countries_path,
            output_gdb=os.path.join(output_folder, "earthquakes.gdb"),
            label="Earthquakes",
            date_col="start_time",
            verbose_geometry_logging=verbose_geometry_logging,
        )

    if run_tsunamis and not tsu_out.empty:
        write_single_hazard_gdb(
            tsu_out,
            countries_path=countries_path,
            output_gdb=os.path.join(output_folder, "tsunamis.gdb"),
            label="Tsunamis",
            date_col="start_time",
            verbose_geometry_logging=verbose_geometry_logging,
        )

    # --- COMBINED UNIFIED LAYER ---
    if not (storm_out.empty if hasattr(storm_out, "empty") else True) or \
       not (eq_out.empty if hasattr(eq_out, "empty") else True) or \
       not (tsu_out.empty if hasattr(tsu_out, "empty") else True):
        combined = pd.concat(
            [g for g in [storm_out, eq_out, tsu_out] if hasattr(g, "empty") and not g.empty],
            ignore_index=True
        )
        combined_gpkg = os.path.join(output_folder, "all_disasters.gpkg")
        try:
            combined.to_file(combined_gpkg, driver="GPKG", layer="all_disasters")
            logging.info(f"Wrote unified combined layer: {combined_gpkg}")
        except Exception as e:
            logging.warning(f"GPKG write failed ({e}); falling back to existing combiner.")
            combined_gdb = os.path.join(output_folder, "all_disasters.gdb")
            combine_disasters_to_gdb(
                storm_gdf, earthquake_gdf, countries_path, combined_gdb,
                tsunami_gdf=tsunami_gdf, output_folder=output_folder
            )

    logging.info("Pipeline completed.")

if __name__ == "__main__":
    main()
