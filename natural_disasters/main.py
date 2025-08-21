# ----
# main.py
# ----
import os, sys, logging
import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv
from .helpers import setup_logging, write_single_hazard_gdb, combine_disasters_to_gdb, data_path, output_path
from .storms import process_ibtracs_data
from .quakes import process_earthquake_data
from .tsunamis import process_tsunami_data

os.environ.setdefault("GDAL_DATA", r"C:\OSGeo4W\share\gdal")
os.environ.setdefault("PROJ_LIB",  r"C:\OSGeo4W\share\proj")

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

    # --- optional EM-DAT filter ---
    emdat_df = None
    if use_emdat_filtering:
        emdat_df = process_emdat_data(emdat_file)
        if emdat_df.empty:
            logging.warning("EM-DAT empty; disabling filter.")
            use_emdat_filtering = False

    # --- STORMS ---
    storm_gdf = gpd.GeoDataFrame()
    if run_storms:
        storm_gdf = process_ibtracs_data(ibtracs_file, output_folder)
        if use_emdat_filtering and not storm_gdf.empty:
            keep = []
            for i, s in storm_gdf.iterrows():
                overlaps = emdat_df.apply(
                    lambda r: has_temporal_overlap(
                        s["start_time"], s["end_time"], r["start_date"], r["end_date"]
                    ),
                    axis=1,
                )
                if overlaps.any():
                    keep.append(i)
            storm_gdf = storm_gdf.loc[keep]
            logging.info(f"Storms after EM-DAT filtering: {len(storm_gdf)}")

    # --- EARTHQUAKES ---
    earthquake_gdf = gpd.GeoDataFrame()
    if run_earthquakes:
        earthquake_gdf = process_earthquake_data(earthquake_file, output_folder)
        if use_emdat_filtering and not earthquake_gdf.empty:
            keep = []
            for i, eq in earthquake_gdf.iterrows():
                overlaps = emdat_df.apply(
                    lambda r: has_temporal_overlap(eq["date"], eq["date"], r["start_date"], r["end_date"]),
                    axis=1,
                )
                if overlaps.any():
                    keep.append(i)
            earthquake_gdf = earthquake_gdf.loc[keep]
            logging.info(f"Earthquakes after EM-DAT filtering: {len(earthquake_gdf)}")

    # --- TSUNAMIS ---
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


    # --- WRITE PER-HAZARD GDBs (single-hazard writer) ---
    if run_storms and not storm_gdf.empty:
        storm_out = storm_gdf.copy()
        storm_out["event_type"] = "storm"
        storm_gdb_path = os.path.join(output_folder, "storms.gdb")
        write_single_hazard_gdb(
            storm_out,
            countries_path=countries_path,
            output_gdb=storm_gdb_path,
            label="Storms",
            date_col="storm_date",
            verbose_geometry_logging=verbose_geometry_logging,
        )

    if run_earthquakes and not earthquake_gdf.empty:
        eq_out = earthquake_gdf.copy()
        eq_out["event_type"] = "earthquake"
        earthquake_gdb_path = os.path.join(output_folder, "earthquakes.gdb")
        write_single_hazard_gdb(
            eq_out,
            countries_path=countries_path,
            output_gdb=earthquake_gdb_path,
            label="Earthquakes",
            date_col="eq_date",
            verbose_geometry_logging=verbose_geometry_logging,
        )

    if run_tsunamis and not tsunami_gdf.empty:
        tsu_out = tsunami_gdf.copy()
        tsu_out["event_type"] = "tsunami"
        tsunami_gdb_path = os.path.join(output_folder, "tsunamis.gdb")
        write_single_hazard_gdb(
            tsu_out,
            countries_path=countries_path,
            output_gdb=tsunami_gdb_path,
            label="Tsunamis",
            date_col="date",
            verbose_geometry_logging=verbose_geometry_logging,
        )

    # --- OPTIONAL: WRITE A COMBINED GDB ---
    if not (storm_gdf.empty and earthquake_gdf.empty and tsunami_gdf.empty):
        combined_gdb = os.path.join(output_folder, "all_disasters.gdb")
        combine_disasters_to_gdb(
            storm_gdf,
            earthquake_gdf,
            countries_path,
            combined_gdb,
            tsunami_gdf=tsunami_gdf,
            output_folder=output_folder,
        )

    logging.info("Pipeline completed.")


if __name__ == "__main__":
    main()
