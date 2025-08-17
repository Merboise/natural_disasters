# ----
# main.py
# ----
import os, logging, pandas as pd, geopandas as gpd
from .helpers import setup_logging, combine_disasters_to_gdb
from .storms import process_ibtracs_data
from .quakes import process_earthquake_data
from .tsunamis import process_tsunami_data

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
            y=row.get(f'{prefix}year'); m=row.get(f'{prefix}month',1); d=row.get(f'{prefix}day',1)
            if pd.isna(y): return pd.NaT
            m=int(m) if pd.notna(m) else 1
            d=int(d) if pd.notna(d) else 1
            return pd.to_datetime(f"{int(y)}-{m}-{d}", errors='coerce')
        df['start_date']=df.apply(lambda r: mk_date(r,'start'), axis=1)
        df['end_date']  =df.apply(lambda r: mk_date(r,'end'  ), axis=1)
        df = df.dropna(subset=['start_date'])
        logging.info(f"Loaded {len(df)} EM-DAT records with valid start dates.")
        return df
    except Exception as e:
        logging.error(f"EM-DAT processing failed: {e}")
        return pd.DataFrame()

def main(
    use_emdat_filtering: bool = False,
    run_storms: bool = True,
    run_earthquakes: bool = True,
    run_tsunamis: bool = True,
    ibtracs_file: str = "ibtracs.ALL.list.v04r01.csv",
    earthquake_file: str = "earthquakes.csv",
    tsunami_events_file: str = "tsunami_events.csv",
    tsunami_runups_file: str = "tsunami_runup.csv",
    dem_dir: str = "dem_by_iso",
    emdat_file: str = "Top5EMDAT.csv",
    countries_path: str = r"C:\Users\WAS\Desktop\Python\projects\RESEARCH\ne_10m_admin_0_countries\ne_10m_admin_0_countries.shp",
    output_folder: str = "disaster_output",
    fallback_path: str = "disaster_output/fallback_layers",
    verbose_geometry_logging: bool = False,
):
    os.makedirs(output_folder, exist_ok=True)
    setup_logging(output_folder)
    logging.info("Starting natural disaster pipeline...")

    emdat_df = None
    if use_emdat_filtering:
        emdat_df = process_emdat_data(emdat_file)
        if emdat_df.empty:
            logging.warning("EM-DAT empty; disabling filter.")
            use_emdat_filtering = False

    storm_gdf = gpd.GeoDataFrame()
    if run_storms:
        storm_gdf = process_ibtracs_data(ibtracs_file, output_folder)
        if use_emdat_filtering and not storm_gdf.empty:
            keep=[]
            for i, s in storm_gdf.iterrows():
                overlaps = emdat_df.apply(
                    lambda r: has_temporal_overlap(s['start_time'], s['end_time'], r['start_date'], r['end_date']),
                    axis=1
                )
                if overlaps.any(): keep.append(i)
            storm_gdf = storm_gdf.loc[keep]
            logging.info(f"Storms after EM-DAT filtering: {len(storm_gdf)}")

    earthquake_gdf = gpd.GeoDataFrame()
    if run_earthquakes:
        earthquake_gdf = process_earthquake_data(earthquake_file, output_folder)
        if use_emdat_filtering and not earthquake_gdf.empty:
            keep=[]
            for i, eq in earthquake_gdf.iterrows():
                overlaps = emdat_df.apply(
                    lambda r: has_temporal_overlap(eq['date'], eq['date'], r['start_date'], r['end_date']),
                    axis=1
                )
                if overlaps.any(): keep.append(i)
            earthquake_gdf = earthquake_gdf.loc[keep]
            logging.info(f"Earthquakes after EM-DAT filtering: {len(earthquake_gdf)}")

    tsunami_gdf = gpd.GeoDataFrame()
    if run_tsunamis:
        tsunami_gdf = process_tsunami_data(
            tsunami_events_file, tsunami_runups_file,
            countries_path=countries_path, dem_dir=dem_dir,
            output_folder=output_folder, inland_limit_km=10, band_percents=(0.2,0.2)
        )

    # Separate GDBs
    if run_storms and not storm_gdf.empty:
        storm_gdb = os.path.join(output_folder, "storms.gdb")
        combine_disasters_to_gdb(storm_gdf, gpd.GeoDataFrame(), countries_path, storm_gdb,
                                 tsunami_gdf=gpd.GeoDataFrame(),
                                 output_folder=output_folder, fallback_path=fallback_path,
                                 verbose_geometry_logging=verbose_geometry_logging)

    if run_earthquakes and not earthquake_gdf.empty:
        eq_gdb = os.path.join(output_folder, "earthquakes.gdb")
        combine_disasters_to_gdb(gpd.GeoDataFrame(), earthquake_gdf, countries_path, eq_gdb,
                                 tsunami_gdf=gpd.GeoDataFrame(),
                                 output_folder=output_folder, fallback_path=fallback_path,
                                 verbose_geometry_logging=verbose_geometry_logging)

    if run_tsunamis and not tsunami_gdf.empty:
        tsu_gdb = os.path.join(output_folder, "tsunamis.gdb")
        combine_disasters_to_gdb(gpd.GeoDataFrame(), gpd.GeoDataFrame(), countries_path, tsu_gdb,
                                 tsunami_gdf=tsunami_gdf,
                                 output_folder=output_folder, fallback_path=fallback_path,
                                 verbose_geometry_logging=verbose_geometry_logging)

    # Combined GDB
    if not (storm_gdf.empty and earthquake_gdf.empty and tsunami_gdf.empty):
        combined_gdb = os.path.join(output_folder, "all_disasters.gdb")
        combine_disasters_to_gdb(storm_gdf, earthquake_gdf, countries_path, combined_gdb,
                                 tsunami_gdf=tsunami_gdf,
                                 output_folder=output_folder, fallback_path=fallback_path,
                                 verbose_geometry_logging=verbose_geometry_logging)

    logging.info("Pipeline completed.")

if __name__ == "__main__":
    main()
