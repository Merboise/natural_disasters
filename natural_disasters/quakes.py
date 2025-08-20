# ----
# quakes.py
# ----
import os, logging, numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point

def process_earthquake_data(csv_path: str, output_folder: str) -> gpd.GeoDataFrame:
    logging.info("Processing earthquake data...")
    os.makedirs(output_folder, exist_ok=True)
    try:
        df = pd.read_csv(csv_path, parse_dates=['date'])
        df = df.rename(columns={'date': 'eq_date_orig'})
        out_rows=[]
        for _, row in df.iterrows():
            point = Point(row['longitude'], row['latitude'])
            gdf = gpd.GeoDataFrame([row], geometry=[point], crs="EPSG:4326")
            gdf = gdf.to_crs("EPSG:3395")
            radius_m = float(row['earthqk_radius']) * 1609.34
            gdf['geometry'] = gdf.buffer(radius_m)
            gdf = gdf.rename(columns={'earthqk_radius':'radius_mil','eq_date_orig':'eq_date'})
            out_rows.append(gdf)
        final_gdf = gpd.GeoDataFrame(pd.concat(out_rows, ignore_index=True)).to_crs("EPSG:4326")
        logging.info(f"Processed {len(final_gdf)} earthquake buffers.")
        return final_gdf
    except Exception as e:
        logging.error(f"Earthquake processing failed: {e}")
        return gpd.GeoDataFrame()
