#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick validation runner for assemble_flood_data:
- Loads a small sample from each source
- Prints per-source counts and total sampled
- Runs master-id clustering + EM-DAT scope
- Writes a small GPKG + CSV into disaster_output/_dryrun

Usage (PowerShell):
  python -m natural_disasters.validate_assemble_flood_data `
    --dfo-shp "C:/.../data/DFO/FloodArchive_region.shp" `
    --dfo-attr "C:/.../data/DFO/FloodArchive.xlsx" `
    --hanze-csv "C:/.../data/HANZE/HANZE_events.csv" `
    --hanze-regions-2010 "C:/.../data/HANZE/Regions_v2010_simplified/Regions_v2010_simplified.shp" `
    --hanze-regions-2021 "C:/.../data/HANZE/Regions_v2021_simplified/Regions_v2021_simplified.shp" `
    --usfd-csv "C:/.../data/UFSD/USFD_v1.1.csv" `
    --emdat "C:/.../data/EMDAT/emdat_public_floods.csv" `
    --adm-shp "C:/.../data/gadm_410-levels.gpkg" `
    --adm-layer ADM_1 `
    --emdat-scope both `
    --sample 50
"""

from pathlib import Path
import argparse
import pandas as pd
import geopandas as gpd

# Import from package
from . import assemble_flood_data as A

def head_df(path: Path, n: int) -> Path:
    """Write a temp CSV with first n rows from any tabular (csv/xlsx) and return its path."""
    tmp = Path("disaster_output/_dryrun")
    tmp.mkdir(parents=True, exist_ok=True)
    df = A.read_tabular(path)
    df = df.head(n).copy()
    out = tmp / f"_{path.stem}_sample.csv"
    df.to_csv(out, index=False)
    return out

def main():
    ap = argparse.ArgumentParser(description="Validate assemble_flood_data with small samples.")
    ap.add_argument("--dfo-shp", type=Path, required=False)
    ap.add_argument("--dfo-attr", type=Path, required=False)

    ap.add_argument("--hanze-csv", type=Path, required=False)
    ap.add_argument("--hanze-regions-2010", type=Path, required=False)
    ap.add_argument("--hanze-regions-2010-layer", type=str, required=False)
    ap.add_argument("--hanze-regions-2021", type=Path, required=False)
    ap.add_argument("--hanze-regions-2021-layer", type=str, required=False)

    ap.add_argument("--usfd-csv", type=Path, required=False)
    ap.add_argument("--emdat", type=Path, required=False)

    ap.add_argument("--adm-shp", type=Path, required=True)
    ap.add_argument("--adm-layer", type=str, required=False)

    ap.add_argument("--emdat-scope", choices=["both", "only", "non"], default="both")
    ap.add_argument("--sample", type=int, default=50)
    args = ap.parse_args()

    outdir = Path("disaster_output/_dryrun")
    outdir.mkdir(parents=True, exist_ok=True)

    # Admin boundaries
    admin_gdf = A.prep_admin(args.adm_shp, args.adm_layer, ["NAME_1", "NAME_2", "NAME_ENGLI"])

    records = []

    # --- DFO sample ---
    if args.dfo_shp and args.dfo_shp.exists():
        # read only first N features from the shp to reduce IO
        gdf_dfo = gpd.read_file(args.dfo_shp, rows=slice(0, args.sample))
        tmp_shp = outdir / "_dfo_sample.shp"
        gdf_dfo.to_file(tmp_shp)
        # write a small attr table if provided
        dfo_attr = args.dfo_attr if args.dfo_attr and args.dfo_attr.exists() else None
        print(f"[DFO] sampled {len(gdf_dfo)} -> loading")
        records += A.load_dfo(tmp_shp, dfo_attr)

    # --- HANZE sample ---
    if args.hanze_csv and args.hanze_csv.exists():
        temp_h = head_df(args.hanze_csv, args.sample)
        print(f"[HANZE] sampled {A.read_tabular(args.hanze_csv).head(args.sample).shape[0]} rows")
        records += A.load_hanze_csv_to_polys(
            events_csv=temp_h,
            regions_v2010_path=args.hanze_regions_2010,
            regions_v2021_path=args.hanze_regions_2021,
            regions_layer_2010=args.hanze_regions_2010_layer,
            regions_layer_2021=args.hanze_regions_2021_layer,
        )

    # --- USFD sample ---
    if args.usfd_csv and args.usfd_csv.exists():
        temp_u = head_df(args.usfd_csv, args.sample)
        print(f"[USFD] sampled {A.read_tabular(args.usfd_csv).head(args.sample).shape[0]} rows")
        records += A.load_usfd_v11(temp_u)

    # --- EMDAT sample ---
    if args.emdat and args.emdat.exists():
        try:
            temp_e = head_df(args.emdat, args.sample)
            n_e = A.read_tabular(args.emdat).head(args.sample).shape[0]
            print(f"[EMDAT] sampled {n_e} rows")
            records += A.load_emdat_table(temp_e, admin_gdf)
        except Exception as e:
            print(f"[EMDAT] ERROR: {e}\nHint: if .xlsx, install openpyxl or export to CSV.")

    # ---- Per-source breakdown + total ----
    if not records:
        raise SystemExit("No records ingested. Check your paths/exports.")

    src_counts = pd.Series([r["source"] for r in records]).value_counts()
    print(f"[LOAD] total sampled records: {len(records)}")
    print(src_counts.to_string())

    # Build master GeoDataFrame
    gdf = gpd.GeoDataFrame(records, crs=A.WGS84)

    # Cluster & scope
    gdf_master = A.assign_master_ids(gdf)
    gdf_master = A.add_emdat_anchor(gdf_master)
    print("emdat_anchor\n" + gdf_master["emdat_anchor"].value_counts().to_string())

    scoped = A.filter_by_emdat_scope(gdf_master, args.emdat_scope)
    print(f"[SCOPE={args.emdat_scope}] rows: {len(scoped)}, clusters: {scoped['master_id'].nunique()}")

    # Write small artifacts
    master_csv = outdir / f"master_events_{args.emdat_scope}_sample.csv"
    scoped.assign(
        centroid_lon=scoped.geometry.apply(lambda g: g.representative_point().x if g is not None else None),
        centroid_lat=scoped.geometry.apply(lambda g: g.representative_point().y if g is not None else None),
    ).drop(columns="geometry").to_csv(master_csv, index=False)
    print(f"[WRITE] master sample CSV -> {master_csv}")

    gpkg = outdir / f"consolidated_{args.emdat_scope}_sample.gpkg"
    A.consolidate_events(scoped).to_file(gpkg, layer="flood_events", driver="GPKG")
    print(f"[WRITE] consolidated sample GPKG -> {gpkg}")

if __name__ == "__main__":
    main()
