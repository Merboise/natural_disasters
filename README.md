## Elevation data (MERIT Hydro, local-only)

This project can optionally use local MERIT Hydro tiles to refine tsunami inundation.
Because MERIT requires credentials, we **do not** ship DEMs in the repo.

Folder structure example (not committed):

C:\Users\<YOU>\...\Elevation\
  elv_n60w090\
    n60w075_elv.tif
    n60w080_elv.tif
  elv_n30e120\
    n30e120_elv.tif
    ...

File names must contain a lower-left code like `n60w075` or `s10e135`.
By default we assume 5°×5° tiles; adjust `dem_tile_size_deg` if yours differ.

To enable DEM usage in the pipeline, run `main.py` with:
- `run_tsunamis=True`
- `use_dem=True`
- `dem_local_root="C:\\Users\\<YOU>\\...\\Elevation"`

# Data folder (not tracked in git)

Place required datasets here:

- `Top 5 Percent EMDAT.csv` – your EM-DAT subset/export
- `DFO/FloodArchive_region.shp` (plus .dbf/.shx/.prj) – DFO polygons
- `USFD/USFD_v1.0.csv` – USFD events
- `HANZE/hanze_regions.shp` + `HANZE/hanze_events.csv` – HANZE data
- `DEM/…` – local DEM tiles (GeoTIFFs)
- `GHSL/GHS_POP_2020.tif` – GHSL population

You can override the data location with an environment variable:

```bash
export NATDIS_DATA_DIR=/absolute/path/to/data
