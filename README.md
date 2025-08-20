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
