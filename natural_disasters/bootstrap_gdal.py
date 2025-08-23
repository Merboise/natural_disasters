# natural_disasters/bootstrap_gdal.py
import os
from pathlib import Path

OSGEO = r"C:\OSGeo4W"

# set env *before any* GDAL/pyogrio import
os.environ["GDAL_DATA"] = rf"{OSGEO}\apps\gdal\share\gdal"
os.environ["PROJ_LIB"]  = rf"{OSGEO}\share\proj"

# help Windows locate GDAL/PROJ DLLs
try:
    os.add_dll_directory(rf"{OSGEO}\bin")
except Exception:
    pass

def verify_gdal_ready():
    """
    Verify GDAL/PROJ using public APIs only.
    - Confirms the env paths exist
    - Registers GDAL
    - Imports EPSG:4326 via GDAL
    - Builds a 4326->3857 transformer via PROJ (pyproj)
    Returns the resolved (GDAL_DATA, PROJ_LIB).
    """
    gd = os.environ.get("GDAL_DATA", "")
    pj = os.environ.get("PROJ_LIB", "")

    if not Path(gd).is_dir():
        raise RuntimeError(f"GDAL_DATA invalid or missing: {gd}")
    if not Path(pj).is_dir():
        raise RuntimeError(f"PROJ_LIB invalid or missing: {pj}")

    # Now touch GDAL/PROJ
    from osgeo import gdal, osr
    import pyproj

    # Make sure GDAL sees the config
    if gd: gdal.SetConfigOption("GDAL_DATA", gd)
    if pj: gdal.SetConfigOption("PROJ_LIB", pj)
    gdal.AllRegister()

    # Check EPSG load via GDAL (SRS database uses GDAL_DATA)
    srs = osr.SpatialReference()
    if srs.ImportFromEPSG(4326) != 0:
        raise RuntimeError("GDAL failed to load EPSG definitions (check GDAL_DATA).")

    # Check a simple transform via PROJ (uses pyproj data / PROJ_LIB)
    try:
        tr = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
        _x, _y = tr.transform(-122.4, 37.8)  # should succeed
    except Exception as e:
        raise RuntimeError(f"PROJ failed to create transformer (check PROJ_LIB): {e}")

    return gd, pj
