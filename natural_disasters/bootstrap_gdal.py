# bootstrap_gdal.py
from __future__ import annotations
import os, sys, logging
from pathlib import Path

logger = logging.getLogger(__name__)
OSGEO = os.environ.get("OSGEO4W_ROOT", r"C:\OSGeo4W")

from contextlib import contextmanager

def _ensure_osgeo_importable() -> None:
    # Only add the minimal import path for GDAL Python bindings; append (don’t front-load)
    gdal_py = rf"{OSGEO}\apps\gdal\python"
    if Path(gdal_py).is_dir() and gdal_py not in sys.path:
        sys.path.append(gdal_py)

@contextmanager
def _osgeo_dll_dir():
    """Temporarily add OSGeo4W\\bin to the DLL search path for GDAL import."""
    handle = None
    bin_dir = rf"{OSGEO}\bin"
    try:
        if Path(bin_dir).is_dir() and hasattr(os, "add_dll_directory"):
            handle = os.add_dll_directory(bin_dir)
        yield
    finally:
        if handle is not None:
            try:
                handle.close()  # remove the DLL dir from the search path
            except Exception:
                pass


def _first_existing(*cands: str) -> str | None:
    for p in cands:
        if p and Path(p).is_dir():
            return p
    return None

def _diag_banner(tag: str) -> None:
    logger.info("[GDAL DIAG] %s", tag)
    logger.info("[GDAL DIAG] exe=%s", sys.executable)
    pyd = Path(OSGEO) / r"apps\Python312\Lib\site-packages\osgeo\_gdal.cp312-win_amd64.pyd"
    logger.info("[GDAL DIAG] _gdal.pyd exists? %s", pyd.exists())

def verify_gdal_ready():
    gdal_candidates = [rf"{OSGEO}\apps\gdal\share\gdal", rf"{OSGEO}\share\gdal"]
    proj_candidates = [rf"{OSGEO}\share\proj", rf"{OSGEO}\projlib"]

    gd_env = os.environ.get("GDAL_DATA")
    pj_env = os.environ.get("PROJ_LIB")

    gd = gd_env if (gd_env and Path(gd_env).is_dir()) else _first_existing(*gdal_candidates)
    pj = pj_env if (pj_env and Path(pj_env).is_dir()) else _first_existing(*proj_candidates)

    if not gd or not Path(gd).is_dir():
        raise RuntimeError(f"GDAL_DATA invalid or missing: {gd!r}")
    if not pj or not Path(pj).is_dir():
        raise RuntimeError(f"PROJ_LIB invalid or missing: {pj!r}")

    os.environ["GDAL_DATA"] = gd
    os.environ["PROJ_LIB"]  = pj

    # Import only now (no find_spec on submodules!)
    try:
        from osgeo import gdal, osr  # noqa: F401
    except Exception:
        _ensure_osgeo_importable()
        with _osgeo_dll_dir():
            _diag_banner("import with transient OSGeo DLL dir")
            from osgeo import gdal, osr  # noqa: F401

    import pyproj
    gdal.SetConfigOption("GDAL_DATA", gd)
    gdal.SetConfigOption("PROJ_LIB", pj)
    gdal.AllRegister()

    # Be explicit about OSR exception behavior to silence GDAL 4.0 warning
    try:
        osr.UseExceptions()
    except Exception:
        pass
    srs = osr.SpatialReference()
    if srs.ImportFromEPSG(4326) != 0:
        raise RuntimeError("GDAL failed to load EPSG definitions (check GDAL_DATA).")

    # PROJ smoke test
    pyproj.Transformer.from_crs(4326, 3857, always_xy=True).transform(-122.4, 37.8)
    return gd, pj
