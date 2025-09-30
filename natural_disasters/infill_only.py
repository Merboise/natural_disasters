# infill_only.py
from .bootstrap_gdal import verify_gdal_ready
verify_gdal_ready()

import sys, argparse, geopandas as gpd
from natural_disasters.runups import infill_bridge_rulebased
from natural_disasters.tsunami_helpers import WGS84

ap = argparse.ArgumentParser()
ap.add_argument("--gpkg", required=True)
ap.add_argument("--coast", required=True)  # coastlines shapefile or gpkg layer path
ap.add_argument("--out-layer", default="runups_infill_test")

ap.add_argument("--delta-max-km", type=float, default=250.0)
ap.add_argument("--eps-edge-km", type=float, default=10.0)

ap.add_argument("--allow-cross-parent", action="store_true")

# seed toggles
ap.add_argument("--use-rays-seeds", dest="use_rays_seeds", action="store_true", default=True)
ap.add_argument("--no-rays-seeds", dest="use_rays_seeds", action="store_false")
ap.add_argument("--use-lowht-seed", dest="use_lowht_seed", action="store_true", default=True)
ap.add_argument("--no-lowht-seed", dest="use_lowht_seed", action="store_false")

# height band for low-ht seed
ap.add_argument("--ht-lo-m", type=float, default=0.2)
ap.add_argument("--ht-hi-m", type=float, default=0.5)

ap.add_argument("--log-decisions", action="store_true")
ap.add_argument("--log-rejects-limit", type=int, default=100)
args = ap.parse_args()

snapped = gpd.read_file(args.gpkg, layer="runups_snapped")
base    = gpd.read_file(args.gpkg, layer="runups_segments_linestring")

rays = None
try:
    rays = gpd.read_file(args.gpkg, layer="rays_hits")
except Exception:
    pass

coast = gpd.read_file(args.coast)
coast_gs = coast.geometry if isinstance(coast, gpd.GeoDataFrame) else coast

out = infill_bridge_rulebased(
    snapped_gdf=snapped,
    base_segs_gdf=base,
    coast_lines_gs=coast_gs,
    ray_hits_gdf=rays,
    delta_max_km=args.delta_max_km,
    eps_edge_km=args.eps_edge_km,
    ht_lo_m=args.ht_lo_m,
    ht_hi_m=args.ht_hi_m,
    # important: enable the new guard
    allow_cross_parent=args.allow_cross_parent,
    # keep same_parent_only for completeness
    same_parent_only=not args.allow_cross_parent,
    # seed toggles
    use_rays_seeds=args.use_rays_seeds,
    use_lowht_seed=args.use_lowht_seed,
    # logging
    log_decisions=args.log_decisions,
    log_rejects_limit=args.log_rejects_limit,
)

if out is None or out.empty:
    print("no infill produced"); sys.exit(0)

out.to_file(args.gpkg, layer=args.out_layer, driver="GPKG")
print(f"wrote {len(out)} infill segments to layer '{args.out_layer}'")
