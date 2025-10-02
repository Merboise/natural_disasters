# infill_only.py
from .bootstrap_gdal import verify_gdal_ready
verify_gdal_ready()

import sys, argparse, geopandas as gpd
from natural_disasters.runups import infill_bridge_rulebased, build_runup_segments, infill_runup_segments
from natural_disasters.tsunami_helpers import WGS84

ap = argparse.ArgumentParser()
ap.add_argument("--gpkg", required=True)
ap.add_argument("--coast", required=True)  # coastlines shapefile or gpkg layer path
ap.add_argument("--out-layer", default="runups_infill_test")

ap.add_argument("--mode", choices=["segments", "rulebased", "both"], default="rulebased",
                help="Which infill to run: low-Ht segments, rulebased bridging, or both")
ap.add_argument("--out-layer-segments", default=None,
                help="Optional layer name for segments infill when --mode=segments or both")
ap.add_argument("--out-layer-rulebased", default=None,
                help="Optional layer name for rulebased infill when --mode=rulebased or both")

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

# infill_runup_segments knobs (no config object)
ap.add_argument("--seg-eps-km", type=float, default=15.0,
                help="1D clustering epsilon for low-Ht stations")
ap.add_argument("--seg-min-samples", type=int, default=2)
ap.add_argument("--seg-majority", type=float, default=0.5,
                help="Required base coverage inside cluster [0..1]")
ap.add_argument("--seg-same-parent-only", action="store_true",
                help="Only propose clusters that bridge within same parent")
ap.add_argument("--seg-max-bridge-gap-km", type=float, default=500.0)
ap.add_argument("--seg-min-combined-coverage", type=float, default=0.6)
ap.add_argument("--seg-accept-singleton", action="store_true",
                help="Allow a 1-point cluster to form a tiny segment if near base")
ap.add_argument("--seg-allow-zero-cover", action="store_true",
                help="Allow bridging even if cluster itself has 0 base cover when gap≤limit")
ap.add_argument("--seg-bridge-one-side", action="store_true",
                help="Permit bridging when coverage exists only on one side")
ap.add_argument("--seg-seed-without-base", action="store_true",
                help="Allow seeding a minimal segment even with no base on that parent")
ap.add_argument("--seg-seed-min-km", type=float, default=1.0,
                help="Minimum seeded segment length when --seg-seed-without-base")

ap.add_argument("--log-decisions", action="store_true")
ap.add_argument("--log-rejects-limit", type=int, default=100)
ap.add_argument("--debug", action="store_true", help="Enable DEBUG level logging")
args = ap.parse_args()

import logging
level = logging.DEBUG if args.debug else logging.INFO
logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

snapped = gpd.read_file(args.gpkg, layer="runups_snapped")
base    = gpd.read_file(args.gpkg, layer="runups_segments_linestring")

rays = None
try:
    rays = gpd.read_file(args.gpkg, layer="rays_hits")
except Exception:
    pass

coast = gpd.read_file(args.coast)
coast_gs = coast.geometry if isinstance(coast, gpd.GeoDataFrame) else coast

# ---- Run selected mode(s) ----
wrote_any = False
if args.mode in ("segments", "both"):
    from types import SimpleNamespace
    seg_opts = SimpleNamespace(
        allow_zero_cover_bridge=args.seg_allow_zero_cover,
        bridge_one_side=args.seg_bridge_one_side,
        accept_singleton=args.seg_accept_singleton,
        seed_without_base=args.seg_seed_without_base,
        seed_min_km=args.seg_seed_min_km,
        log_rejects_limit=args.log_rejects_limit,
    )
    seg_out = infill_runup_segments(
        snapped_gdf=snapped,
        base_segs_gdf=base,
        coast_lines_gs=coast_gs,
        ht_lo_m=args.ht_lo_m,
        ht_hi_m=args.ht_hi_m,
        eps_km=args.seg_eps_km,
        min_samples=args.seg_min_samples,
        majority_threshold=args.seg_majority,
        same_parent_only=args.seg_same_parent_only,
        max_bridge_gap_km=args.seg_max_bridge_gap_km,
        min_combined_coverage=args.seg_min_combined_coverage,
        opts=seg_opts,  # no InfillConfig; plain namespace
    )
    if seg_out is not None and not seg_out.empty:
        seg_layer = args.out_layer_segments or (args.out_layer if args.mode=="segments" else f"{args.out_layer}_segments")
        seg_out.to_file(args.gpkg, layer=seg_layer, driver="GPKG")
        print(f"wrote {len(seg_out)} segments to layer '{seg_layer}'")
        wrote_any = True

if args.mode in ("rulebased", "both"):
    rb_out = infill_bridge_rulebased(
        snapped_gdf=snapped,
        base_segs_gdf=base,
        coast_lines_gs=coast_gs,
        ray_hits_gdf=rays,
        delta_max_km=args.delta_max_km,
        eps_edge_km=args.eps_edge_km,
        ht_lo_m=args.ht_lo_m,
        ht_hi_m=args.ht_hi_m,
        allow_cross_parent=args.allow_cross_parent,
        same_parent_only=not args.allow_cross_parent,
        use_rays_seeds=args.use_rays_seeds,
        use_lowht_seed=args.use_lowht_seed,
        log_decisions=args.log_decisions,
        log_rejects_limit=args.log_rejects_limit,
    )
    if rb_out is not None and not rb_out.empty:
        rb_layer = args.out_layer_rulebased or (args.out_layer if args.mode=="rulebased" else f"{args.out_layer}_rulebased")
        rb_out.to_file(args.gpkg, layer=rb_layer, driver="GPKG")
        print(f"wrote {len(rb_out)} bridges to layer '{rb_layer}'")
        wrote_any = True

if not wrote_any:
    print("no infill produced"); sys.exit(0)