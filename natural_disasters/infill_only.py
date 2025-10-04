# infill_only.py — unified mode
from .bootstrap_gdal import verify_gdal_ready
verify_gdal_ready()

import sys, argparse, logging, geopandas as gpd
from natural_disasters.runups import (
    infill_coast_unified
)

# ---------------- CLI ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--gpkg", required=True)
ap.add_argument("--coast", required=True)  # coastlines shapefile or gpkg layer path
ap.add_argument("--out-layer", default="runups_infill_unified")

ap.add_argument("--mode", choices=["unified", "rulebased", "segments", "both"], default="unified",
                help="unified = single-pass seeding+bridging; others kept for backward compatibility")

# Core knobs
ap.add_argument("--delta-max-km", type=float, default=750.0)
ap.add_argument("--pad-km", type=float, default=1.0, help="padding used for merges, cuts, and near-edge tests")

# Cross-parent defaults to ON; allow disable
xgrp = ap.add_mutually_exclusive_group()
xgrp.add_argument("--allow-cross-parent", dest="allow_cross_parent", action="store_true")
xgrp.add_argument("--no-cross-parent",  dest="allow_cross_parent", action="store_false")
ap.set_defaults(allow_cross_parent=True)

# Amplifiers (optional)
ap.add_argument("--use-rays-seeds", dest="use_rays_seeds", action="store_true", default=True)
ap.add_argument("--no-rays-seeds",  dest="use_rays_seeds", action="store_false")
ap.add_argument("--use-lowht-seed", dest="use_lowht_seed", action="store_true", default=True)
ap.add_argument("--no-lowht-seed",  dest="use_lowht_seed", action="store_false")

ap.add_argument("--ht-lo-m", type=float, default=0.2)
ap.add_argument("--ht-hi-m", type=float, default=0.5)

ap.add_argument("--graph-snap-m", type=float, default=250.0,
                help="snap tolerance (meters) when building coast graph")

# Seed-from-runups parameters (height -> length)
ap.add_argument("--seed-height-min-m", type=float, default=0.5)
ap.add_argument("--seed-L-min-km",     type=float, default=80.0)
ap.add_argument("--seed-L-max-km",     type=float, default=250.0)
ap.add_argument("--seed-km-per-meter", type=float, default=80.0)
ap.add_argument("--seed-height-exp",   type=float, default=1.0)
ap.add_argument("--seed-length-mult",  type=float, default=1.0)
ap.add_argument("--seed-merge-tol-m",  type=float, default=2000.0)

ap.add_argument("--hop-max-km", type=float, default=None,
                help="max coastal distance per hop; default=--delta-max-km")
ap.add_argument("--total-max-km", type=float, default=None,
                help="total chain cap; default=--delta-max-km")
ap.add_argument("--k-neighbors", type=int, default=6,
                help="candidate neighbors per node")
ap.add_argument("--passes", type=int, default=3,
                help="daisy-chain passes")

# Logging
ap.add_argument("--log-decisions", action="store_true")
ap.add_argument("--log-rejects-limit", type=int, default=100)
ap.add_argument("--debug", action="store_true", help="Enable DEBUG level logging")
args = ap.parse_args()

level = logging.DEBUG if args.debug else logging.INFO
logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

# ---------------- IO ----------------
snapped = gpd.read_file(args.gpkg, layer="runups_snapped")
base    = gpd.read_file(args.gpkg, layer="runups_segments_linestring")

rays = None
try:
    rays = gpd.read_file(args.gpkg, layer="rays_hits")
except Exception:
    pass

coast = gpd.read_file(args.coast)
coast_gs = coast.geometry if isinstance(coast, gpd.GeoDataFrame) else coast

# ---------------- Execute ----------------
wrote_any = False

if args.mode == "unified":
    out = infill_coast_unified(
        snapped_gdf=snapped,
        base_segs_gdf=base,
        coast_lines_gs=coast_gs,
        ray_hits_gdf=rays,
        # chaining
        hop_max_km=args.hop_max_km,
        total_max_km=args.total_max_km,
        k_neighbors=args.k_neighbors,
        passes=args.passes,
        # legacy + opts
        delta_max_km=args.delta_max_km,
        pad_km=args.pad_km,
        allow_cross_parent=args.allow_cross_parent,
        use_rays_seeds=args.use_rays_seeds,
        use_lowht_seed=args.use_lowht_seed,
        ht_lo_m=args.ht_lo_m,
        ht_hi_m=args.ht_hi_m,
        graph_snap_m=args.graph_snap_m,
        seed_height_col="runupHt",
        seed_height_min_m=args.seed_height_min_m,
        seed_L_min_km=args.seed_L_min_km,
        seed_L_max_km=args.seed_L_max_km,
        seed_km_per_meter=args.seed_km_per_meter,
        seed_height_exp=args.seed_height_exp,
        seed_length_mult=args.seed_length_mult,
        seed_merge_tol_m=args.seed_merge_tol_m,
        log_decisions=args.log_decisions,
        log_rejects_limit=args.log_rejects_limit,
        log_timing=args.debug,
    )
    if out is not None and not out.empty:
        out.to_file(args.gpkg, layer=args.out_layer, driver="GPKG")
        print(f"wrote {len(out)} unified bridges to layer '{args.out_layer}'")
        wrote_any = True

    if not wrote_any:
        print("no infill produced")
        sys.exit(0)
