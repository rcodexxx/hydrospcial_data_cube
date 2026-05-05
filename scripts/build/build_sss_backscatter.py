"""
SSS backscatter mosaic pipeline.

Usage:
    python build_sss_backscatter.py --config configs/mudan.yaml
    python build_sss_backscatter.py --config configs/mudan.yaml --mode global_arc
    python build_sss_backscatter.py --config configs/mudan.yaml --mode raw

Modes:
    full       — Zhao per-cluster ARC + port/stbd balance (default)
    global_arc — single global ARC, no clustering (ablation)
    raw        — no correction, just angle mask (baseline)
"""
import argparse
import time

import numpy as np

from src.config import get_config, ROOT
from src.sss.mosaic import run_mosaic
from src.sss.pipeline import (
    collect_jsf_files,
    load_mbes,
    georef_all,
    pool_georef_results,
    normalize_gain,
)
from src.sss.correction import run_correction


def _make_output_paths(sss_cfg, mode):
    """
    Append mode suffix to output paths so different modes don't overwrite.
    full → sss_backscatter_hf.tif
    global_arc → sss_backscatter_hf_global_arc.tif
    raw → sss_backscatter_hf_raw.tif
    """
    bs_tif = ROOT / sss_cfg["outputs"]["bs_tif"]
    lbl_tif = ROOT / sss_cfg["outputs"]["lbl_tif"]

    if mode != "full":
        bs_tif = bs_tif.with_stem(bs_tif.stem + f"_{mode}")
        lbl_tif = lbl_tif.with_stem(lbl_tif.stem + f"_{mode}")
    return bs_tif, lbl_tif


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=False,
                        help="Site yaml, e.g. configs/mudan.yaml")
    parser.add_argument("--mode", default="full",
                        choices=["full", "global_arc", "raw"],
                        help="Correction mode (default: full)")
    args = parser.parse_args()

    cfg = get_config()
    sss_cfg = cfg["sss"]
    mbes_tif = ROOT / cfg["mbes"]["bathymetry_tif"]

    bs_tif, lbl_tif = _make_output_paths(sss_cfg, args.mode)

    print(f"Mode:      {args.mode}")
    print(f"Channels:  {sss_cfg['channels']}")
    print(f"Output:    {bs_tif.relative_to(ROOT)}")

    # ── Stage 1: Georef ──────────────────────────────────
    t0 = time.time()
    jsf_files = collect_jsf_files(sss_cfg)
    print(f"\nFound {len(jsf_files)} .jsf files")

    mbes = load_mbes(mbes_tif)
    results = georef_all(jsf_files, sss_cfg["channels"], mbes, mbes_tif)
    if not results:
        print("No valid georef results.")
        return
    print(f"Georef done in {time.time()-t0:.1f}s")

    # ── Stage 2: Pool samples ────────────────────────────
    t1 = time.time()
    pooled = pool_georef_results(results)
    n = len(pooled["bs_linear"])
    print(f"\nPooled {n:,} samples ({time.time()-t1:.1f}s)")
    print(f"altitude: min={pooled['altitude'].min():.2f}, "
          f"p5={np.percentile(pooled['altitude'], 5):.2f}, "
          f"median={np.median(pooled['altitude']):.2f}, "
          f"p95={np.percentile(pooled['altitude'], 95):.2f}, "
          f"max={pooled['altitude'].max():.2f}")

    print(f"bs_linear: min={pooled['bs_linear'].min():.2e}, "
          f"median={np.median(pooled['bs_linear']):.2e}, "
          f"max={pooled['bs_linear'].max():.2e}")

    print(f"inc_angle distribution:")
    for q in [1, 5, 50, 95, 99]:
        print(f"  p{q}: {np.percentile(pooled['inc_angle'], q):.1f}°")

    # ── Stage 2b: AGC compensation via per-(line, channel) normalization
    print("\nGain normalization per (line, channel)...")
    pooled = normalize_gain(pooled)

    # ── Stage 3: Correction ──────────────────────────────
    t2 = time.time()
    correction_result = run_correction(
        pooled, sss_cfg["cluster"], mode=args.mode
    )
    diag = correction_result["diagnostics"]
    print(f"Correction done in {time.time()-t2:.1f}s")
    if args.mode == "full":
        print(f"  h0:          {diag.get('h0', '?'):.2f} m")
        print(f"  n_clusters:  {diag.get('n_clusters', 0)}")
        print(f"  noise ratio: {diag.get('noise_ratio', 0):.2%}")

    # ── Stage 4: Mosaic ──────────────────────────────────
    t3 = time.time()
    print(f"\nMosaicking → {bs_tif.name}")
    run_mosaic(pooled, correction_result, bs_tif, lbl_tif, mbes_tif)
    print(f"Mosaic done in {time.time()-t3:.1f}s")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()