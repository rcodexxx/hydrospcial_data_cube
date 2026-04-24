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
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from tqdm import tqdm

from src.config import get_config, ROOT
from src.sss.correction import run_correction
from src.sss.georef import georef_line
from src.sss.mosaic import run_mosaic


def _load_mbes(mbes_tif):
    """Preload MBES once for all georef calls."""
    with rasterio.open(mbes_tif) as src:
        return {
            "data": src.read(1).astype(np.float32),
            "transform": src.transform,
            "tr": Transformer.from_crs(
                "EPSG:4326", f"EPSG:{src.crs.to_epsg()}", always_xy=True
            ),
        }


def _collect_jsf_files(sss_cfg):
    """Resolve all .jsf files with their cable lengths."""
    files = []
    for entry in sss_cfg["survey_dirs"]:
        survey_dir = ROOT / entry["path"]
        cable = entry.get("cable_length")
        for f in sorted(survey_dir.glob("*.jsf")):
            files.append((f, cable))
    return files


def _pool_georef_results(results):
    """
    Concatenate per-channel georef dicts into a single flat sample pool.
    Adds line_id and channel_id arrays.
    """
    arrays = {k: [] for k in (
        "lat", "lon", "bs_linear", "altitude",
        "inc_angle", "ping_idx", "heading",
    )}
    line_ids, channel_ids = [], []

    # Reassign global ping_idx so pings from different lines don't collide
    global_ping_offset = 0

    for line_id, channel_id, r in results:
        n = len(r["bs_linear"])
        for k in arrays:
            arrays[k].append(r[k])
        arrays["ping_idx"][-1] = arrays["ping_idx"][-1] + global_ping_offset
        line_ids.append(np.full(n, line_id, dtype=np.int32))
        channel_ids.append(np.full(n, channel_id, dtype=np.int8))
        global_ping_offset = int(arrays["ping_idx"][-1].max()) + 1

    pooled = {k: np.concatenate(v) for k, v in arrays.items()}
    pooled["line_id"] = np.concatenate(line_ids)
    pooled["channel_id"] = np.concatenate(channel_ids)
    return pooled


def _georef_all(jsf_files, channels, mbes, mbes_tif):
    """
    Run georef on every (jsf, channel) pair. Returns list of
    (line_id, channel_id, georef_dict) tuples.

    line_id = index into jsf_files (stable per survey line)
    channel_id = 0 for port, 1 for stbd
    """
    results = []
    for line_id, (jsf_path, cable) in enumerate(tqdm(jsf_files, desc="Georef")):
        for ch in channels:
            r = georef_line(jsf_path, mbes_tif, ch, cable_length=cable,
                            mbes_preloaded=mbes)
            if r is None:
                continue
            channel_id = 0 if "port" in ch else 1
            results.append((line_id, channel_id, r))
    return results


def _make_output_paths(sss_cfg, mode):
    """
    Append mode suffix to output paths so different modes don't overwrite.
    full → sss_backscatter_hf.tif
    global_arc → sss_backscatter_hf_global_arc.tif
    raw → sss_backscatter_hf_raw.tif
    """
    bs_tif  = ROOT / sss_cfg["outputs"]["bs_tif"]
    lbl_tif = ROOT / sss_cfg["outputs"]["lbl_tif"]

    if mode != "full":
        bs_tif  = bs_tif.with_stem(bs_tif.stem + f"_{mode}")
        lbl_tif = lbl_tif.with_stem(lbl_tif.stem + f"_{mode}")
    return bs_tif, lbl_tif

def _normalize_gain(pooled):
    """
    AGC compensation: normalize bs_linear per (line, channel) by 40th percentile.

    Edgetech's per-ping dynamic AGC produces up to 50x amplitude differences
    across survey lines. This preprocessing places all lines on a comparable
    linear scale before Zhao's radiometric correction.

    Uses 40th percentile rather than median to avoid bias from anomaly-rich lines.
    Modifies pooled["bs_linear"] in place.
    """
    print("\nGain normalization per (line, channel)...")
    refs = []
    n_groups = 0
    for line in np.unique(pooled["line_id"]):
        for ch in (0, 1):
            mask = (pooled["line_id"] == line) & (pooled["channel_id"] == ch)
            if mask.sum() < 100:
                continue
            ref = np.percentile(pooled["bs_linear"][mask], 40)
            if ref > 0:
                pooled["bs_linear"][mask] /= ref
                refs.append(float(ref))
                n_groups += 1
    print(f"  Normalized {n_groups} (line, channel) groups")
    if refs:
        refs = np.array(refs)
        print(f"  Ref range: p5={np.percentile(refs, 5):.3f}, "
              f"median={np.median(refs):.3f}, "
              f"p95={np.percentile(refs, 95):.3f}")
        print(f"  Max/min ratio: {refs.max() / refs.min():.1f}x")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=False,
                        help="Site yaml, e.g. configs/mudan.yaml")
    parser.add_argument("--mode", default="full",
                        choices=["full", "global_arc", "raw"],
                        help="Correction mode (default: full)")
    args = parser.parse_args()

    cfg      = get_config()
    sss_cfg  = cfg["sss"]
    mbes_tif = ROOT / cfg["mbes"]["out_tif"]

    bs_tif, lbl_tif = _make_output_paths(sss_cfg, args.mode)

    print(f"Mode:      {args.mode}")
    print(f"Channels:  {sss_cfg['channels']}")
    print(f"Output:    {bs_tif.relative_to(ROOT)}")

    # ── Stage 1: Georef ──────────────────────────────────
    t0 = time.time()
    jsf_files = _collect_jsf_files(sss_cfg)
    print(f"\nFound {len(jsf_files)} .jsf files")

    mbes = _load_mbes(mbes_tif)
    results = _georef_all(jsf_files, sss_cfg["channels"], mbes, mbes_tif)
    if not results:
        print("No valid georef results.")
        return
    print(f"Georef done in {time.time()-t0:.1f}s")

    # ── Stage 2: Pool samples ────────────────────────────
    t1 = time.time()
    pooled = _pool_georef_results(results)
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
    _normalize_gain(pooled)

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

     # ── Stage 3b: Cross-swath leveling (full mode only) ──
    if args.mode == "full":
        from src.sss.correction import level_swaths
        t3a = time.time()
        correction_result["bs_db"], offsets = level_swaths(   # ← 接 offsets
            correction_result["bs_db"],
            pooled["lon"], pooled["lat"],
            pooled["line_id"], pooled["channel_id"],
            resolution=cfg["grid"]["resolution"],
            epsg=cfg["grid"]["epsg"],
        )
        print(f"Leveling done in {time.time()-t3a:.1f}s")
        
        # Post-leveling BS diagnostic
        valid = np.isfinite(correction_result["bs_db"])
        bs_lev = correction_result["bs_db"][valid]
        print(f"Post-leveling BS: p5={np.percentile(bs_lev, 5):.2f}, "
              f"median={np.median(bs_lev):.2f}, "
              f"p95={np.percentile(bs_lev, 95):.2f} dB")

    # ── Stage 4: Mosaic ──────────────────────────────────
    t3 = time.time()
    print(f"\nMosaicking → {bs_tif.name}")
    run_mosaic(pooled, correction_result, bs_tif, lbl_tif, mbes_tif)
    print(f"Mosaic done in {time.time()-t3:.1f}s")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()