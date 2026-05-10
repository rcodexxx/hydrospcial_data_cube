"""
SSS backscatter mosaic pipeline.

Builds two mosaics per frequency (cube res + hires for viewer), each
with backscatter, cluster labels, and angular-Gaussian confidence.
"""
import argparse
import time

from src.config import get_config, ROOT
from src.sss.mosaic import run_mosaic
from src.sss.pipeline import run_pipeline


def _output_paths(sss_cfg, freq, suffix=""):
    """Resolve mosaic / cluster / confidence GeoTIFF paths."""
    output_dir = ROOT / sss_cfg["output_dir"]
    f = freq.lower()
    bs_tif   = output_dir / f"sss_backscatter_{f}{suffix}.tif"
    lbl_tif  = output_dir / f"sss_clusters_{f}{suffix}.tif"
    conf_tif = output_dir / f"sss_confidence_{f}{suffix}.tif"
    return bs_tif, lbl_tif, conf_tif


def _build_one(freq):
    """Run pipeline once, mosaic twice (cube res + hires)."""
    cfg = get_config()
    sss_cfg = cfg["sss"]
    mbes_tif = ROOT / cfg["mbes"]["bathymetry_tif"]

    print(f"\n{'=' * 70}")
    print(f"Building {freq} mosaics")
    print(f"{'=' * 70}")

    t0 = time.time()
    samples = run_pipeline(freq)
    if samples is None:
        print(f"  No valid samples for {freq}.")
        return

    correction_result = {
        "bs_db": samples["bs_db"],
        "sample_labels": samples["sample_labels"],
    }
    pooled_for_mosaic = {
        "lat": samples["lat"], "lon": samples["lon"],
        "inc_angle": samples["inc_angle"],
    }

    # Cube-resolution mosaic
    bs_tif, lbl_tif, conf_tif = _output_paths(sss_cfg, freq)
    print(f"\n  Mosaicking → {bs_tif.name} (cube resolution)")
    run_mosaic(pooled_for_mosaic, correction_result,
               bs_tif, lbl_tif, conf_tif, mbes_tif)

    # Hires mosaic
    hires_res = sss_cfg[f"hires_resolution_{freq.lower()}"]
    bs_tif_h, lbl_tif_h, conf_tif_h = _output_paths(sss_cfg, freq, suffix="_hires")
    print(f"\n  Mosaicking → {bs_tif_h.name} (hires)")
    run_mosaic(pooled_for_mosaic, correction_result,
               bs_tif_h, lbl_tif_h, conf_tif_h, mbes_tif,
               resolution=hires_res)

    print(f"\n  {freq} done in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=False)
    parser.add_argument("--freq", default="all", choices=["all", "hf", "lf"])
    args = parser.parse_args()

    t_total = time.time()
    if args.freq in ("all", "hf"):
        _build_one("HF")
    if args.freq in ("all", "lf"):
        _build_one("LF")

    print(f"\nTotal: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()