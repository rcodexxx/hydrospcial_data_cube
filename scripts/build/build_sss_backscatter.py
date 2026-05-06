"""
SSS backscatter mosaic pipeline.

Usage:
    python build_sss_backscatter.py                # default: both HF and LF
    python build_sss_backscatter.py --freq hf      # HF only
    python build_sss_backscatter.py --freq lf      # LF only
"""
import argparse
import time

from src.config import get_config, ROOT
from src.sss.mosaic import run_mosaic
from src.sss.pipeline import run_pipeline


def _output_paths(sss_cfg, freq):
    """Resolve mosaic / cluster GeoTIFF paths for the given frequency."""
    output_dir = ROOT / sss_cfg["output_dir"]
    f = freq.lower()
    bs_tif = output_dir / f"sss_backscatter_{f}.tif"
    lbl_tif = output_dir / f"sss_clusters_{f}.tif"
    return bs_tif, lbl_tif


def _build_one(freq):
    """Run pipeline + mosaic for a single frequency."""
    cfg = get_config()
    sss_cfg = cfg["sss"]
    mbes_tif = ROOT / cfg["mbes"]["bathymetry_tif"]

    bs_tif, lbl_tif = _output_paths(sss_cfg, freq)

    print(f"\n{'=' * 70}")
    print(f"Building {freq} mosaic → {bs_tif.relative_to(ROOT)}")
    print(f"{'=' * 70}")

    t0 = time.time()
    samples = run_pipeline(freq)
    if samples is None:
        print(f"  No valid samples for {freq}.")
        return

    print(f"\n  Mosaicking → {bs_tif.name}")
    correction_result = {
        "bs_db": samples["bs_db"],
        "sample_labels": samples["sample_labels"],
    }
    pooled_for_mosaic = {
        "lat": samples["lat"], "lon": samples["lon"],
        "inc_angle": samples["inc_angle"],
    }
    run_mosaic(pooled_for_mosaic, correction_result,
               bs_tif, lbl_tif, mbes_tif)

    print(f"  {freq} done in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=False,
                        help="Site yaml, e.g. configs/mudan.yaml")
    parser.add_argument("--freq", default="all",
                        choices=["all", "hf", "lf"],
                        help="Which frequency to build (default: all)")
    args = parser.parse_args()

    t_total = time.time()
    if args.freq in ("all", "hf"):
        _build_one("HF")
    if args.freq in ("all", "lf"):
        _build_one("LF")

    print(f"\nTotal: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()