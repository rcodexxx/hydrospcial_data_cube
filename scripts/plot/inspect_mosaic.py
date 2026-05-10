"""
Inspect SSS hires mosaic at 1:1 pixel resolution.

Outputs raw PNG with each GeoTIFF pixel = 1 PNG pixel. No axes,
no colorbar, no padding. For visual inspection of detail and seams.

Usage:
    python scripts/plot/inspect_mosaic.py             # both HF and LF
    python scripts/plot/inspect_mosaic.py --freq hf
"""
import argparse

import numpy as np
import rasterio
from matplotlib import cm
from PIL import Image

from src.config import ROOT, get_config


def render_raw(tif_path, out_path, cmap_name="copper", clip_percentile=(2, 98)):
    """Render a GeoTIFF as a 1:1 pixel PNG with the given colormap."""
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan

    valid = np.isfinite(data)
    if not valid.any():
        print(f"  skip: no valid data in {tif_path.name}")
        return

    vmin, vmax = np.percentile(data[valid], clip_percentile)
    norm = np.zeros_like(data, dtype=np.float32)
    norm[valid] = np.clip((data[valid] - vmin) / max(vmax - vmin, 1e-6), 0, 1)

    cmap = cm.get_cmap(cmap_name)
    rgba = (cmap(norm) * 255).astype(np.uint8)
    rgba[~valid] = [0, 0, 0, 0]   # transparent for nodata

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(out_path, optimize=False)

    h, w = data.shape
    size_mb = out_path.stat().st_size / 1e6
    print(f"  saved: {out_path.relative_to(ROOT)}  "
          f"({w}x{h}, {size_mb:.1f} MB, vmin={vmin:.1f}, vmax={vmax:.1f})")


def render_cluster(tif_path, out_path):
    """Render cluster label map. Each cluster gets a distinct color from tab10."""
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.int16)
        nd = 255 if src.nodata is None else int(src.nodata)

    valid = data != nd
    if not valid.any():
        print(f"  skip: no valid data in {tif_path.name}")
        return

    n_clusters = int(data[valid].max()) + 1
    cmap = cm.get_cmap("tab10", n_clusters)

    norm = np.zeros_like(data, dtype=np.float32)
    norm[valid] = data[valid] / max(n_clusters - 1, 1)
    rgba = (cmap(norm) * 255).astype(np.uint8)
    rgba[~valid] = [0, 0, 0, 0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(out_path, optimize=False)

    h, w = data.shape
    size_mb = out_path.stat().st_size / 1e6
    print(f"  saved: {out_path.relative_to(ROOT)}  "
          f"({w}x{h}, {size_mb:.1f} MB, K={n_clusters})")


def _inspect_one(sss_cfg, freq):
    output_dir = ROOT / sss_cfg["output_dir"]
    f = freq.lower()

    bs_tif  = output_dir / f"sss_backscatter_{f}_hires.tif"
    lbl_tif = output_dir / f"sss_clusters_{f}_hires.tif"

    out_dir = ROOT / "outputs/inspect"
    bs_png  = out_dir / f"sss_backscatter_{f}_hires.png"
    lbl_png = out_dir / f"sss_clusters_{f}_hires.png"

    print(f"\n[{freq}]")
    if bs_tif.exists():
        render_raw(bs_tif, bs_png, cmap_name="copper")
    else:
        print(f"  missing: {bs_tif.name}")

    if lbl_tif.exists():
        render_cluster(lbl_tif, lbl_png)
    else:
        print(f"  missing: {lbl_tif.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=False)
    parser.add_argument("--freq", default="all", choices=["all", "hf", "lf"])
    args = parser.parse_args()

    cfg = get_config()
    sss_cfg = cfg["sss"]

    if args.freq in ("all", "hf"):
        _inspect_one(sss_cfg, "HF")
    if args.freq in ("all", "lf"):
        _inspect_one(sss_cfg, "LF")


if __name__ == "__main__":
    main()