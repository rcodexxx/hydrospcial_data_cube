"""
Render SSS mosaic GeoTIFFs as publication-quality figures.

Reads bs_tif and lbl_tif produced by build_sss_backscatter.py,
outputs two PNGs (backscatter + cluster label map).

Usage:
    python scripts/plot/plot_backscatter.py --config configs/mudan.yaml
    python scripts/plot/plot_backscatter.py --config configs/mudan.yaml --mode global_arc
"""
import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import Transformer

from src.config import ROOT, get_config


def _load_tif(path, nodata=None):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nd = src.nodata if nodata is None else nodata
        if nd is not None:
            data[data == nd] = np.nan
        return data, src.bounds, src.crs


def _setup_latlon_axes(ax, bounds, tr_to_wgs84):
    """Format axis ticks as decimal-minute lat/lon."""
    mid_x = (bounds.left + bounds.right) / 2
    mid_y = (bounds.bottom + bounds.top) / 2

    def fmt_lon(val, _):
        lon, _ = tr_to_wgs84.transform(val, mid_y)
        d, m = int(abs(lon)), (abs(lon) % 1) * 60
        return f"{d}°{m:05.2f}′{'E' if lon >= 0 else 'W'}"

    def fmt_lat(val, _):
        _, lat = tr_to_wgs84.transform(mid_x, val)
        d, m = int(abs(lat)), (abs(lat) % 1) * 60
        return f"{d}°{m:05.2f}′{'N' if lat >= 0 else 'S'}"

    ax.set_xticks(np.linspace(bounds.left, bounds.right, 4))
    ax.set_yticks(np.linspace(bounds.bottom, bounds.top, 4))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_lat))
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, color="white", linewidth=0.4, alpha=0.5, linestyle="--")


def _render(data, bounds, tr_to_wgs84, out_path, title, plot_type, n_clusters=None):
    """Render one figure. plot_type: 'bs' or 'cluster'."""
    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    valid = ~np.isnan(data)
    if not valid.any():
        print(f"  skip: no valid data for {title}")
        plt.close(fig)
        return

    if plot_type == "cluster":
        cmap = plt.get_cmap("tab10", n_clusters)
        vmin, vmax = -0.5, n_clusters - 0.5
        cbar_label = "Acoustic Facies (Cluster ID)"
        cbar_ticks = range(n_clusters)
    else:
        vmin, vmax = np.percentile(data[valid], [2, 98])
        cmap = "copper"
        cbar_label = "Backscatter Strength (dB)"
        cbar_ticks = None

    im = ax.imshow(
        data, cmap=cmap, origin="upper", aspect="equal", extent=extent,
        vmin=vmin, vmax=vmax, interpolation="none",
    )

    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.05)
    plt.colorbar(im, cax=cax, label=cbar_label, ticks=cbar_ticks)

    # ax.set_title(title, fontsize=14, fontweight="bold")
    _setup_latlon_axes(ax, bounds, tr_to_wgs84)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path.relative_to(ROOT)}")


def _resolve_paths(mode):
    """Resolve tif / output png paths for the given mode."""
    cfg = get_config()
    sss_cfg = cfg["sss"]

    bs_tif  = ROOT / sss_cfg["outputs"]["bs_tif"]
    lbl_tif = ROOT / sss_cfg["outputs"]["lbl_tif"]
    if mode != "full":
        bs_tif  = bs_tif.with_stem(bs_tif.stem + f"_{mode}")
        lbl_tif = lbl_tif.with_stem(lbl_tif.stem + f"_{mode}")

    fig_dir = ROOT / "outputs/figures"
    bs_png  = fig_dir / f"{bs_tif.stem}.png"
    lbl_png = fig_dir / f"{lbl_tif.stem}.png"

    # Derive frequency label (hf/lf) from filename for title text
    freq = "HF" if "hf" in bs_tif.stem.lower() else "LF"
    return bs_tif, lbl_tif, bs_png, lbl_png, freq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False)
    parser.add_argument("--mode", default="full",
                        choices=["full", "global_arc", "raw"])
    args = parser.parse_args()

    bs_tif, lbl_tif, bs_png, lbl_png, freq = _resolve_paths(args.mode)

    if not bs_tif.exists():
        raise FileNotFoundError(
            f"{bs_tif} not found. Run build_sss_backscatter.py --mode {args.mode} first."
        )

    _, _, crs = _load_tif(bs_tif, nodata=-9999.0)
    tr_to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    # Mode suffix for titles
    mode_suffix = {"full": "", "global_arc": " (Global ARC)", "raw": " (Raw)"}[args.mode]

    # Backscatter
    print("Rendering backscatter...")
    bs_data, bounds, _ = _load_tif(bs_tif, nodata=-9999.0)
    _render(
        bs_data, bounds, tr_to_wgs84, bs_png,
        title=f"Calibrated SSS Backscatter ({freq}){mode_suffix}",
        plot_type="bs",
    )

    # Clusters — only meaningful for full mode
    if args.mode == "full" and lbl_tif.exists():
        print("Rendering clusters...")
        lbl_data, _, _ = _load_tif(lbl_tif, nodata=255)
        valid = ~np.isnan(lbl_data)
        if valid.any():
            n_clusters = int(lbl_data[valid].max()) + 1
            _render(
                lbl_data, bounds, tr_to_wgs84, lbl_png,
                title=f"Acoustic Facies ({freq}, K={n_clusters})",
                plot_type="cluster",
                n_clusters=n_clusters,
            )


if __name__ == "__main__":
    main()