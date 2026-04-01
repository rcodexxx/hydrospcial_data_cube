# scripts/plot/plot_backscatter.py
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import Transformer

ROOT = Path(__file__).parent.parent.parent
MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
BS_TIF   = ROOT / "outputs/tif/sss_backscatter_lf.tif"
OUT_FIG  = ROOT / "outputs/figures/backscatter_lf.png"

CUT_NORTH = 2449050  # remove turning artifact


def load_tif(path, nodata=None):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nd   = src.nodata if nodata is None else nodata
        if nd is not None:
            data[data == nd] = np.nan
        return data, src.bounds, src.crs


def setup_ax(ax, bounds, tr):
    mid_x = (bounds.left + bounds.right)  / 2
    mid_y = (bounds.bottom + bounds.top)  / 2

    def fmt_lon(val, _):
        lon, _ = tr.transform(val, mid_y)
        d, m = int(abs(lon)), (abs(lon) % 1) * 60
        return f"{d}°{m:05.2f}′{'E' if lon>=0 else 'W'}"

    def fmt_lat(val, _):
        _, lat = tr.transform(mid_x, val)
        d, m = int(abs(lat)), (abs(lat) % 1) * 60
        return f"{d}°{m:05.2f}′{'N' if lat>=0 else 'S'}"

    ax.set_xticks(np.linspace(bounds.left, bounds.right, 4))
    ax.set_yticks(np.linspace(bounds.bottom, bounds.top, 4))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_lat))
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, color="white", linewidth=0.4, alpha=0.5, linestyle="--")


def main():
    dem, bounds, crs = load_tif(MBES_TIF)
    bs,  _,      _   = load_tif(BS_TIF, nodata=-9999.0)

    # Mask and cut northern artifact
    rows    = dem.shape[0]
    res     = (bounds.top - bounds.bottom) / rows
    cut_row = int((bounds.top - CUT_NORTH) / res)
    mask    = ~np.isnan(dem)
    mask[:cut_row, :] = False

    bs = np.where(mask, bs, np.nan)
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    extent  = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    vmin, vmax = np.nanpercentile(bs[~np.isnan(bs)], [2, 98])

    im  = ax.imshow(bs, cmap="gray_r", origin="upper",
                    aspect="equal", extent=extent,
                    vmin=vmin, vmax=vmax)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.05)
    plt.colorbar(im, cax=cax, label="dB")

    ax.set_title("Mudan Reservoir — SSS LF Backscatter", fontsize=13)
    setup_ax(ax, bounds, tr)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    main()