from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
from matplotlib.colors import LightSource, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import Transformer

OUT = Path(r"/outputs")
MBES = Path(r"/outputs/tif/G1m_142m.tif")

with rasterio.open(MBES) as src:
    dem = src.read(1)
    bounds = src.bounds
    crs = src.crs
    res = src.res[0]

extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
mbes_mask = ~np.isnan(dem)
tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


def to_dms(decimal_deg, is_lon):
    d = int(abs(decimal_deg))
    m = int((abs(decimal_deg) - d) * 60)
    s = ((abs(decimal_deg) - d) * 60 - m) * 60
    hemi = (
        ("E" if decimal_deg >= 0 else "W")
        if is_lon
        else ("N" if decimal_deg >= 0 else "S")
    )
    return f"{d}°{m}′{s:04.1f}″{hemi}"


def make_tick_formatter(is_lon, ref, transformer):
    def fmt(val, pos):
        if is_lon:
            lon, _ = transformer.transform(val, ref)
            return to_dms(lon, is_lon=True)
        else:
            _, lat = transformer.transform(ref, val)
            return to_dms(lat, is_lon=False)

    return fmt


def load_masked(path, mask):
    with rasterio.open(path) as src:
        data = src.read(1)
    return np.where(mask, data, np.nan)


def compute_aspect(dem, res):
    z = np.pad(dem, 1, constant_values=np.nan)
    dz_dx = (z[1:-1, 2:] - z[1:-1, :-2]) / (2 * res)
    dz_dy = (z[2:, 1:-1] - z[:-2, 1:-1]) / (2 * res)
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx)) % 360
    aspect[np.isnan(dem)] = np.nan
    return aspect.astype(np.float32)


slope = load_masked(OUT / "tif/mbes_slope.tif", mbes_mask)
rugosity = load_masked(OUT / "tif/mbes_rugosity.tif", mbes_mask)
aspect = compute_aspect(dem, res)
aspect = np.where(mbes_mask, aspect, np.nan)

# hillshade
ls = LightSource(azdeg=315, altdeg=45)
dem_f = np.where(mbes_mask, dem, np.nanmean(dem))
shade = ls.hillshade(dem_f, vert_exag=3, dx=res, dy=res)
shade = np.where(mbes_mask, shade, np.nan)

# aspect + hillshade RGBA
norm = Normalize(vmin=0, vmax=360)
aspect_rgb = plt.get_cmap("twilight")(norm(np.nan_to_num(aspect)))
aspect_rgb[..., :3] *= np.nan_to_num(shade)[..., np.newaxis]
aspect_rgb[~mbes_mask] = [1, 1, 1, 0]  # 遮罩外透明

x_ticks = np.linspace(bounds.left, bounds.right, 4)
y_ticks = np.linspace(bounds.bottom, bounds.top, 4)
mid_x = (bounds.left + bounds.right) / 2
mid_y = (bounds.bottom + bounds.top) / 2

datasets = [
    (dem, "turbo_r", "Bathymetry (m)", True, None),
    (slope, "YlOrRd", "Slope (deg)", True, None),
    (rugosity, "cividis", "Rugosity", True, None),
    (None, "twilight", "Aspect (deg)", False, aspect_rgb),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for ax, (data, cmap, title, show_cbar, rgba) in zip(axes, datasets):
    if rgba is not None:
        im = ax.imshow(rgba, origin="upper", aspect="equal", extent=extent)
        # 單獨畫 colorbar 用的 scalar mappable
        sm = plt.cm.ScalarMappable(cmap="twilight", norm=norm)
        sm.set_array([])
    else:
        valid = data[mbes_mask]
        vmin, vmax = np.nanpercentile(valid, [2, 98])
        im = ax.imshow(
            data,
            cmap=cmap,
            origin="upper",
            aspect="equal",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        sm = im

    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.05)
    plt.colorbar(sm, cax=cax)

    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(
            make_tick_formatter(is_lon=True, ref=mid_y, transformer=tr)
        )
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            make_tick_formatter(is_lon=False, ref=mid_x, transformer=tr)
        )
    )
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, color="white", linewidth=0.4, alpha=0.5, linestyle="--")

plt.suptitle("Mudan Reservoir — Terrain Analysis", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "figures/terrain_analysis.png", dpi=200, bbox_inches="tight")
print("saved")
