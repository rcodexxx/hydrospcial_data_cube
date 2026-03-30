# scripts/plot_mag_report.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import binary_dilation
from skimage.feature import canny
from skimage.measure import label, regionprops

MBES_TIF = Path("../../outputs/tif/mbes_bathymetry.tif")
BG_TIF = Path("../../outputs/tif/mag_background.tif")
RES_TIF = Path("../../outputs/tif/mag_residual.tif")


def read_tif(path, nodata=-9999.0):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nd = src.nodata if nodata is None else nodata
        ext = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        res = src.transform.a
    if nd is not None:
        data[np.abs(data - nd) < 1.0] = np.nan
    return data, ext, res


def main():
    bathy, ext, res = read_tif(MBES_TIF, nodata=None)
    bg, _, _ = read_tif(BG_TIF)
    resid, _, _ = read_tif(RES_TIF)

    H, W = bathy.shape
    east = ext[0] + (np.arange(W) + 0.5) * res
    north = ext[2] + (np.arange(H - 1, -1, -1) + 0.5) * res
    E, N = np.meshgrid(east, north)

    bathy_clean = np.where(np.isfinite(bathy), bathy, np.nan)
    levels = np.arange(5, 40, 5)

    # ── Figure 1: Magnetic Intensity Map ────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))

    bg_abs = float(np.nanpercentile(np.abs(bg), 98))
    vmax_bg = np.ceil(bg_abs / 5) * 5

    pm = ax.pcolormesh(
        E,
        N,
        np.ma.masked_invalid(bg),
        cmap="RdBu_r",
        vmin=-vmax_bg,
        vmax=vmax_bg,
        shading="nearest",
    )
    ax.set_facecolor("#cccccc")

    cs = ax.contour(
        E, N, bathy_clean, levels=levels, colors="black", linewidths=0.5, alpha=0.6
    )
    ax.clabel(cs, fmt="%dm", fontsize=7, inline=True)

    plt.colorbar(pm, ax=ax, shrink=0.8, label="Magnetic Background (nT)")
    ax.set_xlabel("Easting (m)", fontsize=10)
    ax.set_ylabel("Northing (m)", fontsize=10)
    ax.set_title(
        f"Mudan Reservoir — Magnetic Intensity Map\n"
        f"(spike-removed, low-pass filtered, ±{vmax_bg:.0f} nT)",
        fontsize=11,
    )
    ax.ticklabel_format(style="plain")
    ax.tick_params(labelsize=8)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(
        "../outputs/figures/report_mag_background.png", dpi=200, bbox_inches="tight"
    )
    plt.show()

    # ── Anomaly region analysis (terminal only) ──────────────────────
    labeled = label(anom_mask)
    props = regionprops(labeled)
    min_area = 10

    print(f"\nResidual std   : {resid_std:.1f} nT")
    print(f"Threshold      : ±{thresh:.1f} nT")
    print(f"\nAnomaly regions (area >= {min_area} px):")
    print(
        f"{'ID':>4} {'Area(px)':>10} {'Area(m²)':>10} "
        f"{'Centroid X':>12} {'Centroid Y':>12} "
        f"{'Mean(nT)':>10} {'Max|nT|':>10}"
    )
    print("-" * 72)

    regions = []
    for prop in sorted(props, key=lambda p: p.area, reverse=True):
        if prop.area < min_area:
            continue
        cy_px = int(prop.centroid[0])
        cx_px = int(prop.centroid[1])
        cx_m = float(east[cx_px])
        cy_m = float(north[cy_px])
        vals = resid[labeled == prop.label]
        mean_v = float(np.nanmean(vals))
        max_v = float(np.nanmax(np.abs(vals)))
        area_m2 = prop.area * res**2
        regions.append((prop.label, prop.area, area_m2, cx_m, cy_m, mean_v, max_v))
        print(
            f"{prop.label:>4} {prop.area:>10} {area_m2:>10.1f} "
            f"{cx_m:>12.1f} {cy_m:>12.1f} "
            f"{mean_v:>10.1f} {max_v:>10.1f}"
        )


if __name__ == "__main__":
    main()
