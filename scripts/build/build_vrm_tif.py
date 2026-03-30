"""
scripts/build/build_vrm.py
Compute VRM (Sappington et al. 2007) from MBES bathymetry.
"""
from pathlib import Path
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).parent.parent.parent
MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
OUT_TIF  = ROOT / "outputs/tif/mbes_vrm.tif"
WINDOW   = 5  # 5x5 kernel


def compute_vrm(dem, res, window):
    z = np.where(np.isnan(dem), 0.0, dem)
    dz_dx = np.gradient(z, res, axis=1)
    dz_dy = np.gradient(z, res, axis=0)
    slope_rad  = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect_rad = np.arctan2(dz_dy, dz_dx)
    nx = np.sin(slope_rad) * np.cos(aspect_rad)
    ny = np.sin(slope_rad) * np.sin(aspect_rad)
    nz = np.cos(slope_rad)
    n = window * window
    rx = uniform_filter(nx, size=window) * n
    ry = uniform_filter(ny, size=window) * n
    rz = uniform_filter(nz, size=window) * n
    vrm = (1.0 - np.sqrt(rx**2 + ry**2 + rz**2) / n).astype(np.float32)
    vrm[np.isnan(dem)] = np.nan
    return vrm


def main():
    with rasterio.open(MBES_TIF) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        res = src.res[0]

    print(f"DEM: {dem.shape}, res={res}m")
    print(f"Computing VRM ({WINDOW}x{WINDOW} = {WINDOW*res}m window)...")

    vrm = compute_vrm(dem, res, WINDOW)
    valid = vrm[np.isfinite(vrm)]
    print(f"VRM range: {valid.min():.6f} ~ {valid.max():.6f}")

    out = np.where(np.isfinite(vrm), vrm, -9999.0).astype(np.float32)
    with rasterio.open(OUT_TIF, "w", driver="GTiff",
                       height=out.shape[0], width=out.shape[1],
                       count=1, dtype="float32",
                       crs=crs, transform=transform,
                       nodata=-9999.0) as dst:
        dst.write(out, 1)

    print(f"Saved: {OUT_TIF}")


if __name__ == "__main__":
    main()