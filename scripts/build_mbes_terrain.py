"""
Compute MBES-derived terrain layers from bathymetric DEM.

Outputs:
  - VRM:   Vector Ruggedness Measure (Sappington et al. 2007),
           bounded in [0, 1], measures local surface roughness via
           unit normal vector dispersion within a moving window.
  - Slope: Local terrain slope in degrees, computed from gradient
           of the DEM.
  - BPI:   Bathymetric Position Index (Lundblad et al. 2006),
           the difference between focal cell depth and the mean
           depth of an annular neighborhood. Positive = ridge / mound,
           negative = valley / pit.
"""
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter
from scipy.ndimage import binary_erosion

from src.config import get_config, ROOT


# Algorithm constants
VRM_WINDOW = 5            # 5x5 cells for VRM neighborhood
BPI_INNER_RADIUS = 1      # cells (skipped, focal exclusion)
BPI_OUTER_RADIUS = 10     # cells (~5 m at 0.5 m grid)


def compute_vrm(dem, res, window=VRM_WINDOW):
    """VRM via unit normal dispersion (Sappington et al. 2007)."""
    z = np.where(np.isnan(dem), 0.0, dem)
    dz_dx = np.gradient(z, res, axis=1)
    dz_dy = np.gradient(z, res, axis=0)
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    aspect_rad = np.arctan2(dz_dy, dz_dx)
    nx = np.sin(slope_rad) * np.cos(aspect_rad)
    ny = np.sin(slope_rad) * np.sin(aspect_rad)
    nz = np.cos(slope_rad)
    n = window * window
    rx = uniform_filter(nx, size=window) * n
    ry = uniform_filter(ny, size=window) * n
    rz = uniform_filter(nz, size=window) * n
    vrm = (1.0 - np.sqrt(rx ** 2 + ry ** 2 + rz ** 2) / n).astype(np.float32)
    vrm[np.isnan(dem)] = np.nan
    return vrm


def compute_slope(dem, res):
    """Slope in degrees from DEM gradient."""
    z = np.where(np.isnan(dem), 0.0, dem)
    dz_dx = np.gradient(z, res, axis=1)
    dz_dy = np.gradient(z, res, axis=0)
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    slope_deg[np.isnan(dem)] = np.nan
    return slope_deg


def compute_bpi(dem, inner=BPI_INNER_RADIUS, outer=BPI_OUTER_RADIUS):
    """
    Bathymetric Position Index: focal cell depth minus mean depth of
    annular neighborhood [inner, outer] (Lundblad et al. 2006).

    Implemented as difference of two box-mean filters; ignores NaN by
    weighted average.
    """
    mask = np.isfinite(dem).astype(np.float32)
    filled = np.where(mask > 0, dem, 0.0)

    # Outer window mean
    outer_size = 2 * outer + 1
    sum_outer = uniform_filter(filled, size=outer_size) * (outer_size ** 2)
    cnt_outer = uniform_filter(mask, size=outer_size) * (outer_size ** 2)

    # Inner window mean (subtract to form annulus)
    inner_size = 2 * inner + 1
    sum_inner = uniform_filter(filled, size=inner_size) * (inner_size ** 2)
    cnt_inner = uniform_filter(mask, size=inner_size) * (inner_size ** 2)

    sum_ann = sum_outer - sum_inner
    cnt_ann = cnt_outer - cnt_inner

    annulus_mean = np.where(cnt_ann > 0, sum_ann / np.maximum(cnt_ann, 1e-10), np.nan)
    bpi = (dem - annulus_mean).astype(np.float32)
    bpi[np.isnan(dem)] = np.nan
    return bpi


def write_tif(arr, out_path, profile_template):
    """Write a 1-band float32 GeoTIFF with -9999 nodata."""
    out = np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32)
    profile = profile_template.copy()
    profile.update(dtype="float32", nodata=-9999.0, count=1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)


def report(name, arr):
    valid = arr[np.isfinite(arr)]
    if not valid.size:
        print(f"  {name:<8s}: no valid data")
        return
    print(f"  {name:<8s}: min={valid.min():.4f}  max={valid.max():.4f}  "
          f"median={np.median(valid):.4f}  std={valid.std():.4f}")


def erode_mask(dem, n_cells):
    """Erode valid mask by n_cells to avoid edge artifacts."""
    valid = np.isfinite(dem)
    struct = np.ones((2 * n_cells + 1, 2 * n_cells + 1), dtype=bool)
    return binary_erosion(valid, structure=struct)


def main():
    cfg = get_config()
    mbes_tif = ROOT / cfg["mbes"]["bathymetry_tif"]
    vrm_tif = ROOT / cfg["mbes"]["vrm_tif"]
    slope_tif = ROOT / cfg["mbes"]["slope_tif"]
    bpi_tif = ROOT / cfg["mbes"]["bpi_tif"]

    with rasterio.open(mbes_tif) as src:
        dem = src.read(1).astype(np.float32)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        profile = src.profile
        res = src.res[0]

    print(f"DEM    : {dem.shape}  res={res} m")

    # Erode mask once for all derivative layers
    safe_mask = erode_mask(dem, BPI_OUTER_RADIUS)
    n_eroded = np.isfinite(dem).sum() - safe_mask.sum()
    print(f"Edge erosion: {n_eroded} cells "
          f"({BPI_OUTER_RADIUS}px = {BPI_OUTER_RADIUS*res:.1f} m)")

    print(f"\nComputing VRM ({VRM_WINDOW}x{VRM_WINDOW} cells)...")
    vrm = compute_vrm(dem, res, VRM_WINDOW)
    vrm[~safe_mask] = np.nan
    write_tif(vrm, vrm_tif, profile)

    print("\nComputing Slope...")
    slope = compute_slope(dem, res)
    slope[~safe_mask] = np.nan
    write_tif(slope, slope_tif, profile)

    print(f"\nComputing BPI...")
    bpi = compute_bpi(dem, BPI_INNER_RADIUS, BPI_OUTER_RADIUS)
    bpi[~safe_mask] = np.nan
    write_tif(bpi, bpi_tif, profile)

    print("\nResults:")
    report("VRM", vrm)
    report("Slope", slope)
    report("BPI", bpi)

    print(f"\nSaved:")
    print(f"  {vrm_tif.relative_to(ROOT)}")
    print(f"  {slope_tif.relative_to(ROOT)}")
    print(f"  {bpi_tif.relative_to(ROOT)}")


if __name__ == "__main__":
    main()