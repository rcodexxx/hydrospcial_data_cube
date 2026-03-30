# scripts/build/build_bathymetry.py
"""
Build MBES bathymetry DEM from source TIF.
Applies sign correction (positive-down) and edge erosion.
"""
from pathlib import Path
import numpy as np
import rasterio
from scipy.ndimage import binary_erosion

ROOT     = Path(__file__).parent.parent.parent
SRC_TIF  = ROOT / "data/multibeam/G1m_142m.tif"
OUT_TIF  = ROOT / "outputs/tif/mbes_bathymetry.tif"
EDGE_SHRINK = 5  # erode valid mask by N pixels


def main():
    with rasterio.open(SRC_TIF) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        res = src.res[0]

    print(f"Source: {SRC_TIF.name}")
    print(f"  shape: {dem.shape}, res: {res}m, CRS: {crs}")
    print(f"  Z range (raw): {np.nanmin(dem):.3f} ~ {np.nanmax(dem):.3f} m")

    # sign correction: NOAA positive-down convention
    if np.nanmean(dem) < 0:
        print("  Flipping to positive-down")
        dem = -dem

    # edge erosion: remove unreliable pixels at data boundary
    valid = ~np.isnan(dem)
    struct = np.ones((EDGE_SHRINK * 2 + 1, EDGE_SHRINK * 2 + 1), dtype=bool)
    interior = binary_erosion(valid, structure=struct)
    removed = np.sum(valid & ~interior)
    dem[~interior] = np.nan
    print(f"  Edge erosion: removed {removed:,} pixels ({EDGE_SHRINK}px = {EDGE_SHRINK * res}m)")
    print(f"  Depth range: {np.nanmin(dem):.3f} ~ {np.nanmax(dem):.3f} m")

    out = np.where(np.isfinite(dem), dem, -9999.0).astype(np.float32)
    with rasterio.open(OUT_TIF, "w", driver="GTiff",
                       height=out.shape[0], width=out.shape[1],
                       count=1, dtype="float32",
                       crs=crs, transform=transform,
                       nodata=-9999.0) as dst:
        dst.write(out, 1)

    print(f"Saved: {OUT_TIF}")


if __name__ == "__main__":
    main()