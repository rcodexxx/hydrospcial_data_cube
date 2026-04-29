"""
Build MBES bathymetry DEM from XYZ point cloud.

Steps:
    1. Read XYZ txt → grid (bin averaging)
    2. Sign correction (positive-down)
    3. Edge erosion (remove boundary artefacts)
    4. Save GeoTIFF
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from scipy.ndimage import binary_erosion

from src.config import get_config, ROOT

EDGE_SHRINK = 5


def _read_xyz(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["x", "y", "z"])
    print(f"  Loaded {len(df):,} points")
    return df


def _xyz_to_grid(df, resolution):
    df = df.copy()
    df["x"] = df["x"].round(3)
    df["y"] = df["y"].round(3)
    df["z"] = np.abs(df["z"])

    grid_df = df.pivot(index="y", columns="x", values="z").sort_index(ascending=False)
    dem = grid_df.values.astype(np.float32)
    min_x = grid_df.columns.min()
    max_y = grid_df.index.max()

    print(f"  Grid : {dem.shape[0]} x {dem.shape[1]} px @ {resolution} m/px")
    transform = from_origin(min_x, max_y, resolution, resolution)
    return dem, transform


def _erode_edges(dem, resolution):
    valid = ~np.isnan(dem)
    struct = np.ones((EDGE_SHRINK * 2 + 1, EDGE_SHRINK * 2 + 1), dtype=bool)
    interior = binary_erosion(valid, structure=struct)
    removed = int(np.sum(valid & ~interior))
    dem[~interior] = np.nan
    print(f"  Edge erosion: removed {removed:,} px "
          f"({EDGE_SHRINK}px = {EDGE_SHRINK * resolution} m)")
    return dem


def _print_summary(dem, transform, resolution):
    valid = dem[np.isfinite(dem)]
    n_valid = valid.size
    n_total = dem.size
    h, w = dem.shape

    left = transform.c
    top = transform.f
    right = left + w * resolution
    bottom = top - h * resolution

    print("\nSummary")
    print(f"  Bounds  : left={left:.1f}  right={right:.1f}  "
          f"bottom={bottom:.1f}  top={top:.1f}")
    print(f"  Coverage: {100 * n_valid / n_total:.1f}% "
          f"({n_valid:,} / {n_total:,} px)")
    if n_valid:
        print(f"  Depth   : min={valid.min():.2f} m  "
              f"max={valid.max():.2f} m  "
              f"median={np.median(valid):.2f} m  "
              f"mean={valid.mean():.2f} m")

    print("\nCopy to configs/<site>.yaml under grid.bounds:")
    print(f"  bounds:")
    print(f"    left:   {round(left, 3)}")
    print(f"    right:  {round(right, 3)}")
    print(f"    bottom: {round(bottom, 3)}")
    print(f"    top:    {round(top, 3)}")


def main():
    cfg = get_config()
    src_xyz = ROOT / cfg["mbes"]["source"]
    out_tif = ROOT / cfg["mbes"]["bathymetry_tif"]
    resolution = cfg["grid"]["resolution"]
    epsg = cfg["grid"]["epsg"]

    out_tif.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {src_xyz.name}")
    df = _read_xyz(src_xyz)
    dem, transform = _xyz_to_grid(df, resolution)
    dem = _erode_edges(dem, resolution)

    out = np.where(np.isfinite(dem), dem, -9999.0).astype(np.float32)
    with rasterio.open(out_tif, "w", driver="GTiff",
                       height=out.shape[0], width=out.shape[1],
                       count=1, dtype="float32",
                       crs=CRS.from_epsg(epsg),
                       transform=transform,
                       nodata=-9999.0) as dst:
        dst.write(out, 1)
    print(f"Saved: {out_tif}")

    _print_summary(dem, transform, resolution)


if __name__ == "__main__":
    main()