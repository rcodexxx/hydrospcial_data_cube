"""
scripts/build_mbes_tif.py
Convert MBES xyz point cloud to GeoTIFF (EPSG:3826, 1m resolution).
Run from project root: python scripts/build_mbes_tif.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).parent.parent
MB_PATH = PROJECT_ROOT / "data/multibeam/G1m_142m.txt"
TIF_PATH = PROJECT_ROOT / "data/multibeam/G1m_142m.tif"


def read_xyz(path):
    """Read space-delimited xyz txt file, columns: x y z."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["x", "y", "z"])
    print(f"  loaded {len(df):,} points")
    return df


def xyz_to_tif(df, out_path, resolution=0.5):
    df = df.copy()
    df["x"] = df["x"].round(3)
    df["y"] = df["y"].round(3)
    df["z"] = np.abs(df["z"])

    grid_df = df.pivot(index="y", columns="x", values="z")
    print(f"pivot columns: {grid_df.columns.min():.1f} ~ {grid_df.columns.max():.1f}")
    print(f"pivot index:   {grid_df.index.min():.1f} ~ {grid_df.index.max():.1f}")
    print(f"pivot shape:   {grid_df.shape}")
    grid_df = grid_df.sort_index(ascending=False)
    grid = grid_df.values.astype(np.float32)

    min_x = grid_df.columns.min()
    max_y = grid_df.index.max()
    transform = from_origin(min_x, max_y, resolution, resolution)

    # 印出實際範圍確認
    print(f"  x range: {min_x:.1f} ~ {grid_df.columns.max():.1f}")
    print(f"  y range: {grid_df.index.min():.1f} ~ {max_y:.1f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=grid.shape[0],
        width=grid.shape[1],
        count=1,
        dtype=grid.dtype,
        crs="EPSG:3826",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(grid, 1)

    print(f"  grid size : {grid.shape[0]} x {grid.shape[1]} px")
    print(f"  x range   : {min_x:.1f} ~ {grid_df.columns.max():.1f} m")
    print(f"  y range   : {grid_df.index.min():.1f} ~ {max_y:.1f} m")
    print(f"  depth range: {np.nanmin(grid):.2f} ~ {np.nanmax(grid):.2f} m")
    print(f"  saved: {out_path}")

    print(f"unique x count: {df['x'].nunique()}")
    print(f"unique y count: {df['y'].nunique()}")
    print(f"x step sample: {sorted(df['x'].unique())[:10]}")


if __name__ == "__main__":
    print("reading MBES xyz...")
    df = read_xyz(MB_PATH)
    print("converting to GeoTIFF...")
    xyz_to_tif(df, TIF_PATH)
