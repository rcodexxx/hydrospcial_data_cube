# scripts/build/build_mbes_bathymetry.py
"""
Build MBES bathymetry DEM from XYZ point cloud.

Steps:
    1. Read XYZ txt → grid (bin averaging)
    2. Sign correction (positive-down)
    3. Edge erosion (remove boundary artefacts)
    4. Save GeoTIFF + write grid bounds back to mudan.yaml
"""
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.crs import CRS
from rasterio.transform import from_origin
from scipy.ndimage import binary_erosion

from src.config import CFG, ROOT, RESOLUTION, EPSG

CONFIG_PATH = ROOT / "configs/mudan.yaml"
EDGE_SHRINK = 5


def _read_xyz(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["x", "y", "z"])
    print(f"  Loaded {len(df):,} points")
    return df


def _xyz_to_grid(df, resolution):
    df = df.copy()
    df["x"] = df["x"].round(3)
    df["y"] = df["y"].round(3)
    df["z"] = np.abs(df["z"])  # positive-down

    grid_df = df.pivot(index="y", columns="x", values="z").sort_index(ascending=False)
    dem     = grid_df.values.astype(np.float32)
    min_x   = grid_df.columns.min()
    max_y   = grid_df.index.max()

    print(f"  Grid: {dem.shape[0]}x{dem.shape[1]} px, res={resolution}m")
    print(f"  X: {min_x:.1f} ~ {grid_df.columns.max():.1f}")
    print(f"  Y: {grid_df.index.min():.1f} ~ {max_y:.1f}")
    print(f"  Depth: {np.nanmin(dem):.2f} ~ {np.nanmax(dem):.2f} m")

    transform = from_origin(min_x, max_y, resolution, resolution)
    return dem, transform


def _erode_edges(dem, resolution):
    valid    = ~np.isnan(dem)
    struct   = np.ones((EDGE_SHRINK * 2 + 1, EDGE_SHRINK * 2 + 1), dtype=bool)
    interior = binary_erosion(valid, structure=struct)
    removed  = np.sum(valid & ~interior)
    dem[~interior] = np.nan
    print(f"  Edge erosion: removed {removed:,} px ({EDGE_SHRINK}px = {EDGE_SHRINK * resolution}m)")
    return dem


def main():
    mbes       = CFG["instruments"]["mbes"]
    src_xyz    = ROOT / mbes["source"]
    out_tif    = ROOT / mbes["out_tif"]
    resolution = RESOLUTION
    epsg       = EPSG

    out_tif.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {src_xyz.name}")
    df             = _read_xyz(src_xyz)
    dem, transform = _xyz_to_grid(df, resolution)
    dem            = _erode_edges(dem, resolution)

    out = np.where(np.isfinite(dem), dem, -9999.0).astype(np.float32)
    with rasterio.open(out_tif, "w", driver="GTiff",
                       height=out.shape[0], width=out.shape[1],
                       count=1, dtype="float32",
                       crs=CRS.from_epsg(epsg),
                       transform=transform,
                       nodata=-9999.0) as dst:
        dst.write(out, 1)
    print(f"Saved: {out_tif}")

    # write bounds back to config
    with rasterio.open(out_tif) as dst:
        b = dst.bounds

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    cfg["grid"]["bounds"] = {
        "left":   round(b.left,   3),
        "right":  round(b.right,  3),
        "bottom": round(b.bottom, 3),
        "top":    round(b.top,    3),
    }
    CONFIG_PATH.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False))
    print(f"Grid bounds written to {CONFIG_PATH.name}")


if __name__ == "__main__":
    main()