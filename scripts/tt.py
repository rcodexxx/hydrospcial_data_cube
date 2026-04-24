# scripts/diagnose_mbes.py
"""
Check MBES tif coverage and quality for terrain-aware geometry.
"""
import os
os.environ.setdefault("HYDRO_CONFIG", "configs/mudan.yaml")

import numpy as np
import rasterio
from src.config import ROOT, get_config


def main():
    cfg = get_config()
    mbes_tif = ROOT / cfg["mbes"]["out_tif"]

    with rasterio.open(mbes_tif) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        h, w = data.shape

    print(f"File: {mbes_tif.name}")
    print(f"Shape: {h} x {w} ({h*w:,} cells)")
    print(f"Resolution: {transform.a:.3f} m x {-transform.e:.3f} m")
    print(f"CRS: {crs}")
    print(f"Nodata: {nodata}")
    print(f"Extent:")
    print(f"  x: {transform.c:.1f} to {transform.c + w*transform.a:.1f}")
    print(f"  y: {transform.f + h*transform.e:.1f} to {transform.f:.1f}")
    print()

    # Valid mask
    if nodata is not None:
        valid = np.isfinite(data) & (data != nodata)
    else:
        valid = np.isfinite(data)

    n_valid = int(valid.sum())
    n_total = h * w
    print(f"Valid cells: {n_valid:,} / {n_total:,} ({100*n_valid/n_total:.1f}%)")
    print(f"NaN cells:   {n_total - n_valid:,}")
    print()

    if n_valid == 0:
        print("No valid data.")
        return

    v = data[valid]
    print("Depth distribution (valid cells):")
    for q in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
        print(f"  p{q:3d}: {np.percentile(v, q):.2f} m")
    print(f"  mean: {v.mean():.2f} m, std: {v.std():.2f}")
    print()

    # Check for nodata holes inside the valid area
    from scipy.ndimage import binary_erosion, label
    eroded = binary_erosion(valid, iterations=3)
    hole_candidates = valid.astype(np.uint8) - eroded.astype(np.uint8)
    # Count inner holes (nodata cells surrounded by valid)
    nodata_mask = ~valid
    labeled, n_holes = label(nodata_mask)
    hole_sizes = np.bincount(labeled.ravel())[1:]
    if len(hole_sizes) > 0:
        internal_holes = hole_sizes[hole_sizes < 100]  # small holes inside
        print(f"Internal nodata holes (< 100 cells): {len(internal_holes)}")
        print(f"  Total internal hole cells: {internal_holes.sum()}")
    print()

    # Edge nodata fraction
    edge_rows = np.concatenate([data[0, :], data[-1, :]])
    edge_cols = np.concatenate([data[:, 0], data[:, -1]])
    edge_all = np.concatenate([edge_rows, edge_cols])
    if nodata is not None:
        edge_valid = np.isfinite(edge_all) & (edge_all != nodata)
    else:
        edge_valid = np.isfinite(edge_all)
    print(f"Edge pixels valid: {edge_valid.sum():,} / {len(edge_all):,} "
          f"({100*edge_valid.sum()/len(edge_all):.1f}%)")


if __name__ == "__main__":
    main()