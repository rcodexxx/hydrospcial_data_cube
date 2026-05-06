"""
SSS mosaic: weighted accumulation + IDW hole fill + GeoTIFF write.
"""
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.coords import BoundingBox
from rasterio.transform import from_origin
from scipy.spatial import cKDTree

from src.config import get_config
from src.sss.config import (
    MOSAIC_WEIGHT_FLOOR,
    MOSAIC_FAR_ANGLE_PENALTY,
    IDW_SEARCH_RADIUS_M,
    IDW_NEIGHBORS,
    REFERENCE_ANGLE_DEG,
)


# ════════════════════════════════════════════════════════════════
# Weighted accumulator
# ════════════════════════════════════════════════════════════════
class WeightedAccumulator:
    """
    Rasterize scattered samples into a grid with angle-based weighting.
    Samples near the reference angle count more.
    """

    def __init__(self, bounds, resolution):
        self.resolution = resolution
        self.min_x = bounds.left
        self.max_y = bounds.top
        self.ncols = int(round((bounds.right - bounds.left) / resolution))
        self.nrows = int(round((bounds.top - bounds.bottom) / resolution))

        self.sum_weight = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        self.sum_db     = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        self.best_score = np.full((self.nrows, self.ncols), np.inf, dtype=np.float32)
        self.label_best = np.full((self.nrows, self.ncols), -1, dtype=np.int16)

    def add(self, xs, ys, bs_db, inc_angle, labels):
        col = ((xs - self.min_x) / self.resolution).astype(np.int32)
        row = ((self.max_y - ys) / self.resolution).astype(np.int32)

        valid = (np.isfinite(bs_db) &
                 (col >= 0) & (col < self.ncols) &
                 (row >= 0) & (row < self.nrows))

        r, c = row[valid], col[valid]

        # Gaussian feathering centered at empirical inc_angle median (55°).
        # σ=12° ensures edge samples (20°, 70°) receive near-zero weight,
        # enabling genuine smooth blending in overlap regions.
        sigma = 12.0
        reference = 55.0
        angle_deviation = np.abs(inc_angle[valid] - reference)
        w = np.exp(-(angle_deviation / sigma) ** 2).astype(np.float32)

        np.add.at(self.sum_weight, (r, c), w)
        np.add.at(self.sum_db,     (r, c), bs_db[valid] * w)

        # Best-sample label (closest to reference angle wins)
        score = np.abs(inc_angle[valid] - REFERENCE_ANGLE_DEG)
        flat_idx = r * self.ncols + c
        order = np.lexsort((score, flat_idx))

        sorted_flat   = flat_idx[order]
        sorted_score  = score[order]
        sorted_labels = labels[valid][order]
        sorted_r      = r[order]
        sorted_c      = c[order]

        _, unique_idx = np.unique(sorted_flat, return_index=True)
        best_r = sorted_r[unique_idx]
        best_c = sorted_c[unique_idx]
        best_score_new = sorted_score[unique_idx]
        best_labels_new = sorted_labels[unique_idx]

        better = best_score_new < self.best_score[best_r, best_c]
        self.best_score[best_r[better], best_c[better]] = best_score_new[better]
        self.label_best[best_r[better], best_c[better]] = best_labels_new[better]

    def result(self):
        # Require minimum cumulative weight: avoids cells backed only by
        # swath-edge samples with very low gaussian weights.
        mask = self.sum_weight > MOSAIC_WEIGHT_FLOOR
        out_db = np.full((self.nrows, self.ncols), -9999.0, dtype=np.float32)
        out_db[mask] = self.sum_db[mask] / self.sum_weight[mask]
        return out_db, self.label_best

    @property
    def transform(self):
        return from_origin(self.min_x, self.max_y, self.resolution, self.resolution)


# ════════════════════════════════════════════════════════════════
# Grid resolution
# ════════════════════════════════════════════════════════════════
def _load_grid(mbes_tif):
    cfg = get_config()
    grid_cfg = cfg["grid"]
    bounds_cfg = grid_cfg.get("bounds", {}) or {}

    if all(bounds_cfg.get(k) is not None for k in ("left", "right", "bottom", "top")):
        bounds = BoundingBox(
            left=bounds_cfg["left"], right=bounds_cfg["right"],
            bottom=bounds_cfg["bottom"], top=bounds_cfg["top"],
        )
    else:
        with rasterio.open(mbes_tif) as src:
            bounds = src.bounds
        print(f"  Grid bounds auto-derived from MBES. Pin in yaml to lock.")

    return bounds, grid_cfg["epsg"], grid_cfg["resolution"]


def _load_dem_mask(mbes_tif):
    with rasterio.open(mbes_tif) as src:
        dem = src.read(1).astype(np.float32)
        mask = np.isnan(dem)
        if src.nodata is not None:
            mask |= (dem == src.nodata)
    return mask


# ════════════════════════════════════════════════════════════════
# IDW hole fill
# ════════════════════════════════════════════════════════════════
def _idw_fill(out_db, out_lbl, dem_mask, resolution):
    valid_mask = (out_db != -9999.0) & ~dem_mask
    hole_mask  = (out_db == -9999.0) & ~dem_mask

    if not hole_mask.any() or not valid_mask.any():
        return out_db, out_lbl

    radius_px = IDW_SEARCH_RADIUS_M / resolution

    known_pts = np.column_stack(np.where(valid_mask)[::-1])
    query_pts = np.column_stack(np.where(hole_mask)[::-1])
    tree = cKDTree(known_pts)

    # IDW for bs_db
    dists, indices = tree.query(
        query_pts, k=IDW_NEIGHBORS,
        distance_upper_bound=radius_px, workers=-1,
    )
    valid_k = np.isfinite(dists)
    safe_idx = np.minimum(indices, len(known_pts) - 1)
    known_db = out_db[valid_mask]
    weights = np.where(valid_k, 1.0 / (dists ** 2 + 1e-6), 0.0)
    sum_w = weights.sum(axis=1)
    has_nbr = sum_w > 0

    idw_db = np.full(len(query_pts), -9999.0, dtype=np.float32)
    idw_db[has_nbr] = (weights * known_db[safe_idx]).sum(axis=1)[has_nbr] / sum_w[has_nbr]

    hy, hx = np.where(hole_mask)
    out_db[hy, hx] = idw_db

    # Nearest-neighbor for label (labels don't interpolate)
    dists_nn, idx_nn = tree.query(
        query_pts, k=1,
        distance_upper_bound=radius_px * 1.5, workers=-1,
    )
    valid_nn = np.isfinite(dists_nn)
    if valid_nn.any():
        known_lbl = out_lbl[valid_mask]
        out_lbl[hy[valid_nn], hx[valid_nn]] = known_lbl[idx_nn[valid_nn]]

    return out_db, out_lbl


# ════════════════════════════════════════════════════════════════
# Main entry
# ════════════════════════════════════════════════════════════════
def run_mosaic(pooled, correction_result, out_bs_tif, out_lbl_tif, mbes_tif):
    """
    Build a mosaic from corrected samples.

    pooled:             dict with lon, lat (WGS84 degrees)
    correction_result:  dict with bs_db, sample_labels
    """
    bounds, epsg, resolution = _load_grid(mbes_tif)
    dem_mask = _load_dem_mask(mbes_tif)

    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    xs, ys = tr.transform(pooled["lon"], pooled["lat"])

    acc = WeightedAccumulator(bounds, resolution)
    acc.add(
        xs.astype(np.float32),
        ys.astype(np.float32),
        correction_result["bs_db"],
        pooled["inc_angle"],
        correction_result["sample_labels"],
    )

    out_db, out_lbl = acc.result()

    # Mask DEM nodata, then IDW fill remaining holes
    out_db[dem_mask] = -9999.0
    out_lbl[dem_mask] = -1
    out_db, out_lbl = _idw_fill(out_db, out_lbl, dem_mask, resolution)
    out_db[dem_mask] = -9999.0
    out_lbl[dem_mask] = -1

    # Write
    tif_kw = dict(
        driver="GTiff", height=acc.nrows, width=acc.ncols,
        count=1, crs=f"EPSG:{epsg}", transform=acc.transform,
    )
    out_bs_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_bs_tif, "w", dtype="float32",
                       nodata=-9999.0, **tif_kw) as dst:
        dst.write(out_db, 1)

    # uint8 for labels: -1 → 255
    out_lbl_u8 = np.where(out_lbl < 0, 255, out_lbl).astype(np.uint8)
    with rasterio.open(out_lbl_tif, "w", dtype="uint8",
                       nodata=255, **tif_kw) as dst:
        dst.write(out_lbl_u8, 1)

    # Report
    valid = out_db[out_db != -9999.0]
    print(f"  Saved: {out_bs_tif.name}")
    if valid.size:
        print(f"  BS range:  {valid.min():.1f} ~ {valid.max():.1f} dB")
        print(f"  BS median: {np.median(valid):.1f} dB")
        print(f"  Coverage:  {100 * valid.size / (acc.nrows * acc.ncols):.1f}%")