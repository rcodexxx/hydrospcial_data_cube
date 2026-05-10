"""
SSS mosaic: max-winner accumulation + IDW hole fill + GeoTIFF write.

Each grid cell holds the dB value of the single sample whose incidence
angle is closest to REFERENCE_ANGLE_DEG (45°). No averaging across
samples — preserves waterfall-grade detail at the cost of hard seams
in overlap regions.

Outputs:
  - bs_tif         : backscatter dB
  - lbl_tif        : cluster labels
  - conf_tif       : confidence ∈ [0, 1], angular Gaussian weight of
                     the winning sample. Used by downstream ML / cube.
"""
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.coords import BoundingBox
from rasterio.transform import from_origin
from scipy.spatial import cKDTree

from src.config import get_config
from src.sss.config import (
    IDW_SEARCH_RADIUS_M,
    IDW_NEIGHBORS,
    REFERENCE_ANGLE_DEG,
    CONFIDENCE_ANGULAR_SIGMA_DEG,
)


# ════════════════════════════════════════════════════════════════
# Max-winner accumulator
# ════════════════════════════════════════════════════════════════
class MaxWinnerAccumulator:
    """
    Rasterize scattered samples. Each cell keeps the sample whose
    incidence angle is closest to REFERENCE_ANGLE_DEG, plus its
    angular Gaussian weight as confidence.
    """

    def __init__(self, bounds, resolution):
        self.resolution = resolution
        self.min_x = bounds.left
        self.max_y = bounds.top
        self.ncols = int(round((bounds.right - bounds.left) / resolution))
        self.nrows = int(round((bounds.top - bounds.bottom) / resolution))

        self.best_score  = np.full((self.nrows, self.ncols), np.inf, dtype=np.float32)
        self.best_db     = np.full((self.nrows, self.ncols), -9999.0, dtype=np.float32)
        self.best_weight = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        self.label_best  = np.full((self.nrows, self.ncols), -1, dtype=np.int16)

    def add(self, xs, ys, bs_db, inc_angle, labels):
        col = ((xs - self.min_x) / self.resolution).astype(np.int32)
        row = ((self.max_y - ys) / self.resolution).astype(np.int32)

        valid = (np.isfinite(bs_db) &
                 (col >= 0) & (col < self.ncols) &
                 (row >= 0) & (row < self.nrows))

        r, c   = row[valid], col[valid]
        ang    = inc_angle[valid].astype(np.float32)
        score  = np.abs(ang - REFERENCE_ANGLE_DEG)
        weight = np.exp(-((ang - REFERENCE_ANGLE_DEG)
                          / CONFIDENCE_ANGULAR_SIGMA_DEG) ** 2).astype(np.float32)
        db     = bs_db[valid].astype(np.float32)
        lbl    = labels[valid]

        # Within this batch: pick the best score per (r, c).
        flat_idx = r * self.ncols + c
        order = np.lexsort((score, flat_idx))
        sorted_flat   = flat_idx[order]
        sorted_score  = score[order]
        sorted_db     = db[order]
        sorted_weight = weight[order]
        sorted_lbl    = lbl[order]
        sorted_r      = r[order]
        sorted_c      = c[order]

        _, unique_idx = np.unique(sorted_flat, return_index=True)
        best_r = sorted_r[unique_idx]
        best_c = sorted_c[unique_idx]
        best_score_new  = sorted_score[unique_idx]
        best_db_new     = sorted_db[unique_idx]
        best_weight_new = sorted_weight[unique_idx]
        best_lbl_new    = sorted_lbl[unique_idx]

        better = best_score_new < self.best_score[best_r, best_c]
        bi_r = best_r[better]
        bi_c = best_c[better]
        self.best_score[bi_r, bi_c]  = best_score_new[better]
        self.best_db[bi_r, bi_c]     = best_db_new[better]
        self.best_weight[bi_r, bi_c] = best_weight_new[better]
        self.label_best[bi_r, bi_c]  = best_lbl_new[better]

    def result(self):
        return self.best_db, self.label_best, self.best_weight

    @property
    def transform(self):
        return from_origin(self.min_x, self.max_y, self.resolution, self.resolution)


# ════════════════════════════════════════════════════════════════
# Grid resolution
# ════════════════════════════════════════════════════════════════
def _load_grid(mbes_tif, resolution_override=None):
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

    resolution = resolution_override if resolution_override is not None \
                 else grid_cfg["resolution"]
    return bounds, grid_cfg["epsg"], resolution


def _load_dem_mask(mbes_tif, target_shape=None, target_transform=None, target_crs=None):
    """Load DEM nodata mask, optionally reprojected to a target grid."""
    with rasterio.open(mbes_tif) as src:
        if target_shape is None:
            dem = src.read(1).astype(np.float32)
            mask = np.isnan(dem)
            if src.nodata is not None:
                mask |= (dem == src.nodata)
            return mask

        from rasterio.warp import reproject, Resampling
        dem_src = src.read(1).astype(np.float32)
        if src.nodata is not None:
            dem_src[dem_src == src.nodata] = np.nan
        valid_src = np.isfinite(dem_src).astype(np.uint8)

        valid_dst = np.zeros(target_shape, dtype=np.uint8)
        reproject(
            source=valid_src,
            destination=valid_dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
        )
        return valid_dst == 0


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
def run_mosaic(pooled, correction_result, out_bs_tif, out_lbl_tif, out_conf_tif,
               mbes_tif, resolution=None):
    """
    Build a mosaic from corrected samples.

    pooled:             dict with lon, lat, inc_angle (WGS84 degrees)
    correction_result:  dict with bs_db, sample_labels
    resolution:         optional override (m) for grid resolution
    """
    bounds, epsg, res = _load_grid(mbes_tif, resolution_override=resolution)

    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    xs, ys = tr.transform(pooled["lon"], pooled["lat"])

    acc = MaxWinnerAccumulator(bounds, res)
    acc.add(
        xs.astype(np.float32),
        ys.astype(np.float32),
        correction_result["bs_db"],
        pooled["inc_angle"],
        correction_result["sample_labels"],
    )

    out_db, out_lbl, out_conf = acc.result()

    dem_mask = _load_dem_mask(
        mbes_tif,
        target_shape=(acc.nrows, acc.ncols),
        target_transform=acc.transform,
        target_crs=f"EPSG:{epsg}",
    )

    out_db[dem_mask] = -9999.0
    out_lbl[dem_mask] = -1
    out_conf[dem_mask] = 0.0
    out_db, out_lbl = _idw_fill(out_db, out_lbl, dem_mask, res)
    # IDW-filled cells are interpolated, not measured → confidence stays 0
    out_db[dem_mask] = -9999.0
    out_lbl[dem_mask] = -1
    out_conf[dem_mask] = 0.0

    tif_kw = dict(
        driver="GTiff", height=acc.nrows, width=acc.ncols,
        count=1, crs=f"EPSG:{epsg}", transform=acc.transform,
    )
    out_bs_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_bs_tif, "w", dtype="float32",
                       nodata=-9999.0, **tif_kw) as dst:
        dst.write(out_db, 1)

    out_lbl_u8 = np.where(out_lbl < 0, 255, out_lbl).astype(np.uint8)
    with rasterio.open(out_lbl_tif, "w", dtype="uint8",
                       nodata=255, **tif_kw) as dst:
        dst.write(out_lbl_u8, 1)

    with rasterio.open(out_conf_tif, "w", dtype="float32",
                       nodata=-9999.0, **tif_kw) as dst:
        # Mask DEM area as nodata; valid area stays in [0, 1]
        out_conf_write = out_conf.copy()
        out_conf_write[dem_mask] = -9999.0
        dst.write(out_conf_write, 1)

    valid = out_db[out_db != -9999.0]
    print(f"  Saved: {out_bs_tif.name} @ {res} m/px ({acc.nrows}x{acc.ncols})")
    if valid.size:
        print(f"  BS range:    {valid.min():.1f} ~ {valid.max():.1f} dB")
        print(f"  BS median:   {np.median(valid):.1f} dB")
        print(f"  Coverage:    {100 * valid.size / (acc.nrows * acc.ncols):.1f}%")
    conf_valid = out_conf[(out_conf > 0) & ~dem_mask]
    if conf_valid.size:
        print(f"  Conf median: {np.median(conf_valid):.2f}  "
              f"(p5={np.percentile(conf_valid, 5):.2f}, "
              f"p95={np.percentile(conf_valid, 95):.2f})")