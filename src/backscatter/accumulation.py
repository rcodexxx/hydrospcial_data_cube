import numpy as np
from rasterio.transform import from_origin
from src.config import RESOLUTION

class WeightedAccumulator:
    def __init__(self, bounds):
        self.min_x = bounds.left
        self.max_y = bounds.top
        self.ncols = int(round((bounds.right - bounds.left) / RESOLUTION))
        self.nrows = int(round((bounds.top - bounds.bottom) / RESOLUTION))

        self.sum_weight = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        self.sum_db = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        self.sum_raw = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        
        self.best_score = np.full((self.nrows, self.ncols), np.inf, dtype=np.float32)
        self.label_best = np.full((self.nrows, self.ncols), 255, dtype=np.uint8)

    def add(self, xs, ys, bs_db, bs_raw, inc_angle, labels=None):
        col = ((xs - self.min_x) / RESOLUTION).astype(np.int32)
        row = ((self.max_y - ys) / RESOLUTION).astype(np.int32)

        valid = (
            np.isfinite(bs_db) & 
            (col >= 0) & (col < self.ncols) & 
            (row >= 0) & (row < self.nrows)
        )

        weight = np.clip(1.0 - np.abs(inc_angle - 45.0) / 45.0, 0.05, 1.0)
        far_angle_mask = inc_angle > 65.0
        weight[far_angle_mask] *= 0.3

        valid_r = row[valid]
        valid_c = col[valid]
        valid_w = weight[valid]

        # 高速向量化羽化累加 (C 底層，極快)
        np.add.at(self.sum_weight, (valid_r, valid_c), valid_w)
        np.add.at(self.sum_db, (valid_r, valid_c), bs_db[valid] * valid_w)
        np.add.at(self.sum_raw, (valid_r, valid_c), bs_raw[valid] * valid_w)

        # 高速向量化 Label 更新
        if labels is not None:
            valid_labels = labels[valid]
            score = np.abs(inc_angle[valid] - 45.0)
            
            flat_idx = valid_r * self.ncols + valid_c
            order = np.lexsort((score, flat_idx))
            
            sorted_flat = flat_idx[order]
            sorted_score = score[order]
            sorted_labels = valid_labels[order]
            sorted_r = valid_r[order]
            sorted_c = valid_c[order]
            
            _, unique_idx = np.unique(sorted_flat, return_index=True)
            
            best_r = sorted_r[unique_idx]
            best_c = sorted_c[unique_idx]
            best_score = sorted_score[unique_idx]
            best_labels = sorted_labels[unique_idx]
            
            better = best_score < self.best_score[best_r, best_c]
            r_update = best_r[better]
            c_update = best_c[better]
            
            self.best_score[r_update, c_update] = best_score[better]
            self.label_best[r_update, c_update] = best_labels[better]

    def result(self):
        mask = self.sum_weight > 0
        out_db = np.full((self.nrows, self.ncols), -9999.0, dtype=np.float32)
        out_raw = np.full((self.nrows, self.ncols), -9999.0, dtype=np.float32)

        out_db[mask] = self.sum_db[mask] / self.sum_weight[mask]
        out_raw[mask] = self.sum_raw[mask] / self.sum_weight[mask]

        return out_db, out_raw, self.label_best

    @property
    def transform(self):
        return from_origin(self.min_x, self.max_y, RESOLUTION, RESOLUTION)