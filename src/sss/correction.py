"""
Zhao (2017) radiometric correction for SSS backscatter.

Pipeline:
  1. Along-track TVG residual: bs *= h_0 / altitude
  2. Convert to dB
  3. Build angular waterfall (ping × angle_bin)
  4. Robust z-score per angle column (median + MAD)
  5. Cluster (pluggable: HDBSCAN default)
  6. Mode filter on label image
  7. Per-cluster ABC: ABC(c, φ) = mean(bs_db | cluster=c, angle=φ)
  8. Apply Zhao Eq.8: bs_corrected = bs_raw - ABC(c, φ) + BS_0(c)
  9. Per-line port/stbd balance
"""
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt2d

from src.sss.config import (
    REFERENCE_ANGLE_DEG,
    NADIR_CUTOFF_DEG, FAR_CUTOFF_DEG,
    ANGLE_MIN_DEG, ANGLE_MAX_DEG, ANGLE_BIN_WIDTH_DEG,
    MIN_SAMPLES_PER_ANGLE_BIN, MIN_SAMPLES_PER_CLUSTER_BIN,
    MODE_FILTER_WINDOW,
)


# ════════════════════════════════════════════════════════════════
# Part A: Along-track TVG residual correction
# ════════════════════════════════════════════════════════════════
def altitude_correct(bs_linear, altitude, h0=None, clip_percentile=99.5,
                     ratio_min=0.5, ratio_max=2.0):
    """
    Zhao Eq.(2): bs *= h_0 / altitude (in linear domain).

    Also clips extreme amplitude values (specular reflections, bottom-track
    errors) at the given percentile to prevent them from dominating
    downstream statistics.
    """
    if h0 is None:
        h0 = float(np.nanmedian(altitude))

    threshold = np.percentile(bs_linear[bs_linear > 0], clip_percentile)
    bs_clipped = np.minimum(bs_linear, threshold)

    ratio = np.clip(h0 / altitude, ratio_min, ratio_max)
    corrected = bs_clipped * ratio.astype(np.float32)
    return corrected.astype(np.float32), h0


# ════════════════════════════════════════════════════════════════
# Part B: Angular waterfall construction
# ════════════════════════════════════════════════════════════════
def build_angular_waterfall(bs_db, inc_angle, ping_idx):
    """
    Reshape per-sample arrays into 2D (n_pings × n_angle_bins) grid.
    Each cell = median of samples in that (ping, angle_bin).

    Returns:
      waterfall:    (n_pings, n_bins) float32, NaN where empty
      angle_centers: (n_bins,) float32
      ping_ids:     unique ping ids (1D), in sorted order
    """
    bins = np.arange(ANGLE_MIN_DEG, ANGLE_MAX_DEG + ANGLE_BIN_WIDTH_DEG,
                     ANGLE_BIN_WIDTH_DEG)
    n_bins = len(bins) - 1
    angle_centers = ((bins[:-1] + bins[1:]) / 2).astype(np.float32)

    valid = (np.isfinite(bs_db) &
             (inc_angle >= ANGLE_MIN_DEG) &
             (inc_angle <  ANGLE_MAX_DEG))

    df = pd.DataFrame({
        "ping": ping_idx[valid],
        "bin":  np.digitize(inc_angle[valid], bins) - 1,
        "bs":   bs_db[valid],
    })
    df = df[(df["bin"] >= 0) & (df["bin"] < n_bins)]

    grouped = df.groupby(["ping", "bin"])["bs"].median().unstack("bin")
    grouped = grouped.reindex(columns=range(n_bins))

    ping_ids = grouped.index.values
    waterfall = grouped.values.astype(np.float32)
    return waterfall, angle_centers, ping_ids


# ════════════════════════════════════════════════════════════════
# Part C: Robust z-score
# ════════════════════════════════════════════════════════════════
def robust_zscore_per_angle(waterfall):
    """
    Robust z-score per angle column: z = (x - median) / (1.4826 * MAD).
    NaN stays NaN. Columns with <10 valid pings stay NaN.
    """
    z = np.full_like(waterfall, np.nan, dtype=np.float32)
    for j in range(waterfall.shape[1]):
        col = waterfall[:, j]
        valid = np.isfinite(col)
        if valid.sum() < 10:
            continue
        med = np.median(col[valid])
        mad = np.median(np.abs(col[valid] - med))
        if mad < 1e-6:
            continue
        z[valid, j] = (col[valid] - med) / (1.4826 * mad)
    return z


# ════════════════════════════════════════════════════════════════
# Part D: Clustering
# ════════════════════════════════════════════════════════════════
def _kmeans_with_merge(features, k_init, bs_range,
                       min_proportion=0.01, merge_distance_ratio=0.05):
    """
    K-means++ with iterative merge (Zhao 2017, Section 3.2.2).

    Drops clusters with proportion < min_proportion.
    Merges clusters with center distance < merge_distance_ratio * bs_range.
    Repeats until k is stable.
    """
    from sklearn.cluster import KMeans

    k = k_init
    labels = None
    merge_threshold = merge_distance_ratio * bs_range

    for _ in range(10):  # max 10 iterations to avoid infinite loop
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=0)
        labels = km.fit_predict(features).astype(np.int16)
        centers = km.cluster_centers_.ravel()

        # Step 1: drop under-proportioned clusters
        unique, counts = np.unique(labels, return_counts=True)
        total = counts.sum()
        keep_mask = (counts / total) >= min_proportion
        keep_ids = unique[keep_mask]

        if len(keep_ids) == 0:
            break

        # Step 2: find cluster pairs to merge (by center distance)
        keep_centers = centers[keep_ids]
        order = np.argsort(keep_centers)
        sorted_ids = keep_ids[order]
        sorted_centers = keep_centers[order]

        merged_ids = [sorted_ids[0]]
        for m in range(1, len(sorted_ids)):
            if sorted_centers[m] - sorted_centers[m - 1] < merge_threshold:
                continue   # merge with previous
            merged_ids.append(sorted_ids[m])

        new_k = len(merged_ids)
        if new_k == k:
            break
        k = new_k

    return labels


def run_clustering(zscore_waterfall, cluster_cfg):
    """Dispatch clustering based on yaml config."""
    valid = np.isfinite(zscore_waterfall)
    features = zscore_waterfall[valid].reshape(-1, 1)

    print(f"  z-score features: n={len(features)}, "
          f"p5={np.percentile(features, 5):.2f}, "
          f"p50={np.median(features):.2f}, "
          f"p95={np.percentile(features, 95):.2f}, "
          f"std={features.std():.2f}")

    method = cluster_cfg["method"]
    if method == "kmeans":
        bs_range = float(features.max() - features.min())
        labels_flat = _kmeans_with_merge(
            features,
            k_init=cluster_cfg.get("k_init", 7),
            bs_range=bs_range,
        )
    elif method == "hdbscan":
        import hdbscan
        n_valid = features.shape[0]
        min_frac = cluster_cfg["min_cluster_fraction"]
        min_cluster_size = max(int(n_valid * min_frac), 10)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=cluster_cfg["min_samples"],
            core_dist_n_jobs=-1,
        )
        labels_flat = clusterer.fit_predict(features).astype(np.int16)
    else:
        raise ValueError(f"Unknown cluster method: {method}")

    labels = np.full(zscore_waterfall.shape, -1, dtype=np.int16)
    labels[valid] = labels_flat
    return labels


# ════════════════════════════════════════════════════════════════
# Part E: Label post-processing
# ════════════════════════════════════════════════════════════════
def smooth_labels(labels):
    """Mode filter on 2D label image. NaN (-1) stays -1."""
    if MODE_FILTER_WINDOW <= 1:
        return labels

    # medfilt2d on int doesn't respect NaN; approximate mode via median
    # (works because clusters are few and contiguous)
    smoothed = medfilt2d(labels.astype(np.int16), kernel_size=MODE_FILTER_WINDOW)
    smoothed[labels == -1] = -1   # preserve NaN pixels
    return smoothed


# ════════════════════════════════════════════════════════════════
# Part F: ABC computation and application
# ════════════════════════════════════════════════════════════════
def compute_global_abc(waterfall):
    """
    Single global ABC across all pixels (no sediment consideration).
    Used for mode='global_arc' ablation.
    Returns: abc (n_bins,), bs_0 (scalar)
    """
    n_bins = waterfall.shape[1]
    abc = np.full(n_bins, np.nan, dtype=np.float32)
    for j in range(n_bins):
        col = waterfall[:, j]
        valid = np.isfinite(col)
        if valid.sum() >= MIN_SAMPLES_PER_ANGLE_BIN:
            abc[j] = np.median(col[valid])

    valid_bins = np.isfinite(abc)
    if valid_bins.sum() > 3:
        abc[valid_bins] = uniform_filter1d(abc[valid_bins], size=5)

    bs_0 = float(np.nanmean(abc))
    return abc, bs_0


def compute_per_cluster_abc(waterfall, labels):
    """
    Per-cluster ABC: ABC(c, φ) = mean(bs_db | cluster=c, angle=φ).
    Returns dict {cluster_id: (abc_curve, bs_0)}.
    Ignores cluster -1 (HDBSCAN noise).
    """
    cluster_ids = np.unique(labels)
    cluster_ids = cluster_ids[cluster_ids >= 0]

    abc_by_cluster = {}
    n_bins = waterfall.shape[1]

    for c in cluster_ids:
        curve = np.full(n_bins, np.nan, dtype=np.float32)
        for j in range(n_bins):
            mask = (labels[:, j] == c) & np.isfinite(waterfall[:, j])
            if mask.sum() >= MIN_SAMPLES_PER_CLUSTER_BIN:
                curve[j] = np.mean(waterfall[mask, j])

        valid = np.isfinite(curve)
        if valid.sum() < 3:
            continue
        curve[valid] = uniform_filter1d(curve[valid], size=5)
        bs_0 = float(np.nanmean(curve))
        abc_by_cluster[int(c)] = (curve, bs_0)

    return abc_by_cluster


def apply_global_abc(bs_db, inc_angle, abc, bs_0, angle_centers):
    """Apply single global ABC. Returns corrected bs_db (per-sample)."""
    valid = np.isfinite(abc)
    if valid.sum() < 2:
        return bs_db.copy()
    correction = np.interp(inc_angle, angle_centers[valid], abc[valid])
    return (bs_db - correction + bs_0).astype(np.float32)


def apply_per_cluster_abc(bs_db, inc_angle, ping_idx, sample_labels,
                          abc_by_cluster, angle_centers):
    """
    Apply Zhao Eq.8 per sample using its cluster's ABC.
    Returns corrected bs_db (per-sample).
    """
    bs_corr = bs_db.copy()

    for c, (abc, bs_0) in abc_by_cluster.items():
        mask = sample_labels == c
        if not mask.any():
            continue
        valid = np.isfinite(abc)
        if valid.sum() < 2:
            continue
        correction = np.interp(inc_angle[mask], angle_centers[valid], abc[valid])
        bs_corr[mask] = bs_db[mask] - correction + bs_0

    # samples with cluster -1 (noise) or unassigned: leave untouched
    return bs_corr.astype(np.float32)


# ════════════════════════════════════════════════════════════════
# Part G: Port/stbd balancing (per line)
# ════════════════════════════════════════════════════════════════
def balance_port_stbd_per_line(bs_db, line_id, channel_id):
    """
    For each line, shift port and stbd so their medians meet in the middle.
    channel_id: 0=port, 1=stbd.
    Modifies bs_db in place and returns it.
    """
    for line in np.unique(line_id):
        mask_line = line_id == line
        mp = mask_line & (channel_id == 0) & np.isfinite(bs_db)
        ms = mask_line & (channel_id == 1) & np.isfinite(bs_db)
        if mp.sum() < 100 or ms.sum() < 100:
            continue
        mp_med = np.median(bs_db[mp])
        ms_med = np.median(bs_db[ms])
        shift = (mp_med - ms_med) / 2
        bs_db[mp] -= shift
        bs_db[ms] += shift
    return bs_db


# ════════════════════════════════════════════════════════════════
# Part H: Cross-swath L1 leveling
# ════════════════════════════════════════════════════════════════
def _compute_pairwise_constraints(lon, lat, bs_db, line_id, channel_id,
                                  resolution, epsg, min_overlap=50):
    """
    For each pair of swaths (line_id, channel_id), compute median BS diff
    over their shared grid cells.
    Extends Haar et al. (2023) bulk shift to the per-swath level.
    """
    from pyproj import Transformer
    from tqdm import tqdm

    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    xs, ys = tr.transform(lon, lat)
    col = np.round(xs / resolution).astype(np.int64)
    row = np.round(ys / resolution).astype(np.int64)

    valid = np.isfinite(bs_db)
    swath = line_id.astype(np.int64) * 10 + channel_id

    df = pd.DataFrame({
        "swath": swath[valid], "row": row[valid],
        "col": col[valid], "bs": bs_db[valid],
    })
    cell_median = df.groupby(["swath", "row", "col"])["bs"].median()

    unique_swaths = np.unique(df["swath"].values)
    swath_cells = {}
    for sw in unique_swaths:
        sub = cell_median.loc[sw]
        cells = np.array(list(sub.index))
        key = cells[:, 0] * 100000 + cells[:, 1]
        order = np.argsort(key)
        swath_cells[sw] = (key[order], sub.values[order])

    constraints = []
    sw_list = list(unique_swaths)
    for i in tqdm(range(len(sw_list)), desc="  Constraints"):
        sw_a = sw_list[i]
        keys_a, meds_a = swath_cells[sw_a]
        for j in range(i + 1, len(sw_list)):
            sw_b = sw_list[j]
            keys_b, meds_b = swath_cells[sw_b]

            idx = np.searchsorted(keys_a, keys_b)
            idx_safe = np.minimum(idx, len(keys_a) - 1)
            match = keys_a[idx_safe] == keys_b
            if match.sum() < min_overlap:
                continue

            diff = meds_b[match] - meds_a[idx_safe[match]]
            constraints.append((sw_a, sw_b, float(np.median(diff))))

    return constraints, unique_swaths


def _solve_l1(constraints, swath_ids, anchor_swath):
    """
    L1 minimization: min sum |o_a - o_b + d_ab|  s.t. o[anchor] = 0.
    Reformulated as LP using auxiliary slack variables t_k.
    """
    from scipy.optimize import linprog

    sw_idx = {sw: i for i, sw in enumerate(swath_ids)}
    n_sw = len(swath_ids)
    n_con = len(constraints)
    anchor_idx = sw_idx[anchor_swath]

    # Variables: [o_0..o_{n_sw-1}, t_0..t_{n_con-1}]
    c = np.zeros(n_sw + n_con)
    c[n_sw:] = 1.0

    A_ub_rows, b_ub = [], []
    for k, (sw_a, sw_b, d_ab) in enumerate(constraints):
        ia, ib = sw_idx[sw_a], sw_idx[sw_b]
        row1 = np.zeros(n_sw + n_con)
        row1[ia], row1[ib], row1[n_sw + k] = 1.0, -1.0, -1.0
        A_ub_rows.append(row1)
        b_ub.append(-d_ab)

        row2 = np.zeros(n_sw + n_con)
        row2[ia], row2[ib], row2[n_sw + k] = -1.0, 1.0, -1.0
        A_ub_rows.append(row2)
        b_ub.append(d_ab)

    A_ub = np.vstack(A_ub_rows)
    b_ub = np.array(b_ub)

    A_eq = np.zeros((1, n_sw + n_con))
    A_eq[0, anchor_idx] = 1.0
    b_eq = np.array([0.0])

    bounds = [(None, None)] * n_sw + [(0, None)] * n_con

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    offsets = {sw: float(res.x[sw_idx[sw]]) for sw in swath_ids}
    residuals = res.x[n_sw:]
    return offsets, residuals


def level_swaths(bs_db, lon, lat, line_id, channel_id, resolution, epsg):
    """
    Cross-swath L1 radiometric leveling.

    Each (line_id, channel_id) defines a swath. Pairwise overlap cells
    produce median BS diff constraints. Solve for per-swath offsets that
    minimize sum of absolute residuals.

    Returns corrected bs_db and the offsets dict.
    """
    print("\nCross-swath L1 leveling...")
    constraints, unique_swaths = _compute_pairwise_constraints(
        lon, lat, bs_db, line_id, channel_id, resolution, epsg,
    )
    print(f"  {len(constraints)} constraints from {len(unique_swaths)} swaths")

    if not constraints:
        print("  No overlapping swaths; skipping leveling")
        return bs_db, {}

    # Anchor = most-connected swath
    degree = {sw: 0 for sw in unique_swaths}
    for sw_a, sw_b, _ in constraints:
        degree[sw_a] += 1
        degree[sw_b] += 1
    anchor = max(degree, key=degree.get)
    print(f"  Anchor swath: {anchor} (degree {degree[anchor]})")

    offsets, residuals = _solve_l1(constraints, unique_swaths, anchor)
    print(f"  L1 residual: mean={np.mean(residuals):.3f}, "
          f"p95={np.percentile(residuals, 95):.3f} dB")

    swath = line_id.astype(np.int64) * 10 + channel_id
    bs_out = bs_db.copy()
    for sw, off in offsets.items():
        mask = swath == sw
        bs_out[mask] -= off

    off_vals = np.array(list(offsets.values()))
    print(f"  Offsets: p5={np.percentile(off_vals, 5):.2f}, "
          f"p50={np.median(off_vals):.2f}, "
          f"p95={np.percentile(off_vals, 95):.2f} dB")

    return bs_out, offsets


# ════════════════════════════════════════════════════════════════
# Part I: Orchestration
# ════════════════════════════════════════════════════════════════
def to_db(bs_linear):
    return 10.0 * np.log10(np.maximum(bs_linear, 1e-12)).astype(np.float32)


def mask_out_of_range_angles(bs_db, inc_angle):
    """Set samples outside [NADIR_CUTOFF, FAR_CUTOFF] to NaN."""
    bad = (inc_angle < NADIR_CUTOFF_DEG) | (inc_angle > FAR_CUTOFF_DEG)
    bs_db = bs_db.copy()
    bs_db[bad] = np.nan
    return bs_db


def run_correction(pooled, cluster_cfg, mode="full"):
    """
    Full correction pipeline on a pooled sample array.

    pooled dict must contain:
      bs_linear, altitude, inc_angle, ping_idx, line_id, channel_id
      (all 1D float/int arrays of equal length)

    mode:
      "raw"        — no correction, just dB + angle mask
      "global_arc" — along-track + single global ABC
      "full"       — along-track + per-cluster ABC + port/stbd balance

    Returns dict with:
      bs_db           — corrected per-sample dB (NaN outside angle range)
      sample_labels   — per-sample cluster label (-1 if none)
      diagnostics     — dict of intermediate results for validation
    """
    bs_lin = pooled["bs_linear"].copy()
    alt    = pooled["altitude"]
    inc    = pooled["inc_angle"]
    pid    = pooled["ping_idx"]
    line   = pooled["line_id"]
    chan   = pooled["channel_id"]

    diag = {}

    # ── Mode: raw ─────────────────────────────────────────
    if mode == "raw":
        bs_db = to_db(bs_lin)
        bs_db = mask_out_of_range_angles(bs_db, inc)
        sample_labels = np.full(len(bs_db), -1, dtype=np.int16)
        return {"bs_db": bs_db, "sample_labels": sample_labels,
                "diagnostics": diag}

    # ── Step 1: along-track TVG residual correction ───────
    bs_lin, h0 = altitude_correct(bs_lin, alt)
    diag["h0"] = h0
    ratio = h0 / pooled["altitude"]
    print(f"correction ratio: min={ratio.min():.2f}, "
        f"median={np.median(ratio):.2f}, "
        f"max={ratio.max():.2f}")

    # ── Step 2: to dB ─────────────────────────────────────
    bs_db = to_db(bs_lin)

    # ── Step 3: build angular waterfall ───────────────────
    waterfall, angle_centers, ping_ids = build_angular_waterfall(bs_db, inc, pid)
    diag["waterfall_shape"] = waterfall.shape
    diag["angle_centers"] = angle_centers
    diag["waterfall"] = waterfall  # kept for plotting

    # ── Mode: global_arc ─────────────────────────────────
    if mode == "global_arc":
        abc, bs_0 = compute_global_abc(waterfall)
        bs_corrected = apply_global_abc(bs_db, inc, abc, bs_0, angle_centers)
        bs_corrected = mask_out_of_range_angles(bs_corrected, inc)
        diag["global_abc"] = abc
        diag["global_bs_0"] = bs_0
        sample_labels = np.full(len(bs_db), -1, dtype=np.int16)
        return {"bs_db": bs_corrected, "sample_labels": sample_labels,
                "diagnostics": diag}

    # ── Step 4: robust z-score ───────────────────────────
    zscore = robust_zscore_per_angle(waterfall)

    # ── Step 5: cluster ──────────────────────────────────
    labels_2d = run_clustering(zscore, cluster_cfg)

    # ── Step 6: mode filter ──────────────────────────────
    labels_2d = smooth_labels(labels_2d)
    diag["labels_2d"]   = labels_2d
    diag["ping_ids"]    = ping_ids
    diag["n_clusters"]  = int(labels_2d[labels_2d >= 0].max() + 1) \
                          if (labels_2d >= 0).any() else 0
    diag["noise_ratio"] = float((labels_2d == -1).sum() / labels_2d.size)

    # Broadcast ping-bin labels back to sample level
    ping_to_row = {p: i for i, p in enumerate(ping_ids)}
    sample_labels = np.full(len(bs_db), -1, dtype=np.int16)
    bin_edges = np.arange(ANGLE_MIN_DEG,
                          ANGLE_MAX_DEG + ANGLE_BIN_WIDTH_DEG,
                          ANGLE_BIN_WIDTH_DEG)
    sample_bin = np.digitize(inc, bin_edges) - 1
    in_range = (sample_bin >= 0) & (sample_bin < labels_2d.shape[1])
    row_idx = np.searchsorted(ping_ids, pid)
    row_idx = np.clip(row_idx, 0, len(ping_ids) - 1)
    row_valid = ping_ids[row_idx] == pid

    bin_edges = np.arange(ANGLE_MIN_DEG,
                          ANGLE_MAX_DEG + ANGLE_BIN_WIDTH_DEG,
                          ANGLE_BIN_WIDTH_DEG)
    sample_bin = np.digitize(inc, bin_edges) - 1
    in_range = (sample_bin >= 0) & (sample_bin < labels_2d.shape[1])

    sample_labels = np.full(len(bs_db), -1, dtype=np.int16)
    final_valid = in_range & row_valid
    sample_labels[final_valid] = labels_2d[
        row_idx[final_valid], sample_bin[final_valid]
    ]

    # ── Step 7-8: per-cluster ABC + apply ────────────────
    abc_by_cluster = compute_per_cluster_abc(waterfall, labels_2d)
    diag["abc_by_cluster"] = abc_by_cluster

    bs_corrected = apply_per_cluster_abc(
        bs_db, inc, pid, sample_labels, abc_by_cluster, angle_centers)

    # ── Step 9: port/stbd balance ────────────────────────
    # bs_corrected = balance_port_stbd_per_line(bs_corrected, line, chan)

    # ── Final: mask out-of-range angles ──────────────────
    bs_corrected = mask_out_of_range_angles(bs_corrected, inc)

    return {"bs_db": bs_corrected, "sample_labels": sample_labels,
            "diagnostics": diag}