# src/backscatter/correct.py
"""
SSS correction pipeline.

Pass 1a: Georef + collect raw BS → build global ARC
Pass 1b: Ping features → fit K-means → predict + smooth labels
Pass 2:  Per-cluster ARC correction
Pass 2b: Port/stbd balancing + per-cluster cross-line normalization
Output:  per-line .npz + .json stats in corrected_dir
"""
import json
import numpy as np
from pathlib import Path
from pyproj import Transformer
from tqdm import tqdm
import rasterio
from itertools import groupby

from src.sss.correction import (
    build_global_arc, collect_features, fit_kmeans,
    predict_labels, apply_arc_correction,
)
from src.sss.georef import georef_line


def _load_mbes(mbes_tif: Path) -> dict:
    with rasterio.open(mbes_tif) as src:
        return {
            "data": src.read(1).astype(np.float32),
            "transform": src.transform,
            "tr": Transformer.from_crs(
                "EPSG:4326", f"EPSG:{src.crs.to_epsg()}", always_xy=True
            ),
        }


def _arc_rms(bs_corr, inc_angle, bins=np.arange(15, 66, 1)) -> float:
    bin_centers = (bins[:-1] + bins[1:]) / 2
    curve = np.full(len(bins) - 1, np.nan)
    for j in range(len(bins) - 1):
        mask = (inc_angle >= bins[j]) & (inc_angle < bins[j + 1]) & np.isfinite(bs_corr)
        if mask.sum() >= 10:
            curve[j] = np.median(bs_corr[mask])
    valid = np.isfinite(curve)
    if valid.sum() < 3:
        return np.nan
    return float(np.sqrt(np.nanmean((curve[valid] - np.nanmean(curve[valid])) ** 2)))


def run_correction(cfg: dict, sss_cfg: dict, root: Path) -> None:
    freq = sss_cfg["freq"]
    channels = sss_cfg["channels"]
    n_clusters = sss_cfg["n_clusters"]
    nadir_cut = float(sss_cfg["arc_correction"]["nadir_cutoff"])
    far_cut = float(sss_cfg["arc_correction"]["far_cutoff"])
    georef_kw = sss_cfg["georef"]
    out_dir = root / sss_cfg["outputs"]["corrected_dir"]
    mbes_tif = root / cfg["instruments"]["mbes"]["out_tif"]
    survey_dirs = cfg["instruments"]["sss"]["survey_dirs"]

    out_dir.mkdir(parents=True, exist_ok=True)

    jsf_files = [
        (root / survey_dir / f, (params or {}).get("cable_length"))
        for survey_dir, params in survey_dirs.items()
        for f in sorted((root / survey_dir).glob("*.jsf"))
    ]
    print(f"JSF files: {len(jsf_files)}  channels: {channels}")

    mbes = _load_mbes(mbes_tif)

    # ── Pass 1a: georef + global ARC ────────────────────────
    print("\nPass 1a: building global ARC...")
    raw_bs_all, raw_inc_all = [], []
    all_results, result_sides, result_keys = [], [], []

    for jsf_path, cable in tqdm(jsf_files, desc="Pass 1a"):
        for ch in channels:
            r = georef_line(
                jsf_path, mbes_tif, ch,
                cable_length=cable,
                mbes_preloaded=mbes,
                **georef_kw,
            )
            if r is None:
                continue
            bs_db = 10 * np.log10(np.maximum(r["bs"], 1e-12))
            raw_bs_all.append(bs_db)
            raw_inc_all.append(r["inc_angle"])
            all_results.append(r)
            result_sides.append("port" if "port" in ch else "stbd")
            result_keys.append((jsf_path.stem, ch))

    if not all_results:
        print("No valid data.")
        return

    bin_centers_arc, arc_curve = build_global_arc(
        np.concatenate(raw_bs_all),
        np.concatenate(raw_inc_all),
    )
    print("Global ARC built.")

    # ── Pass 1b: ping features + K-means ────────────────────
    print(f"\nPass 1b: K-means (K={n_clusters})...")
    sample_feats, feat_cache = [], {}

    for i, r in enumerate(all_results):
        feat, valid_pings = collect_features(
            r["bs"], r["inc_angle"], r["ping_idx"], bin_centers_arc, arc_curve
        )
        if feat is not None:
            sample_feats.append(feat)
            feat_cache[i] = (feat, valid_pings)

    km = fit_kmeans(sample_feats, n_clusters)
    line_labels_map = {
        i: predict_labels(km, feat, vp)
        for i, (feat, vp) in feat_cache.items()
    }
    feat_by_stem_side = {}
    for i, (feat, vp) in feat_cache.items():
        stem, ch = result_keys[i]
        side = "port" if "port" in ch else "stbd"
        feat_by_stem_side[(stem, side)] = feat

    # ── Pass 2: per-cluster ARC correction ──────────────────
    print("\nPass 2: per-cluster ARC correction...")
    corrected = []

    for i, (r, side, (stem, ch)) in enumerate(
            tqdm(zip(all_results, result_sides, result_keys),
                 total=len(all_results), desc="Pass 2")):
        if i not in line_labels_map:
            continue

        valid_pings, ping_labels = line_labels_map[i]
        bs_corr, bs_raw, labels = apply_arc_correction(
            r["bs"], r["inc_angle"], r["ping_idx"],
            valid_pings, ping_labels,
            bin_centers_arc, arc_curve,
            n_clusters=n_clusters, 
            nadir_cutoff=nadir_cut,
            far_cutoff=far_cut,
        )
        corrected.append({
            "stem": stem, "ch": ch, "side": side, "freq": freq,
            "bs_corr": bs_corr, "bs_raw": bs_raw,
            "lon": r["lon"], "lat": r["lat"],
            "inc_angle": r["inc_angle"], "labels": labels,
            "ping_idx": r["ping_idx"],
        })

    # ── Pass 2b: port/stbd balance + per-cluster normalization
    print("\nPass 2b: normalizing...")

    port_meds, stbd_meds = [], []
    for item in corrected:
        valid = np.isfinite(item["bs_corr"])
        if not valid.any():
            continue
        med = float(np.median(item["bs_corr"][valid]))
        (port_meds if item["side"] == "port" else stbd_meds).append(med)

    grand_median = np.nanmedian(port_meds + stbd_meds)
    port_offset  = grand_median - np.nanmedian(port_meds)
    stbd_offset  = grand_median - np.nanmedian(stbd_meds)
    print(f"  Port offset: {port_offset:+.2f} dB  Stbd offset: {stbd_offset:+.2f} dB")

    for item in corrected:
        item["bs_corr"] = item["bs_corr"] + (
            port_offset if item["side"] == "port" else stbd_offset
        )

    # per-cluster grand medians across all lines
    cluster_grand = {}
    for c in range(n_clusters):
        meds = []
        for item in corrected:
            mask = (item["labels"] == c) & np.isfinite(item["bs_corr"])
            if mask.sum() > 50:
                meds.append(float(np.median(item["bs_corr"][mask])))
        if meds:
            cluster_grand[c] = float(np.nanmedian(meds))

    for item in corrected:
        for c, grand_med in cluster_grand.items():
            mask = (item["labels"] == c) & np.isfinite(item["bs_corr"])
            if mask.sum() > 50:
                line_med = float(np.median(item["bs_corr"][mask]))
                item["bs_corr"][mask] += grand_med - line_med

    # ── Save npz + json ──────────────────────────────────────
    print("\nSaving...")
    corrected_sorted = sorted(corrected, key=lambda x: x["stem"])
    stems = [(stem, list(group)) 
         for stem, group in groupby(corrected_sorted, key=lambda x: x["stem"])]

    for stem, group in tqdm(stems, desc="Saving"):
        items = {item["side"]: item for item in group}
        out_stem = out_dir / stem

        npz_data  = {}
        json_data = {"file": stem, "freq": freq}

        for side in ("port", "stbd"):
            if side not in items:
                continue
            item   = items[side]
            valid  = np.isfinite(item["bs_corr"])

            npz_data[f"{side}_bs_corr"] = item["bs_corr"]
            npz_data[f"{side}_bs_raw"]  = item["bs_raw"]
            npz_data[f"{side}_lon"]     = item["lon"]
            npz_data[f"{side}_lat"]     = item["lat"]
            npz_data[f"{side}_inc"]     = item["inc_angle"]
            npz_data[f"{side}_labels"]  = item["labels"]
            npz_data[f"{side}_ping_features"] = feat_by_stem_side.get((stem, side), np.array([]))

            json_data[side] = {
                "n_samples":      int(valid.sum()),
                "bs_median":      round(float(np.median(item["bs_corr"][valid])), 3) if valid.any() else None,
                "bs_p5":          round(float(np.percentile(item["bs_corr"][valid],  5)), 3) if valid.any() else None,
                "bs_p95":         round(float(np.percentile(item["bs_corr"][valid], 95)), 3) if valid.any() else None,
                "arc_rms":        round(_arc_rms(item["bs_corr"], item["inc_angle"]), 4),
                "cluster_counts": [int((item["labels"] == c).sum()) for c in range(n_clusters)],
            }

        np.savez_compressed(out_stem.with_suffix(".npz"), **npz_data)
        out_stem.with_suffix(".json").write_text(json.dumps(json_data, indent=2))

    print(f"Done. {len(stems)} files saved to {out_dir}")