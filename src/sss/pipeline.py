"""
SSS sample-level pipeline.

Provides the orchestration from .jsf collection through Zhao
correction, returning sample-level arrays ready for downstream use
(mosaicking, target validation, etc.).

This module separates pipeline logic from CLI scripts so multiple
callers (build_sss_backscatter, validate_mag_targets, future analysis
scripts) can reuse the same processing without duplicating code.
"""
import numpy as np
import rasterio
from pyproj import Transformer
from tqdm import tqdm

from src.config import get_config, ROOT
from src.sss.correction import run_correction
from src.sss.georef import georef_line


def load_mbes(mbes_tif):
    """Preload MBES once for all georef calls in a survey."""
    with rasterio.open(mbes_tif) as src:
        return {
            "data": src.read(1).astype(np.float32),
            "transform": src.transform,
            "tr": Transformer.from_crs(
                "EPSG:4326", f"EPSG:{src.crs.to_epsg()}", always_xy=True
            ),
        }


def collect_jsf_files(sss_cfg):
    """Resolve all .jsf files with their cable lengths from yaml."""
    files = []
    for entry in sss_cfg["survey_dirs"]:
        survey_dir = ROOT / entry["path"]
        cable = entry.get("cable_length")
        for f in sorted(survey_dir.glob("*.jsf")):
            files.append((f, cable))
    return files


def georef_all(jsf_files, channels, mbes, mbes_tif):
    """
    Run georef on every (jsf, channel) pair. Returns list of
    (line_id, channel_id, georef_dict) tuples.

    line_id = index into jsf_files (stable per survey line)
    channel_id = 0 for port, 1 for stbd
    """
    results = []
    for line_id, (jsf_path, cable) in enumerate(tqdm(jsf_files, desc="Georef")):
        for ch in channels:
            r = georef_line(jsf_path, mbes_tif, ch, cable_length=cable,
                            mbes_preloaded=mbes)
            if r is None:
                continue
            channel_id = 0 if "port" in ch else 1
            results.append((line_id, channel_id, r))
    return results


def pool_georef_results(results):
    """
    Concatenate per-channel georef dicts into a single flat sample pool.
    Adds line_id and channel_id arrays. Reassigns global ping_idx so
    pings from different lines don't collide.
    """
    arrays = {k: [] for k in (
        "lat", "lon", "bs_linear", "altitude",
        "inc_angle", "ping_idx", "heading",
    )}
    line_ids, channel_ids = [], []
    global_ping_offset = 0

    for line_id, channel_id, r in results:
        n = len(r["bs_linear"])
        for k in arrays:
            arrays[k].append(r[k])
        arrays["ping_idx"][-1] = arrays["ping_idx"][-1] + global_ping_offset
        line_ids.append(np.full(n, line_id, dtype=np.int32))
        channel_ids.append(np.full(n, channel_id, dtype=np.int8))
        global_ping_offset = int(arrays["ping_idx"][-1].max()) + 1

    pooled = {k: np.concatenate(v) for k, v in arrays.items()}
    pooled["line_id"] = np.concatenate(line_ids)
    pooled["channel_id"] = np.concatenate(channel_ids)
    return pooled


def normalize_gain(pooled, ref_percentile=40):
    """
    AGC compensation: normalize bs_linear per (line, channel) by
    `ref_percentile`-th percentile.

    Returns a new pooled dict with normalized bs_linear; original is
    not modified.

    EdgeTech's per-ping dynamic AGC produces up to 50x amplitude
    differences across survey lines. This places all lines on a
    comparable linear scale before Zhao's radiometric correction.
    """
    pooled = {k: v.copy() if hasattr(v, "copy") else v
              for k, v in pooled.items()}
    bs_lin = pooled["bs_linear"]

    refs = []
    n_groups = 0
    for line in np.unique(pooled["line_id"]):
        for ch in (0, 1):
            mask = (pooled["line_id"] == line) & (pooled["channel_id"] == ch)
            if mask.sum() < 100:
                continue
            ref = np.percentile(bs_lin[mask], ref_percentile)
            if ref > 0:
                bs_lin[mask] = bs_lin[mask] / ref
                refs.append(float(ref))
                n_groups += 1

    print(f"  Normalized {n_groups} (line, channel) groups")
    if refs:
        refs = np.array(refs)
        print(f"  Ref range: p5={np.percentile(refs, 5):.3f}, "
              f"median={np.median(refs):.3f}, "
              f"p95={np.percentile(refs, 95):.3f}")
        print(f"  Max/min ratio: {refs.max() / refs.min():.1f}x")

    return pooled


def prepare_sss_samples(channels, mode="full"):
    """
    High-level convenience: from yaml + channels list, return
    correction-ready sample-level data.

    Encapsulates: collect → load_mbes → georef_all → pool →
                  normalize_gain → run_correction.

    Returns a dict with bs_db (corrected, sample-level), sample_labels
    (cluster id per sample), and projected x_m/y_m alongside the
    original pooled fields. None if no valid samples.
    """
    cfg = get_config()
    sss_cfg = cfg["sss"]
    mbes_tif = ROOT / cfg["mbes"]["bathymetry_tif"]

    print(f"  Channels: {channels}")
    print(f"  Mode: {mode}")

    jsf_files = collect_jsf_files(sss_cfg)
    print(f"  Found {len(jsf_files)} .jsf files")

    mbes = load_mbes(mbes_tif)
    results = georef_all(jsf_files, channels, mbes, mbes_tif)
    if not results:
        return None

    pooled = pool_georef_results(results)
    print(f"  Pooled {len(pooled['bs_linear']):,} samples")

    pooled = normalize_gain(pooled)
    correction = run_correction(pooled, sss_cfg["cluster"], mode=mode)

    # Project lat/lon to grid CRS
    epsg = cfg["grid"]["epsg"]
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x_m, y_m = tr.transform(pooled["lon"], pooled["lat"])

    return {
        **pooled,
        "x_m": np.asarray(x_m, dtype=np.float64),
        "y_m": np.asarray(y_m, dtype=np.float64),
        "bs_db": correction["bs_db"],
        "sample_labels": correction["sample_labels"],
        "diagnostics": correction["diagnostics"],
    }