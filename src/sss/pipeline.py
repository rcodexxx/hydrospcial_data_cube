"""
SSS sample-level pipeline.

Orchestrates .jsf collection through Zhao correction, returning
sample-level arrays ready for downstream use (mosaicking, MAG-target
overlays, future analysis scripts).

Separates pipeline logic from CLI scripts so multiple callers can
reuse the same processing without duplicating code.
"""
import numpy as np
import rasterio
from pyproj import Transformer
from tqdm import tqdm

from src.config import get_config, ROOT
from src.sss.config import channels_for_freq
from src.sss.correction import normalize_gain, run_correction
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
    Concatenate per-channel georef dicts into a single flat sample
    pool. Adds line_id and channel_id arrays. Reassigns global
    ping_idx so pings from different lines don't collide.
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


def run_pipeline(freq):
    """
    End-to-end SSS pipeline for one frequency.

    Encapsulates: collect → load_mbes → georef → pool → normalize →
                  Zhao correction → projected coordinates.

    freq: 'HF' or 'LF'
    Returns dict with:
      - all pooled fields (lat, lon, bs_linear, altitude, ...)
      - x_m, y_m: projected coords in grid CRS
      - bs_db, sample_labels, diagnostics: from Zhao correction
    Returns None if no valid samples.
    """
    cfg = get_config()
    sss_cfg = cfg["sss"]
    mbes_tif = ROOT / cfg["mbes"]["bathymetry_tif"]

    channels = channels_for_freq(freq)
    print(f"  Frequency: {freq}  Channels: {channels}")

    jsf_files = collect_jsf_files(sss_cfg)
    print(f"  Found {len(jsf_files)} .jsf files")

    mbes = load_mbes(mbes_tif)
    results = georef_all(jsf_files, channels, mbes, mbes_tif)
    if not results:
        return None

    pooled = pool_georef_results(results)
    n = len(pooled["bs_linear"])
    print(f"  Pooled {n:,} samples")
    print(f"  altitude:  min={pooled['altitude'].min():.2f}  "
          f"p5={np.percentile(pooled['altitude'], 5):.2f}  "
          f"median={np.median(pooled['altitude']):.2f}  "
          f"p95={np.percentile(pooled['altitude'], 95):.2f}  "
          f"max={pooled['altitude'].max():.2f}")
    print(f"  inc_angle: " + "  ".join(
        f"p{q}={np.percentile(pooled['inc_angle'], q):.1f}°"
        for q in (1, 5, 50, 95, 99)
    ))

    print("\n  Gain normalization per (line, channel)...")
    pooled = normalize_gain(pooled)

    print(f"\n  Running Zhao correction...")
    correction = run_correction(pooled, sss_cfg["cluster"], freq=freq)
    diag = correction["diagnostics"]
    print(f"  h0:          {diag.get('h0', float('nan')):.2f} m")
    print(f"  n_clusters:  {diag.get('n_clusters', 0)}")
    print(f"  noise ratio: {diag.get('noise_ratio', 0):.2%}")

    # Project to grid CRS
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