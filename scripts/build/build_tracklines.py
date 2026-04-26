"""
Build tracklines GeoJSON from raw SSS/SBP .jsf files.

Output is read by /api/tracklines endpoint. Each LineString feature
has properties: file, instrument (SSS|SBP), pings.

Coordinates are subsampled (~100 points per trackline) to keep the
GeoJSON small and avoid loading hundreds of points per line in the
browser.
"""
import json

import numpy as np
from pyproj import Transformer

from src.config import get_config, ROOT
from src.sbp.read_sbp_jsf import read_sbp_jsf
from src.sss.read_sss_jsf import read_sss_jsf

SUBSAMPLE_TARGET = 100


def subsample_coords(lons, lats, target=SUBSAMPLE_TARGET):
    step = max(1, len(lons) // target)
    return [[float(lons[i]), float(lats[i])] for i in range(0, len(lons), step)]


def collect_sss(survey_dirs):
    features = []
    for entry in survey_dirs:
        d = ROOT / entry["path"]
        if not d.exists():
            print(f"  skip (missing): {d}")
            continue
        for jsf in sorted(d.glob("*.jsf")):
            try:
                data = read_sss_jsf(jsf)
            except Exception as e:
                print(f"  read fail {jsf.name}: {e}")
                continue
            for ch in ["HF_stbd", "LF_stbd", "HF_port", "LF_port"]:
                if ch not in data:
                    continue
                ch_data = data[ch]
                valid = ~np.isnan(ch_data["lon"]) & ~np.isnan(ch_data["lat"])
                if valid.sum() < 2:
                    break
                lons, lats = ch_data["lon"][valid], ch_data["lat"][valid]
                features.append({
                    "type": "Feature",
                    "properties": {
                        "file": jsf.name,
                        "instrument": "SSS",
                        "pings": int(valid.sum()),
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": subsample_coords(lons, lats),
                    },
                })
                break
    return features


def collect_sbp(survey_dirs):
    features = []
    for entry in survey_dirs:
        d = ROOT / entry["path"]
        if not d.exists():
            print(f"  skip (missing): {d}")
            continue
        for jsf in sorted(d.glob("*.jsf")):
            try:
                data = read_sbp_jsf(jsf)
            except Exception as e:
                print(f"  read fail {jsf.name}: {e}")
                continue
            if "SBP" not in data:
                continue
            ch_data = data["SBP"]
            valid = ~np.isnan(ch_data["lon"]) & ~np.isnan(ch_data["lat"])
            if valid.sum() < 2:
                continue
            lons, lats = ch_data["lon"][valid], ch_data["lat"][valid]
            features.append({
                "type": "Feature",
                "properties": {
                    "file": jsf.name,
                    "instrument": "SBP",
                    "pings": int(valid.sum()),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": subsample_coords(lons, lats),
                },
            })
    return features


def main():
    cfg = get_config()
    out_path = ROOT / cfg["viewer"]["tracklines"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Collecting SSS tracklines...")
    sss = collect_sss(cfg["sss"]["survey_dirs"])

    print("Collecting SBP tracklines...")
    sbp = collect_sbp(cfg["sbp"]["survey_dirs"])

    geojson = {
        "type": "FeatureCollection",
        "features": sss + sbp,
    }

    with open(out_path, "w") as f:
        json.dump(geojson, f)

    print(f"\nSaved: {out_path.relative_to(ROOT)}")
    print(f"  SSS tracklines: {len(sss)}")
    print(f"  SBP tracklines: {len(sbp)}")


if __name__ == "__main__":
    main()