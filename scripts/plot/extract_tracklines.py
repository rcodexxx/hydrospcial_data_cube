import json
from pathlib import Path

import numpy as np
from pyproj import Transformer

from src.sbp.read_sbp_jsf import read_sbp_jsf
from src.sss.read_sss_jsf import read_sss_jsf

ROOT = Path(".")
OUT = ROOT / "outputs" / "tracklines.json"
to_ll = Transformer.from_crs("EPSG:3826", "EPSG:4326", always_xy=True)

features = []

# SSS tracklines
for day_dir in [ROOT / "data/sss/20251223", ROOT / "data/sss/20251224"]:
    if not day_dir.exists():
        continue
    for jsf in sorted(day_dir.glob("*.jsf")):
        try:
            data = read_sss_jsf(jsf)
        except Exception:
            continue
        # use first available channel for coordinates
        for ch in ["HF_stbd", "LF_stbd", "HF_port", "LF_port"]:
            if ch not in data:
                continue
            d = data[ch]
            valid = ~np.isnan(d["lon"]) & ~np.isnan(d["lat"])
            if valid.sum() < 2:
                break
            lons = d["lon"][valid]
            lats = d["lat"][valid]
            # subsample to reduce size (every 10th point)
            step = max(1, len(lons) // 100)
            coords = [
                [float(lons[i]), float(lats[i])] for i in range(0, len(lons), step)
            ]
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "file": jsf.name,
                        "instrument": "SSS",
                        "day": day_dir.name,
                        "pings": int(valid.sum()),
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords,
                    },
                }
            )
            break  # one channel per file is enough

# SBP tracklines
sbp_dir = ROOT / "data/sbp"
if sbp_dir.exists():
    for jsf in sorted(sbp_dir.glob("*.jsf")):
        try:
            data = read_sbp_jsf(jsf)
        except Exception:
            continue
        if "SBP" not in data:
            continue
        d = data["SBP"]
        valid = ~np.isnan(d["lon"]) & ~np.isnan(d["lat"])
        if valid.sum() < 2:
            continue
        lons = d["lon"][valid]
        lats = d["lat"][valid]
        step = max(1, len(lons) // 100)
        coords = [[float(lons[i]), float(lats[i])] for i in range(0, len(lons), step)]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "file": jsf.name,
                    "instrument": "SBP",
                    "pings": int(valid.sum()),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
            }
        )

geojson = {"type": "FeatureCollection", "features": features}

with open(OUT, "w") as f:
    json.dump(geojson, f)

print(f"Saved: {OUT}")
print(
    f"  SSS tracklines: {sum(1 for f in features if f['properties']['instrument']=='SSS')}"
)
print(
    f"  SBP tracklines: {sum(1 for f in features if f['properties']['instrument']=='SBP')}"
)
