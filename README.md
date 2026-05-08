# Hydrospatial Data Cube

Multi-sensor underwater survey data processing pipeline (MBES, SSS, SBP, magnetometer) with a web viewer for cross-instrument exploration. Site: Mudan Reservoir.

> Work in progress — not a finished release.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

VS Code auto-detects `.venv/`. Outside VS Code, prefix commands with `uv run`.

## Build pipeline

Run scripts in order. Each writes to `outputs/`:

```
scripts/build_mbes_bathymetry.py     # XYZ → bathymetry GeoTIFF
scripts/build_mbes_terrain.py        # VRM, slope, BPI
scripts/find_sbp_cc.py               # → manually update yaml: sbp.calibration_constant
scripts/build_sss_backscatter.py     # SSS HF + LF mosaics
scripts/build_sbp_sediment.py        # SBP RL → sediment classification
scripts/build_mag_anomaly.py         # magnetometer anomaly + residual
scripts/detect_mag_targets.py        # → mag_targets.csv
scripts/build_sediment_rgb.py        # RGB tile (depends on sediment classification)
scripts/build_tracklines.py          # tracklines GeoJSON
scripts/build_contours.py            # contours GeoJSON
scripts/build_datacube.py            # pack everything into NetCDF
```

Plot scripts in `scripts/plot/` are independent; they read finished outputs.

## Viewer

```
uv run python viewer/server.py
```

Open http://localhost:8000.

## Structure

```
src/             Python library (per-sensor logic)
scripts/         Build pipeline
scripts/plot/    Figure generation
viewer/          Web viewer (frontend + server)
configs/         Site YAMLs
data/            Raw survey data (gitignored)
outputs/         Build artifacts (gitignored)
```

## Adding a site

1. Create `configs/<site>.yaml` from `configs/mudan.yaml`
2. Place raw data in `data/`
3. Run with `--config configs/<site>.yaml`

When `configs/` has multiple YAMLs, every script requires explicit `--config`.