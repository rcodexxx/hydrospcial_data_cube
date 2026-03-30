# scripts/build_datacube.py
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from tqdm import tqdm

TIFS = [
    (
        "bathymetry",
        Path("../../outputs/tif/mbes_bathymetry.tif"),
        -9999.0,
        "m",
        "MBES bathymetry",
    ),
    (   "slope",
        Path("../../outputs/tif/mbes_slope.tif"),
        -9999.0,
        "deg",
        "Seafloor slope"
    ),
    (
        "rugosity",
        Path("../../outputs/tif/mbes_rugosity.tif"),
        -9999.0,
        "dimensionless",
        "Seafloor rugosity",
    ),
    (
        "backscatter",
        Path("../outputs/tif/sss_lf_backscatter_db.tif"),
        -9999.0,
        "dB",
        "SSS LF backscatter (230 kHz)",
    ),
    (
        "sbp_impedance",
        Path("../../outputs/tif/sbp_impedance.tif"),
        -9999.0,
        "Pa·s/m",
        "Acoustic impedance Z from SBP",
    ),
    (
        "sbp_confidence",
        Path("../../outputs/tif/sbp_confidence.tif"),
        255,
        "0/1/255",
        "SBP confidence (0=measured, 1=RF predicted, 255=nodata)",
    ),
    (
        "mag_background",
        Path("../../outputs/tif/mag_background.tif"),
        -9999.0,
        "nT",
        "Magnetic background field (low-pass filtered)",
    ),
    (
        "mag_residual",
        Path("../../outputs/tif/mag_residual.tif"),
        -9999.0,
        "nT",
        "Magnetic residual anomaly (UCH candidate)",
    ),
    (
        "mag_confidence",
        Path("../../outputs/tif/mag_confidence_v2.tif"),
        255,
        "0/1/255",
        "MAG confidence (0=measured, 1=interpolated, 255=nodata)",
    ),
    (
        "sediment_class",
        Path("../../outputs/tif/sbp_sediment_class.tif"),
        -1,
        "class",
        "Sediment class (Hamilton Table, 0-7)",
    ),
]

SEDIMENT_LABELS = [
    "Coarse sand",
    "Fine sand",
    "Very fine sand",
    "Silty sand",
    "Sand-silt-clay",
    "Sandy silt",
    "Clayey silt",
    "Silty clay",
]

OUT_NC = Path("../../outputs/mudan_datacube.nc")


def read_channel(tif_path, nodata):
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        nd = nodata if nodata is not None else src.nodata
        transform = src.transform
        height, width = src.height, src.width
        bounds = src.bounds
    if nd is not None:
        data[data == nd] = np.nan
    return data, transform, height, width, bounds


def main():
    # 1. Read reference grid from first channel
    _, transform, height, width, bounds = read_channel(*TIFS[0][1:3])

    res = transform.a  # 0.5m
    xs = transform.c + (np.arange(width) + 0.5) * res
    ys = transform.f + (np.arange(height) + 0.5) * (-res)

    print(f"Grid: ({height}, {width}), res={res}m")
    print(f"Easting : {xs.min():.1f} ~ {xs.max():.1f}")
    print(f"Northing: {ys.min():.1f} ~ {ys.max():.1f}")

    # 2. Build xarray Dataset
    data_vars = {}

    for name, path, nodata, units, long_name in tqdm(TIFS, desc="Loading"):
        data, t, h, w, _ = read_channel(path, nodata)
        assert (h, w) == (height, width), f"{name}: shape mismatch"

        data_vars[name] = xr.DataArray(
            data,
            dims=["northing", "easting"],
            attrs={
                "units": units,
                "long_name": long_name,
                "source": str(path.name),
            },
        )
        print(
            f"  {name:15s}: range=[{np.nanmin(data):.4g}, {np.nanmax(data):.4g}]"
            f"  nan={np.isnan(data).sum()}"
        )

    # 3. Create Dataset with coordinates
    ds = xr.Dataset(
        data_vars,
        coords={
            "easting": (
                "easting",
                xs,
                {"units": "m", "long_name": "Easting (TWD97 TM2)"},
            ),
            "northing": (
                "northing",
                ys,
                {"units": "m", "long_name": "Northing (TWD97 TM2)"},
            ),
        },
        attrs={
            "title": "Mudan Reservoir Hydrospace Data Cube",
            "institution": "TKUOC",
            "source": "MBES, SSS, SBP, MAG",
            "history": f"Created {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "crs": "EPSG:3826 (TWD97 TM2)",
            "resolution": "0.5m",
            "survey_date": "2025-12-23 / 2025-12-24",
            "instruments": "MBES, Edgetech 4205MP SSS, Edgetech 3200XS SBP, SeaSPY2 MAG",
            "channels": ", ".join([name for name, *_ in TIFS]),
            "sediment_labels": ", ".join(
                [f"{i}={v}" for i, v in enumerate(SEDIMENT_LABELS)]
            ),
            "Conventions": "CF-1.8",
            "sound_velocity_ms": "1488.931",
            "sound_velocity_source": "SVP cast 20251222 06:38 and 20251223 23:48",
            "sound_velocity_range": "1488.674 ~ 1489.253 m/s",
            "sdep_m_per_sample": "0.015247",
        },
    )

    # 4. Save to NetCDF
    print(f"\nSaving to {OUT_NC}...")
    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32"} for name, *_ in TIFS
    }
    ds.to_netcdf(OUT_NC, encoding=encoding)

    size_mb = OUT_NC.stat().st_size / 1e6
    print(f"Saved: {OUT_NC} ({size_mb:.1f} MB)")

    # 5. Summary
    print(f"\n=== Data Cube Summary ===")
    print(f"Shape     : ({height}, {width}, {len(TIFS)})")
    print(f"Variables : {len(data_vars)}")
    print(ds)


if __name__ == "__main__":
    main()
