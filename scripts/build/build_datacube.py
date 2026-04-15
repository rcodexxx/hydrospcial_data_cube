# scripts/build/build_datacube.py
"""
Pack all GeoTIFF layers into a single NetCDF4 data cube.
"""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
import xarray as xr

ROOT    = Path(__file__).parent.parent.parent
TIF_DIR = ROOT / "outputs/tif"
OUT_NC  = ROOT / "outputs/hydrospatial_datacube.nc"

LAYERS = {
    # (filename, variable_name, units, description)
    "mbes_bathymetry.tif":      ("bathymetry",      "m",      "Depth (positive down)"),
    "mbes_vrm.tif":             ("vrm",             "",       "Vector Ruggedness Measure"),
    "sss_backscatter_lf.tif":   ("bs_lf",           "dB",     "SSS Backscatter 225 kHz"),
    "sss_backscatter_hf.tif":   ("bs_hf",           "dB",     "SSS Backscatter 830 kHz"),
    "sbp_rl.tif":               ("rl",              "dB",     "Reflection Loss"),
    "sbp_impedance.tif":        ("impedance",       "Pa·s/m", "Acoustic Impedance"),
    "sbp_pulse_width.tif":      ("pulse_width",     "m",      "Seafloor Return Pulse Width"),
    "sbp_sediment_class.tif":   ("sediment_class",  "",       "Hamilton Sediment Classification"),
    "sbp_isopach.tif":          ("isopach",         "m",      "Sediment Thickness"),
    "mag_background.tif":       ("mag_background",  "nT",     "Magnetic Background Field"),
    "mag_residual.tif":         ("mag_residual",    "nT",     "Magnetic Residual Anomaly"),
}


def main():
    # read reference grid from bathymetry
    ref_path = TIF_DIR / "mbes_bathymetry.tif"
    with rasterio.open(ref_path) as src:
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

    res = transform.a
    xs = transform.c + (np.arange(width) + 0.5) * res
    ys = transform.f + (np.arange(height) + 0.5) * (-res)

    ds = xr.Dataset(
        coords={
            "y": ("y", ys, {"units": "m", "long_name": "Northing (EPSG:3826)"}),
            "x": ("x", xs, {"units": "m", "long_name": "Easting (EPSG:3826)"}),
        },
        attrs={
            "title": "Hydrospatial Data Cube - Mudan Reservoir",
            "crs": str(crs),
            "resolution_m": res,
        },
    )

    for filename, (varname, units, desc) in LAYERS.items():
        tif_path = TIF_DIR / filename
        if not tif_path.exists():
            print(f"  SKIP: {filename} (not found)")
            continue

        with rasterio.open(tif_path) as src:
            data = src.read(1).astype(np.float32)
            nd = src.nodata

        if nd is not None:
            data[data == nd] = np.nan

        ds[varname] = xr.DataArray(
            data, dims=["y", "x"],
            attrs={"units": units, "long_name": desc},
        )
        print(f"  Added: {varname} ({filename})")

    ds.to_netcdf(OUT_NC, engine="netcdf4")
    print(f"\nSaved: {OUT_NC}")
    print(f"  Layers: {len(ds.data_vars)}")
    print(f"  Size: {OUT_NC.stat().st_size / 1e6:.1f} MB")

    try:
        with rasterio.open(OUT_NC, 'r+') as dst:
            dst.build_overviews([2, 4, 8, 16], Resampling.average)
            dst.update_tags(ns='rio_overview', resampling='average')
        print("✅ 成功建立 Overviews")
    except Exception as e:
        print(f"⚠️ 無法自動建立 Overviews: {e}")


if __name__ == "__main__":
    main()