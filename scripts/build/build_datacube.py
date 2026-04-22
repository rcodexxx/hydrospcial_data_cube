# scripts/build/build_datacube.py
"""
Pack all GeoTIFF layers into a single NetCDF4 data cube.
Resamples all layers to the MBES reference grid before packing.
"""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import xarray as xr

ROOT    = Path(__file__).parent.parent.parent
TIF_DIR = ROOT / "outputs/tif"
OUT_NC  = ROOT / "outputs/hydrospatial_datacube.nc"

# (filename, variable_name, units, description, dtype)
LAYERS = [
    # MBES
    ("mbes_bathymetry.tif",    "bathymetry",      "m",      "Water depth (positive down)",              "float32"),
    ("mbes_vrm.tif",           "vrm",             "",       "Vector Ruggedness Measure",                "float32"),
    # SSS
    ("sss_backscatter_lf.tif", "bs_lf",           "dB",     "SSS backscatter 230 kHz",                 "float32"),
    ("sss_backscatter_hf.tif", "bs_hf",           "dB",     "SSS backscatter 850 kHz",                 "float32"),
    ("sss_clusters_hf.tif",    "facies_hf",       "",       "Acoustic facies HF (K-means cluster ID)", "uint8"),
    ("sss_clusters_lf.tif",    "facies_lf",       "",       "Acoustic facies LF (K-means cluster ID)", "uint8"),
    # SBP
    ("sbp_rl.tif",             "rl",              "dB",     "Reflection Loss",                          "float32"),
    ("sbp_sediment_class.tif", "sediment_class",  "",       "Hamilton sediment classification (int8)",  "int8"),
    ("sbp_isopach.tif",        "isopach",         "m",      "Sediment thickness",                       "float32"),
    ("sbp_confidence.tif",     "sbp_confidence",  "",       "SBP confidence: 0=measured 1=interp",      "uint8"),
    # MAG
    ("mag_background.tif",     "mag_background",  "nT",     "Magnetic background field (IGRF residual)","float32"),
    ("mag_residual.tif",       "mag_residual",    "nT",     "Magnetic local anomaly",                   "float32"),
    ("mag_confidence.tif",     "mag_confidence",  "",       "MAG confidence: 0=measured 1=interp",      "uint8"),
]


def resample_to_ref(src_path, ref_transform, ref_crs, ref_height, ref_width,
                    resampling=Resampling.bilinear):
    """Reproject and resample a TIF to the reference grid."""
    with rasterio.open(src_path) as src:
        data = np.full((ref_height, ref_width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
    return data


def main():
    # reference grid from bathymetry
    ref_path = TIF_DIR / "mbes_bathymetry.tif"
    with rasterio.open(ref_path) as src:
        ref_transform = src.transform
        ref_crs       = src.crs
        ref_height    = src.height
        ref_width     = src.width
        ref_res       = src.transform.a

    xs = ref_transform.c + (np.arange(ref_width)  + 0.5) * ref_res
    ys = ref_transform.f + (np.arange(ref_height) + 0.5) * (-ref_res)

    ds = xr.Dataset(
        coords={
            "y": ("y", ys, {"units": "m", "long_name": "Northing EPSG:3826"}),
            "x": ("x", xs, {"units": "m", "long_name": "Easting EPSG:3826"}),
        },
        attrs={
            "title":       "Hydrospatial Data Cube — Mudan Reservoir",
            "crs":         str(ref_crs),
            "resolution_m": ref_res,
            "conventions": "CF-1.8",
        },
    )

    print("Building NetCDF Data Cube...")
    skipped = []
    for filename, varname, units, desc, dtype in LAYERS:
        tif_path = TIF_DIR / filename
        if not tif_path.exists():
            print(f"  SKIP : {filename}")
            skipped.append(filename)
            continue

        # categorical layers use nearest-neighbour resampling
        if dtype in ("uint8", "int8"):
            resamp = Resampling.nearest
        else:
            resamp = Resampling.bilinear

        data = resample_to_ref(tif_path, ref_transform, ref_crs,
                               ref_height, ref_width, resampling=resamp)

        # restore integer nodata as nan not applicable → use masked array
        if dtype == "uint8":
            arr = data.astype(np.float32)
            arr[arr == 255] = np.nan
        elif dtype == "int8":
            arr = data.astype(np.float32)
            arr[arr == -1]  = np.nan
        else:
            arr = data

        ds[varname] = xr.DataArray(
            arr, dims=["y", "x"],
            attrs={"units": units, "long_name": desc},
        )
        print(f"  Added: {varname:20s} ({filename})")

    ds.to_netcdf(OUT_NC, engine="netcdf4")

    print(f"\nSaved: {OUT_NC}")
    print(f"Layers  : {len(ds.data_vars)}")
    print(f"Grid    : {ref_height} x {ref_width} px  ({ref_res} m/px)")
    print(f"Size    : {OUT_NC.stat().st_size / 1e6:.1f} MB")

    if skipped:
        print(f"\nSkipped : {skipped}")

    # summary statistics
    print("\nLayer statistics:")
    print(f"  {'Variable':<20} {'min':>10} {'median':>10} {'max':>10} {'coverage':>10}")
    print("  " + "-" * 62)
    for varname in ds.data_vars:
        v = ds[varname].values
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        total = ref_height * ref_width
        pct   = 100 * len(v) / total
        print(f"  {varname:<20} {v.min():>10.2f} "
              f"{np.median(v):>10.2f} {v.max():>10.2f} {pct:>9.1f}%")


if __name__ == "__main__":
    main()