"""
Pack all GeoTIFF layers into a single NetCDF4 data cube.
Layers are resampled to the reference grid (typically MBES bathymetry).

Layer registry is yaml-driven: see cube.layers in configs/<site>.yaml.
Each layer must specify a path (relative to repo ROOT).
"""
import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.warp import reproject

from src.config import get_config, ROOT


# ──────────────────────────────────────────────────────────
# Categorical nodata sentinels (must match writer conventions)
# ──────────────────────────────────────────────────────────
NODATA_UINT8 = 255
NODATA_INT8 = -1


def resample_to_ref(src_path, ref_transform, ref_crs,
                    ref_height, ref_width, resampling):
    """Reproject and resample a GeoTIFF onto the reference grid."""
    with rasterio.open(src_path) as src:
        src_nodata = src.nodata
        data = np.full((ref_height, ref_width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=resampling,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )
    return data, src_nodata


def categorical_to_float(arr, dtype):
    """Replace categorical nodata sentinels with NaN."""
    out = arr.astype(np.float32)
    if dtype == "uint8":
        out[out == NODATA_UINT8] = np.nan
    elif dtype == "int8":
        out[out == NODATA_INT8] = np.nan
    return out


def main():
    cfg = get_config()
    cube_cfg = cfg["cube"]

    ref_path = ROOT / cube_cfg["reference_grid"]
    out_nc = ROOT / cube_cfg["output"]

    with rasterio.open(ref_path) as src:
        ref_transform = src.transform
        ref_crs = src.crs
        ref_height, ref_width = src.height, src.width
        ref_res = src.transform.a

    xs = ref_transform.c + (np.arange(ref_width) + 0.5) * ref_res
    ys = ref_transform.f + (np.arange(ref_height) + 0.5) * (-ref_res)

    ds = xr.Dataset(
        coords={
            "y": ("y", ys, {"units": "m", "long_name": f"Northing {ref_crs}"}),
            "x": ("x", xs, {"units": "m", "long_name": f"Easting {ref_crs}"}),
        },
        attrs={
            "title": "Hydrospatial Data Cube",
            "crs": str(ref_crs),
            "resolution_m": ref_res,
            "conventions": "CF-1.8",
        },
    )

    print(f"Reference grid: {ref_height}x{ref_width} @ {ref_res} m/px")
    print(f"Building data cube -> {out_nc.name}\n")

    skipped = []
    for layer in cube_cfg["layers"]:
        var = layer["var"]
        src_path = ROOT / layer["path"]

        if not src_path.exists():
            print(f"  SKIP  {var:18s}  ({src_path.name} not found)")
            skipped.append(var)
            continue

        resamp = (Resampling.nearest if layer["kind"] == "categorical"
                  else Resampling.bilinear)
        arr, _ = resample_to_ref(src_path, ref_transform, ref_crs,
                         ref_height, ref_width, resampling=resamp)

        if layer["kind"] == "categorical":
            arr = categorical_to_float(arr, layer["dtype"])

        ds[var] = xr.DataArray(
            arr, dims=["y", "x"],
            attrs={
                "units": layer["units"],
                "long_name": layer["desc"],
                "source": str(src_path.relative_to(ROOT)),
                "kind": layer["kind"],
            },
        )
        print(f"  Added {var:18s}  ({src_path.name})")

    ds.to_netcdf(out_nc, engine="netcdf4")

    print(f"\nSaved: {out_nc.relative_to(ROOT)}")
    print(f"  Layers : {len(ds.data_vars)}")
    print(f"  Size   : {out_nc.stat().st_size / 1e6:.1f} MB")
    if skipped:
        print(f"  Skipped: {skipped}")

    print("\nLayer statistics:")
    print(f"  {'variable':<18} {'min':>10} {'median':>10} {'max':>10} {'coverage':>10}")
    print("  " + "-" * 62)
    total = ref_height * ref_width
    for var in ds.data_vars:
        v = ds[var].values
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        print(f"  {var:<18} {v.min():>10.2f} {np.median(v):>10.2f} "
              f"{v.max():>10.2f} {100*v.size/total:>9.1f}%")


if __name__ == "__main__":
    main()