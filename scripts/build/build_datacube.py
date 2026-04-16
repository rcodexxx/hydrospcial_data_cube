# scripts/build/build_datacube.py
"""
Pack all GeoTIFF layers into a single NetCDF4 data cube.
And generate static GeoJSON contours for web visualization.
"""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
import xarray as xr
import math
import json
from pyproj import Transformer

# ⚠️ 確保 Matplotlib 不會呼叫 GUI，避免在伺服器或終端機當機
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geojsoncontour

ROOT    = Path(__file__).parent.parent.parent
TIF_DIR = ROOT / "outputs/tif"
OUT_NC  = ROOT / "outputs/hydrospatial_datacube.nc"
STATIC_DIR = ROOT / "src/viewer/static"

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


def export_contours(ds, varname, out_path, to_ll, interval=1.0):
    """將 Data Cube 中的指定變數轉換為等高線並存為靜態 GeoJSON"""
    print(f"  Generating contours for {varname}...")
    
    # 1. 降採樣：控制網格不超過 500x500 以維持運算與前端渲染效能
    max_grid = 500
    step = max(1, math.ceil(max(ds.sizes['x'], ds.sizes['y']) / max_grid))
    da = ds[varname].isel(x=slice(None, None, step), y=slice(None, None, step))

    z_data = da.values
    xx, yy = np.meshgrid(da.x.values, da.y.values)
    lon_grid, lat_grid = to_ll.transform(xx, yy)

    # 2. 過濾與計算級距
    valid_z = z_data[np.isfinite(z_data)]
    if len(valid_z) == 0:
        print(f"    -> Skipped (No valid data)")
        return

    z_min, z_max = np.floor(valid_z.min()), np.ceil(valid_z.max())
    depth_range = z_max - z_min
    
    # 智慧級距調整
    actual_interval = interval
    if depth_range > 100: actual_interval = max(interval, 10.0)
    elif depth_range > 50: actual_interval = max(interval, 5.0)
    elif depth_range < 5: actual_interval = 0.5

    levels = np.arange(z_min, z_max + actual_interval, actual_interval)
    if len(levels) < 2:
        print(f"    -> Skipped (Not enough variation)")
        return

    # 3. 繪製與輸出
    fig = plt.figure()
    ax = fig.add_subplot(111)
    try:
        contour = ax.contour(lon_grid, lat_grid, z_data, levels=levels)
        geojson_str = geojsoncontour.contour_to_geojson(
            contour=contour, ndigits=6, stroke_width=1
        )
        
        # 確保資料夾存在並寫入檔案
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(geojson_str, encoding="utf-8")
        print(f"    -> Saved: {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"    -> Error generating contours: {e}")
    finally:
        plt.close(fig)


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

    print("📦 Building NetCDF Data Cube...")
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
    print(f"\n✅ Saved: {OUT_NC}")
    print(f"  Layers: {len(ds.data_vars)}")
    print(f"  Size: {OUT_NC.stat().st_size / 1e6:.1f} MB")

    print("\n🗺️ Generating Static GeoJSON Contours...")
    to_ll = Transformer.from_crs("EPSG:3826", "EPSG:4326", always_xy=True)
    
    if "bathymetry" in ds.data_vars:
        export_contours(ds, "bathymetry", STATIC_DIR / "contours_bathymetry.geojson", to_ll, interval=5.0)
    
    if "mag_residual" in ds.data_vars:
        export_contours(ds, "mag_residual", STATIC_DIR / "contours_mag_residual.geojson", to_ll, interval=10.0)

    print("\n🎉 Build process complete!")


if __name__ == "__main__":
    main()