import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from src import config, data_loader, utils


def main():
    input_file = config.MB_PATH
    output_file = str(config.TIF_PATH.parent / config.TIF_PATH.name)

    df = data_loader.read_xyz(input_file)

    df["x"] = df["x"].round(3)
    df["y"] = df["y"].round(3)

    df["z"] = np.abs(df["z"])

    grid_df = df.pivot(index="y", columns="x", values="z")
    grid_df = grid_df.sort_index(ascending=False)

    grid_depth = grid_df.values
    x_coords = grid_df.columns.values
    y_coords = grid_df.index.values

    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()

    min_lon, min_lat = utils.twd97_to_wgs84(np.array([min_x]), np.array([min_y]))
    max_lon, max_lat = utils.twd97_to_wgs84(np.array([max_x]), np.array([max_y]))

    deg_width = (max_lon[0] - min_lon[0]) / grid_depth.shape[1]
    deg_height = (max_lat[0] - min_lat[0]) / grid_depth.shape[0]

    transform = from_origin(min_lon[0], max_lat[0], deg_width, deg_height)

    with rasterio.open(
        output_file,
        "w",
        driver="GTiff",
        height=grid_depth.shape[0],
        width=grid_depth.shape[1],
        count=1,
        dtype=grid_depth.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(grid_depth, 1)


if __name__ == "__main__":
    main()
