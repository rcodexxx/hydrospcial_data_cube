import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt

from src import config, data_loader, utils


def main():
    print("Reading Multibeam background...")
    try:
        grid_depth, img_extent = data_loader.read_geotiff(config.TIF_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Warning: Multibeam file not found at {config.TIF_PATH}"
        )

    except Exception as e:
        raise Exception(f"Warning: Failed to read Multibeam file. Error: {e}")

    # all_files = sorted(config.SBP_PATH.glob("*.jsf"))[:20]
    # jsf_files = [all_files[i] for i in [5, 11, 14]]

    jsf_files = sorted(config.SBP_PATH.glob("*.jsf"))[:20]

    if not jsf_files:
        raise FileNotFoundError(f"No JSF file found at {config.SBP_PATH}")
    print(f"Found {len(jsf_files)} JSF file(s)")

    for i, jsf in enumerate(jsf_files):
        data = data_loader.read_jsf(jsf)
        print(f"測線 {i} (檔案: {jsf.name}) 總共有 {len(data)} 個資料點")

    fig, ax = plt.subplots(**config.FIGURE_STYLE, layout="tight")

    if grid_depth is not None:
        ax.imshow(
            grid_depth,
            extent=img_extent,
            cmap="turbo_r",
        )

    print(f"Plotting {len(jsf_files)} tracks...")

    for i, jsf in enumerate(jsf_files):
        data = data_loader.read_jsf(jsf)

        if not data:
            continue

        lons = [p["lon"] for p in data]
        lats = [p["lat"] for p in data]

        # for i in range(0, len(lons), 100):
        #     ax.plot(lons[i], lats[i], marker='.', markersize=8, color='w')
        #
        # for i in range(0, len(lons), 500):
        #     ax.plot(lons[i], lats[i], marker='.', markersize=8, color='r')

        ax.plot(lons, lats, linewidth=1.5, label=f"Track {i}")

    ax.legend(loc="upper right", fontsize="small")

    utils.apply_dms_ticks(ax)

    center_lat = (img_extent[2] + img_extent[3]) / 2
    utils.set_map_aspect(ax, [center_lat])

    plt.show()


if __name__ == "__main__":
    main()
