import glob
import os

import matplotlib.pyplot as plt

from src import config, data_loader, utils


def main():
    style = config.FIGURE_STYLE

    print("Reading Multibeam background...")
    try:
        df = data_loader.read_xyz(config.MB_PATH)
        bg_lon, bg_lat = utils.twd97_to_wgs84(df["x"].values, df["y"].values)

    except FileNotFoundError:
        raise f"Warning: Multibeam file not found at {config.MB_PATH}"

    except Exception as e:
        raise f"Warning: Failed to read Multibeam file. Error: {e}"

    jsf_files = sorted(glob.glob(os.path.join(config.SBP_PATH, "*.jsf")))[:]

    fig, ax = plt.subplots(
        figsize=style["figsize"],
        dpi=style["dpi"],
        layout="tight"
    )

    if len(bg_lon) > 0:
        ax.scatter(bg_lon, bg_lat, c=df["z"], s=1, cmap="jet")

    print(f"Plotting {len(jsf_files)} tracks...")

    for i, jsf in enumerate(jsf_files):
        data = data_loader.read_jsf(jsf)

        if not data:
            continue

        lons = [p["lon"] for p in data]
        lats = [p["lat"] for p in data]

        ax.plot(lons, lats, color="w", linewidth=1.5)

    utils.apply_dms_ticks(ax)
    utils.set_map_aspect(ax, bg_lat)

    plt.show()


if __name__ == "__main__":
    main()
