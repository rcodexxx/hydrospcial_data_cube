import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer

INPUT_FILE = r'data\multibeam\G1m_142m.txt'  

def plot_bathymetric_chart():
    try:
        df = pd.read_csv(INPUT_FILE, sep=r'\s+', header=None, names=['x', 'y', 'z'])

        transformer = Transformer.from_crs("epsg:3826", "epsg:4326", always_xy=True)
        lon, lat = transformer.transform(df['x'].values, df['y'].values)

        plt.figure(figsize=(8, 6), dpi=100)
        # sc = plt.scatter(df_plot['x'], df_plot['y'], c=df_plot['z'], s=0.01, cmap='jet')
        sc = plt.scatter(lon, lat, c=df['z'], s=0.01, cmap='jet')

        cbar = plt.colorbar(sc)
        cbar.set_label('Depth (m)')

        plt.title(f"Bathymetry Map")
        plt.xlabel("Longitude (Degree)")
        plt.ylabel("Latitude (Degree)")

        plt.ticklabel_format(useOffset=False, style='plain')

        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.3)

        plt.show()

    except FileNotFoundError:
        print(f"錯誤: 找不到檔案，請確認檔名與路徑是否正確。")
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")

if __name__ == "__main__":
    plot_bathymetric_chart()