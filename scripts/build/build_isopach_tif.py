"""
Build isopach (sediment thickness) layer from SBP data using Acoustic Physical Principles.

Workflow:
  1. Extract acoustic envelope via Hilbert Transform.
  2. Map spatial coordinates to Sediment Classification (Variable Sound Speed).
  3. Track Sub-Bottom Reflector (SBR) using Peak Prominence & Local Sound Speed.
  4. Filter out false Multiples (BS) using Sonar Draft geometry.
  5. Compute physical thickness via Two-Way Travel Time (TWT) dynamically.
  6. Use Spatial Interpolation (IDW) to map thickness.
"""
from pathlib import Path
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.signal import find_peaks, hilbert
from scipy.spatial import cKDTree
from tqdm import tqdm

from src.data_loader.read_sbp_jsf import read_sbp_jsf
from src.config import EPSG, SOUND_SPEED

# ==========================================
# 路徑與常數設定
# ==========================================
ROOT        = Path(__file__).parent.parent.parent
SBP_PATH    = ROOT / "data/sbp"
MBES_TIF    = ROOT / "outputs/tif/mbes_bathymetry.tif"
SED_TIF     = ROOT / "outputs/tif/sbp_sediment_class.tif"  # 新增：底質分類圖
OUT_THICK   = ROOT / "outputs/tif/sbp_isopach.tif"

NS          = 20480e-9
THICK_CLIP  = (0.1, 3.0)  # 厚度合理範圍 (公尺)
SONAR_DRAFT = 1.04        # 聲納吃水深度


VP_DICT = {
    0: SOUND_SPEED * 1.201,  # Coarse sand
    1: SOUND_SPEED * 1.145,  # Fine sand
    2: SOUND_SPEED * 1.115,  # Very fine sand
    3: SOUND_SPEED * 1.078,  # Silty sand
    4: SOUND_SPEED * 1.080,  # Sandy silt
    5: SOUND_SPEED * 1.057,  # Silt
    6: SOUND_SPEED * 1.033,  # Sandy-silt-clay
    7: SOUND_SPEED * 1.014,  # Silty clay
    8: SOUND_SPEED * 0.994,  # Clayey silt
    9: SOUND_SPEED * 1.000,  # Framework-supported mud
    10: SOUND_SPEED * 0.9, # Fluid mud
}

# ==========================================
# 聲學處理函數
# ==========================================
def find_b(amps, blanking=50):
    """鎖定一次海床 (Sea Floor) 位置"""
    return int(np.argmax(amps[blanking:])) + blanking

def extract_valid_thickness(amps, idx_b, local_v):
    """
    結合包絡線、防雷邏輯與「動態區域聲速」的厚度萃取核心
    """
    # 根據該點的材質聲速，計算該點專屬的 SDEP (每取樣點代表深度)
    local_sdep = NS * local_v / 2
    
    s = idx_b + int(0.15 / local_sdep)  # 盲區 15cm
    e = min(len(amps), idx_b + int(4.0 / local_sdep)) # 探測深度 4m
    
    if e - s < 10:
        return None

    # 1. 包絡線萃取
    window = amps[s:e].astype(np.float64)
    envelope = np.abs(hilbert(window))
    db_win = 20 * np.log10(np.maximum(envelope, 1.0))
    noise_floor = np.median(db_win) + 5.0

    # 2. 尋找 SBR
    peaks, properties = find_peaks(
        db_win, 
        prominence=3.0, 
        distance=int(0.15 / local_sdep), # 動態距離限制
        height=noise_floor,
        width=2
    )

    if len(peaks) == 0:
        return None

    best_idx = np.argmax(properties["prominences"] * properties["peak_heights"])
    best_peak_pos = peaks[best_idx]
    
    # 3. 計算實體厚度 (套用該區域專屬聲速)
    thick = (s + best_peak_pos - idx_b) * local_sdep 

    # 4. 防雷過濾：檢查是否撞到 BS 多次反射波
    if abs(thick - SONAR_DRAFT) < 0.15:
        return None

    return thick

# ==========================================
# 主程式
# ==========================================
def main():
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)

    # 讀取底質分類圖，作為聲速查詢表
    print("0. Loading Sediment Classification Map for Velocity Correction...")
    with rasterio.open(SED_TIF) as src_sed:
        sed_data = src_sed.read(1)
        sed_transform = src_sed.transform
        sed_height, sed_width = sed_data.shape

    # 1. 萃取所有測線的有效厚度
    all_x, all_y, all_thick = [], [], []
    n_total, n_valid, n_rejected = 0, 0, 0

    print("1. Extracting Acoustic Thickness along tracklines (with variable Vp)...")
    for jsf in tqdm(sorted(SBP_PATH.glob("*.jsf")), desc="Processing JSF"):
        data = read_sbp_jsf(jsf)
        if "SBP" not in data: continue
        
        d = data["SBP"]
        valid = ~np.isnan(d["lon"])
        if valid.sum() == 0: continue
            
        x, y = transformer.transform(d["lon"][valid], d["lat"][valid])
        amps = d["amps"][valid].astype(np.float32)

        # -- 高速空間抽樣 (Vectorized Spatial Sampling) --
        # 將整條測線的 X, Y 瞬間轉換為底質圖的像素行列 (row, col)
        rows, cols = rasterio.transform.rowcol(sed_transform, x, y)
        rows = np.array(rows)
        cols = np.array(cols)
        
        # 建立一個全塞滿預設水速的陣列
        local_vs = np.full(len(x), SOUND_SPEED, dtype=np.float64)
        
        # 過濾掉超出底質圖範圍的點
        valid_rc = (rows >= 0) & (rows < sed_height) & (cols >= 0) & (cols < sed_width)
        
        # 查表替換聲速
        if valid_rc.any():
            cls_ids = sed_data[rows[valid_rc], cols[valid_rc]]
            for k, v in VP_DICT.items():
                # 找出符合該材質的點，將其聲速覆蓋上去
                match_mask = (cls_ids == k)
                local_vs[np.where(valid_rc)[0][match_mask]] = v

        # 開始逐點計算厚度
        for i in range(len(amps)):
            n_total += 1
            idx_b = find_b(amps[i])
            thick = extract_valid_thickness(amps[i], idx_b, local_vs[i])
            
            if thick is not None and THICK_CLIP[0] <= thick <= THICK_CLIP[1]:
                all_x.append(x[i])
                all_y.append(y[i])
                all_thick.append(thick)
                n_valid += 1
            else:
                n_rejected += 1

    all_x = np.array(all_x, dtype=np.float64)
    all_y = np.array(all_y, dtype=np.float64)
    all_thick = np.array(all_thick, dtype=np.float64)

    print(f"\nExtraction Summary:")
    print(f"  Total Pings: {n_total}")
    print(f"  Valid SBR Detected: {n_valid} ({100*n_valid/n_total:.1f}%)")
    print(f"  Thickness Range: {all_thick.min():.2f}m ~ {all_thick.max():.2f}m")
    print(f"  Median Thickness: {np.median(all_thick):.2f}m")

    if n_valid < 100:
        print("Error: Not enough valid points for interpolation.")
        return

    # 2. 讀取 MBES 網格作為底圖座標基礎
    print("\n2. Loading Target MBES Grid Geometry...")
    with rasterio.open(MBES_TIF) as src:
        profile = src.profile.copy()
        transform = src.transform
        height, width = src.height, src.width
        mbes_data = src.read(1)
        mbes_nodata = src.nodata

    res = transform.a
    xs_grid = transform.c + (np.arange(width) + 0.5) * res
    ys_grid = transform.f + (np.arange(height) + 0.5) * (-res)
    grid_x, grid_y = np.meshgrid(xs_grid, ys_grid)
    
    if mbes_nodata is not None:
        valid_grid = (mbes_data != mbes_nodata)
    else:
        valid_grid = np.isfinite(mbes_data)

    target_x = grid_x[valid_grid]
    target_y = grid_y[valid_grid]

    print("\n3. Performing Spatial Interpolation (IDW via KDTree)...")
    dx = np.diff(all_x)
    dy = np.diff(all_y)
    dist = np.hypot(dx, dy)
    ping_spacing = np.median(dist[dist > 0])

    MAX_GAP = 70
    max_distance = MAX_GAP * 1.5 
    k_neighbors = 200
    smoothing = MAX_GAP * 2

    points = np.column_stack([all_x, all_y])
    tree = cKDTree(points)

    target_points = np.column_stack([target_x, target_y])
    distances, indices = tree.query(target_points, k=k_neighbors, distance_upper_bound=max_distance)

    interpolated_thick = np.full(len(target_points), np.nan, dtype=np.float32)

    for i in range(len(target_points)):
        valid_mask = np.isfinite(distances[i]) & (distances[i] <= max_distance)
        if not np.any(valid_mask):
            continue 
            
        valid_dist = distances[i][valid_mask]
        valid_idx = indices[i][valid_mask]
        
        weights = 1.0 / (valid_dist ** 2 + smoothing)
        interpolated_thick[i] = np.sum(weights * all_thick[valid_idx]) / np.sum(weights)
        
    thick_2d = np.full((height, width), -9999.0, dtype=np.float32)
    valid_interp_mask = np.isfinite(interpolated_thick)
    thick_2d[valid_grid] = np.where(valid_interp_mask, interpolated_thick, -9999.0)

    # 4. 輸出 GeoTIFF
    print("\n4. Saving Isopach GeoTIFF...")
    profile.update(dtype="float32", count=1, nodata=-9999.0)
    
    with rasterio.open(OUT_THICK, "w", **profile) as dst:
        dst.write(thick_2d, 1)
        
    print(f"Successfully saved to: {OUT_THICK}")

if __name__ == "__main__":
    main()