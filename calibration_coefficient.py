import numpy as np
import matplotlib.pyplot as plt
import utils as utils
import os

# ================= 參數設定 =================
# 1. 檔案設定
JSF_FILE = r'data\sbp\20251223030338.jsf'  # 直接指定檔案路徑

# 2. 範圍設定 (公尺)
START_DIST_M = 630.0   # 起始距離
END_DIST_M   = 780.0   # 結束距離

# 3. 訊號參數
BLANKING = 50          # 避開初始發射波 (Samples)
SEARCH_WIN = 300       # 搜尋二次反射的視窗大小 (Samples)
# ===========================================

def main():
    if not os.path.exists(JSF_FILE):
        print(f"Error: File not found: {JSF_FILE}")
        return

    print(f"Reading file: {os.path.basename(JSF_FILE)}...")
    
    # 1. 讀取全部資料
    data = utils.read_jsf(JSF_FILE)
    if not data: return

    # 2. 篩選指定距離範圍的資料
    print(f"Filtering range: {START_DIST_M}m - {END_DIST_M}m...")
    target_pings = []
    
    # 計算累積距離
    lons = [p['lon'] for p in data]
    lats = [p['lat'] for p in data]
    xx, yy = utils.wgs84_to_twd97(lons, lats)
    
    # np.diff 計算相鄰點距離, 累加成總距離
    dists = np.insert(np.cumsum(np.sqrt(np.diff(xx)**2 + np.diff(yy)**2)), 0, 0)

    # 篩選符合範圍的 Ping
    for i, p in enumerate(data):
        if START_DIST_M <= dists[i] <= END_DIST_M:
            target_pings.append(p)
            
    print(f"-> Selected {len(target_pings)} pings for calibration.")
    if len(target_pings) == 0: return

    # 3. 計算 CC 值
    cc_values = []
    example_viz = None # 用來存一筆畫圖用的資料

    for i, p in enumerate(target_pings):
        amps = p['amps']
        if len(amps) < 500: continue # 資料太短就跳過

        # A. 找一次反射 (A1, r1)
        search_region = amps[BLANKING:] # 避開發射波
        idx_local = np.argmax(search_region)
        idx_1 = idx_local + BLANKING
        
        amp_1 = float(amps[idx_1])
        r_1   = float(idx_1)

        if amp_1 <= 0: continue

        # B. 找二次反射 (A2, r2)
        # 物理預測：r2 應該在 r1 的兩倍處
        center_idx = idx_1 * 2
        win_start = max(0, center_idx - SEARCH_WIN)
        win_end   = min(len(amps), center_idx + SEARCH_WIN)
        
        window = amps[win_start:win_end]
        if len(window) == 0: continue
            
        idx_local_2 = np.argmax(window)
        idx_2 = win_start + idx_local_2
        
        amp_2 = float(amps[idx_2])
        r_2   = float(idx_2)

        # C. 套用 CC 公式
        # 公式: CC = (r1 * A1)^2 / (r2 * A2)
        if amp_2 > 0:
            cc = ((r_1 * amp_1)**2) / (r_2 * amp_2)
            cc_values.append(cc)
            
            # 取中間那一筆當範例圖
            if i == len(target_pings) // 2:
                example_viz = (amps, idx_1, amp_1, idx_2, amp_2, center_idx)

    # 4. 輸出統計結果
    if cc_values:
        mean_cc = np.mean(cc_values)
        std_cc = np.std(cc_values)
        
        print("\n" + "="*40)
        print(f"Calibration Result (Mean CC)")
        print(f"Samples : {len(cc_values)}")
        print(f"Mean CC : {mean_cc:.4e}")  # 科學記號顯示
        print(f"Std Dev : {std_cc:.4e}")
        print("="*40 + "\n")

        # 5. 繪製驗證圖 (確認抓得對不對)
        if example_viz:
            amps, i1, v1, i2, v2, center = example_viz
            
            plt.figure(figsize=(10, 5), dpi=120)
            plt.plot(amps, color='black', lw=1, alpha=0.6, label='Signal')
            
            # 標記 A1
            plt.plot(i1, v1, 'r^', ms=10, label='A1 (Bottom)')
            
            # 標記 A2 (搜尋範圍與抓到的點)
            plt.axvspan(center - SEARCH_WIN, center + SEARCH_WIN, color='blue', alpha=0.1, label='Search Window')
            plt.plot(i2, v2, 'b^', ms=10, label='A2 (Multiple)')

            plt.title(f"Calibration Check (Mean CC = {mean_cc:.2e})")
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.tight_layout()
            plt.show()
    else:
        print("Failed to compute CC (no valid multiple reflections found).")

if __name__ == "__main__":
    main()