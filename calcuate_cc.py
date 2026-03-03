import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import calculation, config, data_loader


def main():
    all_files = sorted(config.SBP_PATH.glob("*.jsf"))

    targets = {5: (1300, 1500), 11: (300, 500), 14: (100, 300)}

    fig, ax = plt.subplots(**config.FIGURE_STYLE, layout="tight")

    # 👉 準備兩個 List：一個存 dB 畫圖用，一個存原始數值計算用
    all_lines_data = []  # 存 dB 值
    all_lines_raw_data = []  # 存原始 CC 值

    for file_idx, (start_idx, end_idx) in targets.items():
        if file_idx >= len(all_files):
            continue

        jsf_file = all_files[file_idx]
        data = data_loader.read_jsf(jsf_file)
        if not data:
            continue

        target_pings = data[start_idx:end_idx]
        cc_db_values = []
        cc_raw_values = []  # 👉 新增：用來暫存這條線的原始 CC

        for p in target_pings:
            result = calculation.calculate_ping_cc(p["amps"])
            if result is not None:
                cc, *_ = result
                if cc > 0:
                    cc_db_values.append(20 * np.log10(cc))
                    cc_raw_values.append(cc)  # 👉 把原始 cc 存起來
                else:
                    cc_db_values.append(np.nan)
                    cc_raw_values.append(np.nan)
            else:
                cc_db_values.append(np.nan)
                cc_raw_values.append(np.nan)

        # 使用 pandas 填補偶發的空值
        cc_series = pd.Series(cc_db_values).ffill().bfill()
        raw_series = pd.Series(cc_raw_values).ffill().bfill()  # 👉 原始值也做同樣的填補

        # 畫出單線均線
        ma_values = cc_series.rolling(10, min_periods=1).mean()
        ax.plot(
            ma_values.index,
            ma_values,
            linewidth=1,
            alpha=0.3,
            label=f"File {file_idx} (MA)",
        )

        # 把資料存起來
        all_lines_data.append(cc_series.values)
        all_lines_raw_data.append(raw_series.values)  # 👉 收集原始值

    # 計算總平均
    if all_lines_data:
        # 1. 處理 dB 值的平均與繪圖
        avg_db_data = np.nanmean(all_lines_data, axis=0)
        final_ma_db = pd.Series(avg_db_data).rolling(10, min_periods=1).mean()
        overall_mean_db = final_ma_db.mean()
        ax.plot(final_ma_db.index, final_ma_db, color="red", linewidth=3)

        # 2. 處理原始 CC 值的平均 (算出真正的整體原始 CC 平均)
        avg_raw_data = np.nanmean(all_lines_raw_data, axis=0)
        overall_mean_raw = np.nanmean(avg_raw_data)  # 所有 ping、所有線的總體原始平均

        # 👉 印出兩個數值讓你對照
        print("-" * 30)
        print(f"整體平均 CC (dB): {overall_mean_db:.2f} dB")
        print(
            f"整體平均 CC (原始): {overall_mean_raw:}"
        )  # <--- 這就是你要丟給 calculate_ping_rl 的數字
        print("-" * 30)

    ax.set_title("CC Values Variation in dB (Moving Average)")
    ax.set_xlabel("Relative Ping Index")
    ax.set_ylabel("CC Value (dB)")
    ax.set_ylim(90, 180)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.show()


if __name__ == "__main__":
    main()
