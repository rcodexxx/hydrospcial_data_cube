# src/calculation.py
from typing import Tuple

import numpy as np


def calculate_ping_cc(amps, blanking=50, search_win=300):
    if len(amps) < 500:
        return None

    search_region = amps[blanking:]
    idx_local = np.argmax(search_region)
    idx_1 = idx_local + blanking
    amp_1 = float(amps[idx_1])
    r_1 = float(idx_1)

    if amp_1 <= 0:
        return None

    center_idx = idx_1 * 2
    win_start = max(0, center_idx - search_win)
    win_end = min(len(amps), center_idx + search_win)

    window = amps[win_start:win_end]
    if len(window) == 0:
        return None

    idx_local_2 = np.argmax(window)
    idx_2 = win_start + idx_local_2
    amp_2 = float(amps[idx_2])
    r_2 = float(idx_2)

    if amp_2 > 0:
        cc = ((r_1 * amp_1) ** 2) / (r_2 * amp_2)
        return (cc, idx_1, amp_1, idx_2, amp_2, center_idx)

    return None


def calculate_ping_rl(amps, cc_value, blanking=50):
    if len(amps) < 100:
        return None

    search_region = amps[blanking:]
    if len(search_region) == 0:
        return None

    idx_max = np.argmax(search_region) + blanking
    amp_1 = float(amps[idx_max])
    r_1 = float(idx_max)

    if amp_1 <= 0:
        return None

    R = (r_1 * amp_1) / cc_value

    rl = -20 * np.log10(R)

    return rl
