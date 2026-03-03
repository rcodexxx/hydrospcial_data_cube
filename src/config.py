from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MB_PATH = BASE_DIR / "data" / "multibeam" / "G1m_142m.txt"
SBP_PATH = BASE_DIR / "data" / "sbp"
TIF_PATH = BASE_DIR / "data" / "G1m_142m.tif"

FIGURE_STYLE = {
    "figsize": (8, 6),
    "dpi": 150,
}

CC = 80970325.5244343
