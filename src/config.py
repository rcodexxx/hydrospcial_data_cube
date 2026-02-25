from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MB_PATH = BASE_DIR / "data" / "multibeam" / "G1m_142m.txt"
SBP_PATH = BASE_DIR / "data" / "sbp"

FIGURE_STYLE = {
    "figsize": (8, 6),
    "dpi": 150,
}
