"""
Runtime config loader.

Resolves the active site config from (in priority order):
  1. --config CLI arg
  2. HYDRO_CONFIG env var
  3. Error with list of available configs

Usage:
    from src.config import get_config, ROOT
    cfg = get_config()
"""
import argparse
import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent

def _find_config_path() -> Path:
    # Peek at --config without consuming other args
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    args, _ = parser.parse_known_args()

    name = args.config or os.environ.get("HYDRO_CONFIG")

    if not name:
        available = sorted((ROOT / "configs").glob("*.yaml"))
        listing = "\n  ".join(str(p.relative_to(ROOT)) for p in available) or "(none found)"
        raise RuntimeError(
            "No config specified. Use either:\n"
            "  python build_sss_backscatter.py --config configs/mudan.yaml\n"
            "  export HYDRO_CONFIG=configs/mudan.yaml\n"
            f"\nAvailable configs:\n  {listing}"
        )

    path = (ROOT / name).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return path


@lru_cache(maxsize=1)
def get_config() -> dict:
    return yaml.safe_load(_find_config_path().read_text(encoding="utf-8"))