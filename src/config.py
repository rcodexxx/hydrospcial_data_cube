"""
Runtime config loader.

Resolution rule:
  1. --config <path>           use it
  2. configs/ has one yaml     use it (auto-select)
  3. configs/ has 0 or >1      error, must specify --config

Usage:
    from src.config import get_config, ROOT
    cfg = get_config()
"""
import argparse
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


def _resolve_config_path() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    args, _ = parser.parse_known_args()

    if args.config:
        path = (ROOT / args.config).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        return path

    ymls = sorted((ROOT / "configs").glob("*.yaml"))
    if len(ymls) == 0:
        raise RuntimeError("No config files in configs/")
    if len(ymls) > 1:
        listing = "\n  ".join(str(p.relative_to(ROOT)) for p in ymls)
        raise RuntimeError(
            "Multiple configs found, please specify --config:\n  " + listing
        )

    print(f"[config] using {ymls[0].relative_to(ROOT)}")
    return ymls[0]


@lru_cache(maxsize=1)
def get_config() -> dict:
    return yaml.safe_load(_resolve_config_path().read_text(encoding="utf-8"))