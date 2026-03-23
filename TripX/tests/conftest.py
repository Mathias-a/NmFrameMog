from __future__ import annotations

import sys
from pathlib import Path


def _ensure_tripx_on_path() -> None:
    tripx_root = Path(__file__).resolve().parents[1]
    tripx_root_str = str(tripx_root)
    if tripx_root_str not in sys.path:
        sys.path.insert(0, tripx_root_str)


_ensure_tripx_on_path()
