from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_TEXT = str(ROOT)

if ROOT_TEXT not in sys.path:
    sys.path.insert(0, ROOT_TEXT)
