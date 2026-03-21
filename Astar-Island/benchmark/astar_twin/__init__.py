from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
src_package = Path(__file__).resolve().parent.parent / "src" / "astar_twin"
if src_package.exists():
    __path__.append(str(src_package))
