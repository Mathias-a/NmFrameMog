from __future__ import annotations

import json
from pathlib import Path

from astar_twin.data.models import RoundFixture


def write_fixture(fixture: RoundFixture, path: Path) -> None:
    """Write a RoundFixture to *path* as JSON.

    Creates parent directories if they do not exist.  The file is written
    atomically by serialising to a string first and then writing in a single
    call so a partial write cannot corrupt the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(fixture.model_dump_json())
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
