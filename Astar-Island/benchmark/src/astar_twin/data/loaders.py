from __future__ import annotations

from pathlib import Path

from astar_twin.data.models import RoundFixture


def load_fixture(path: Path) -> RoundFixture:
    """Load a RoundFixture from *path*.

    *path* may be either:
    - The ``round_detail.json`` file itself, or
    - The round directory (e.g. ``data/rounds/<round_id>``), in which case
      ``round_detail.json`` is appended automatically.
    """
    if path.is_dir():
        path = path / "round_detail.json"
    return RoundFixture.model_validate_json(path.read_text())


def list_fixtures(data_dir: Path) -> list[RoundFixture]:
    fixture_paths = sorted(data_dir.glob("rounds/*/round_detail.json"))
    return [load_fixture(path) for path in fixture_paths]
