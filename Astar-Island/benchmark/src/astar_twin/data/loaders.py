from __future__ import annotations

from pathlib import Path

from astar_twin.data.models import RoundFixture


def load_fixture(path: Path) -> RoundFixture:
    return RoundFixture.model_validate_json(path.read_text())


def list_fixtures(data_dir: Path) -> list[RoundFixture]:
    fixture_paths = sorted(data_dir.glob("rounds/*/round_detail.json"))
    return [load_fixture(path) for path in fixture_paths]
