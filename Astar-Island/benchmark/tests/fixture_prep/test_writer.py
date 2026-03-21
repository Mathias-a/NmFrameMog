from __future__ import annotations

import json
from pathlib import Path

import pytest

from astar_twin.data.loaders import load_fixture
from astar_twin.data.models import RoundFixture
from astar_twin.fixture_prep.ground_truth import compute_and_attach_ground_truths
from astar_twin.fixture_prep.writer import write_fixture

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "rounds" / "test-round-001" / "round_detail.json"
)


@pytest.fixture
def fixture_with_ground_truths() -> RoundFixture:
    base = load_fixture(FIXTURE_PATH)
    return compute_and_attach_ground_truths(base, n_runs=5, base_seed=0)


def test_write_fixture_creates_file(
    tmp_path: Path, fixture_with_ground_truths: RoundFixture
) -> None:
    dest = tmp_path / "out" / "round_detail.json"
    write_fixture(fixture_with_ground_truths, dest)
    assert dest.exists()


def test_write_fixture_creates_parent_dirs(
    tmp_path: Path, fixture_with_ground_truths: RoundFixture
) -> None:
    dest = tmp_path / "a" / "b" / "c" / "round_detail.json"
    write_fixture(fixture_with_ground_truths, dest)
    assert dest.exists()


def test_write_fixture_round_trips(
    tmp_path: Path, fixture_with_ground_truths: RoundFixture
) -> None:
    dest = tmp_path / "round_detail.json"
    write_fixture(fixture_with_ground_truths, dest)
    reloaded = load_fixture(dest)
    assert reloaded.id == fixture_with_ground_truths.id
    assert reloaded.map_width == fixture_with_ground_truths.map_width
    assert reloaded.map_height == fixture_with_ground_truths.map_height
    assert reloaded.seeds_count == fixture_with_ground_truths.seeds_count
    assert reloaded.ground_truths is not None


def test_write_fixture_produces_valid_json(
    tmp_path: Path, fixture_with_ground_truths: RoundFixture
) -> None:
    dest = tmp_path / "round_detail.json"
    write_fixture(fixture_with_ground_truths, dest)
    raw = dest.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert "id" in parsed
    assert "ground_truths" in parsed
    assert parsed["ground_truths"] is not None


def test_write_fixture_overwrites_existing_file(
    tmp_path: Path, fixture_with_ground_truths: RoundFixture
) -> None:
    dest = tmp_path / "round_detail.json"
    dest.write_text("old content", encoding="utf-8")
    write_fixture(fixture_with_ground_truths, dest)
    raw = dest.read_text(encoding="utf-8")
    assert "old content" not in raw
    assert fixture_with_ground_truths.id in raw
