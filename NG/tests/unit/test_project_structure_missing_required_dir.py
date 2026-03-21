from __future__ import annotations

from pathlib import Path

from src.ng_data.cli.doctor import find_missing_paths


def test_missing_required_path_fixture_fails_deterministically() -> None:
    fixture_root = Path(__file__).resolve().parent / "fixtures" / "missing_required_dir"

    missing_paths = find_missing_paths(fixture_root)

    assert missing_paths == [Path("src/ng_data/retrieval")]
