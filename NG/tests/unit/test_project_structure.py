from __future__ import annotations

from pathlib import Path

from src.ng_data.cli.doctor import REQUIRED_PATHS, find_missing_paths


def test_required_paths_exist_in_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    missing_paths = find_missing_paths(repo_root)

    assert missing_paths == []
    assert len(REQUIRED_PATHS) > 0
