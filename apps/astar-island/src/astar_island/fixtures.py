"""Ground-truth fixture loading for offline training and evaluation.

Reads frozen JSON fixtures from data/fixtures/ directory.
Each fixture contains initial_grid, ground_truth, and metadata for one seed of one round.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FIXTURE_DIR: Path = (
    Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "fixtures"
)


@dataclass(frozen=True)
class Fixture:
    round_id: str
    seed_index: int
    initial_grid: list[list[int]]
    ground_truth: NDArray[np.float64]


def load_fixture(path: Path) -> Fixture:
    """Load a single fixture from a JSON file.

    Args:
        path: Path to the fixture JSON file.

    Returns:
        Parsed Fixture with initial_grid and ground_truth.
    """
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"Expected dict in {path}, got {type(raw)}"
        raise TypeError(msg)

    gt_array: NDArray[np.float64] = np.array(raw["ground_truth"], dtype=np.float64)
    initial_grid = raw["initial_grid"]
    if not isinstance(initial_grid, list):
        msg = f"initial_grid must be a list in {path}"
        raise TypeError(msg)

    # Parse round_id and seed from filename or data
    round_id = str(raw.get("round_id", path.stem))
    seed_index = int(str(raw.get("seed_index", 0)))

    return Fixture(
        round_id=round_id,
        seed_index=seed_index,
        initial_grid=initial_grid,
        ground_truth=gt_array,
    )


def load_all_fixtures(fixture_dir: Path = FIXTURE_DIR) -> list[Fixture]:
    """Load all standard fixtures (excluding _prediction and _initial files).

    Args:
        fixture_dir: Directory containing fixture JSON files.

    Returns:
        List of all loaded fixtures, sorted by round_id then seed_index.
    """
    if not fixture_dir.exists():
        msg = f"Fixture directory not found: {fixture_dir}"
        raise FileNotFoundError(msg)

    fixtures: list[Fixture] = []
    for path in sorted(fixture_dir.glob("*_seed*.json")):
        # Skip non-standard fixture files
        if "_prediction" in path.stem or "_initial" in path.stem:
            continue
        fixtures.append(load_fixture(path))

    return fixtures


def group_by_round(fixtures: list[Fixture]) -> dict[str, list[Fixture]]:
    """Group fixtures by round_id.

    Returns:
        Dict mapping round_id to list of fixtures for that round.
    """
    groups: dict[str, list[Fixture]] = {}
    for f in fixtures:
        groups.setdefault(f.round_id, []).append(f)
    return groups
