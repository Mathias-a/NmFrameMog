"""Ground-truth fixture storage for offline backtesting.

Saves and loads frozen ground-truth data as JSON files so that
backtesting does not require network access after initial fetch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from astar_island.backtesting.client import AstarClient

FIXTURE_DIR: Path = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "fixtures"
)


@dataclass(frozen=True, slots=True)
class GroundTruthFixture:
    """A frozen ground-truth snapshot for one seed of one round."""

    round_id: str
    seed_index: int
    ground_truth: NDArray[np.float64]
    initial_grid: list[list[int]]
    official_score: float


def fixture_path(round_id: str, seed_index: int) -> Path:
    """Return the canonical path for a fixture file."""
    return FIXTURE_DIR / f"{round_id}_seed{seed_index}.json"


def save_fixture(fixture: GroundTruthFixture) -> Path:
    """Serialize a fixture to JSON and write to disk.

    Returns the path of the written file.
    """
    path = fixture_path(fixture.round_id, fixture.seed_index)
    path.parent.mkdir(parents=True, exist_ok=True)

    gt_list: list[list[list[float]]] = fixture.ground_truth.tolist()
    payload: dict[
        str, str | int | float | list[list[list[float]]] | list[list[int]]
    ] = {
        "round_id": fixture.round_id,
        "seed_index": fixture.seed_index,
        "ground_truth": gt_list,
        "initial_grid": fixture.initial_grid,
        "official_score": fixture.official_score,
    }

    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_fixture(round_id: str, seed_index: int) -> GroundTruthFixture:
    """Load a fixture from disk.

    Raises:
        FileNotFoundError: If the fixture file does not exist.
    """
    path = fixture_path(round_id, seed_index)
    text = path.read_text(encoding="utf-8")
    raw: object = json.loads(text)
    if not isinstance(raw, dict):
        msg = f"Expected dict, got {type(raw)}"
        raise TypeError(msg)

    gt_array: NDArray[np.float64] = np.array(raw["ground_truth"], dtype=np.float64)
    initial_grid = raw["initial_grid"]
    if not isinstance(initial_grid, list):
        msg = "initial_grid must be a list"
        raise TypeError(msg)

    return GroundTruthFixture(
        round_id=str(raw["round_id"]),
        seed_index=int(str(raw["seed_index"])),
        ground_truth=gt_array,
        initial_grid=initial_grid,
        official_score=float(str(raw["official_score"])),
    )


def list_fixtures() -> list[tuple[str, int]]:
    """List all available fixtures as (round_id, seed_index) pairs."""
    if not FIXTURE_DIR.exists():
        return []

    results: list[tuple[str, int]] = []
    for path in sorted(FIXTURE_DIR.glob("*_seed*.json")):
        stem = path.stem
        # Format: {round_id}_seed{seed_index}
        seed_marker = "_seed"
        marker_pos = stem.rfind(seed_marker)
        if marker_pos == -1:
            continue
        round_id = stem[:marker_pos]
        try:
            seed_index = int(stem[marker_pos + len(seed_marker) :])
        except ValueError:
            continue
        results.append((round_id, seed_index))

    return results


def fetch_and_freeze_all(client: AstarClient) -> list[Path]:
    """Fetch all rounds and seeds from the API, save as fixtures.

    Returns the list of paths to saved fixture files.
    """
    saved_paths: list[Path] = []
    rounds = client.get_rounds()

    for round_data in rounds:
        round_info = client.get_round(round_data.round_id)

        for seed_idx in range(round_info.seeds_count):
            analysis = client.get_analysis(round_data.round_id, seed_idx)

            fixture = GroundTruthFixture(
                round_id=analysis.round_id,
                seed_index=analysis.seed_index,
                ground_truth=analysis.ground_truth,
                initial_grid=analysis.initial_grid,
                official_score=analysis.official_score,
            )

            path = save_fixture(fixture)
            saved_paths.append(path)

    return saved_paths
