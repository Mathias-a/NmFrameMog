from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from astar_twin.contracts.types import NUM_CLASSES
from astar_twin.solver.baselines import uniform_baseline
from astar_twin.solver.eval.dump_prediction_stats import dump_stats, main


def test_dump_stats_valid_tensors() -> None:
    tensors = [uniform_baseline(10, 10) for _ in range(5)]

    stats = dump_stats(tensors, 10, 10)

    assert set(stats) == {"seeds", "aggregate"}
    assert stats["aggregate"]["n_seeds"] == 5
    assert len(stats["seeds"]) == 5
    for seed_stats in stats["seeds"]:
        assert set(seed_stats) == {
            "seed_index",
            "min_prob",
            "max_prob",
            "sum_check_passed",
            "mean_entropy",
            "shape",
        }
        assert seed_stats["sum_check_passed"] is True
        assert seed_stats["shape"] == [10, 10, NUM_CLASSES]


def test_dump_stats_wrong_shape_raises() -> None:
    wrong_shape = np.full((10, 10, NUM_CLASSES - 1), 1.0 / (NUM_CLASSES - 1), dtype=np.float64)

    with pytest.raises(AssertionError, match=r"Seed 0: shape"):
        dump_stats([wrong_shape], 10, 10)


def test_cli_runs_successfully(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dump_prediction_stats",
            "data/rounds/test-round-001",
        ],
    )

    main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert set(payload) == {"seeds", "aggregate"}
    assert payload["aggregate"]["n_seeds"] == 5
    assert len(payload["seeds"]) == 5
    assert payload["seeds"][0]["shape"] == [10, 10, NUM_CLASSES]
