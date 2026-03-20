from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from nmframemog.astar_island.solver.baseline import build_baseline_tensor
from nmframemog.astar_island.solver.pipeline import (
    parse_round_detail_payload,
    solve_round,
)
from nmframemog.astar_island.solver.cache import LocalCache
from nmframemog.astar_island.solver.validator import validate_prediction_tensor


class SolverTests(unittest.TestCase):
    def test_baseline_tensor_is_legal(self) -> None:
        grid = [
            [10, 10, 11, 0, 1],
            [10, 11, 11, 1, 2],
            [11, 11, 4, 3, 5],
            [0, 1, 4, 3, 5],
            [0, 0, 4, 11, 5],
        ]
        tensor = build_baseline_tensor(grid)
        validate_prediction_tensor(tensor, width=5, height=5)

    def test_parse_round_detail_accepts_embedded_grid_objects(self) -> None:
        payload = {
            "round_id": "round-1",
            "map_width": 5,
            "map_height": 5,
            "seeds_count": 2,
            "initial_states": [
                {"grid": [[0] * 5 for _ in range(5)]},
                {"initial_grid": [[1] * 5 for _ in range(5)]},
            ],
        }
        round_detail = parse_round_detail_payload(payload)
        self.assertEqual(round_detail.round_id, "round-1")
        self.assertEqual(len(round_detail.initial_states), 2)

    def test_solve_round_writes_predictions_and_debug_artifacts(self) -> None:
        payload = {
            "round_id": "round-2",
            "map_width": 5,
            "map_height": 5,
            "seeds_count": 5,
            "initial_states": [
                [
                    [0, 0, 11, 1, 2],
                    [0, 11, 11, 1, 2],
                    [11, 11, 4, 3, 5],
                    [0, 1, 4, 3, 5],
                    [0, 0, 4, 11, 5],
                ]
                for _ in range(5)
            ],
        }
        round_detail = parse_round_detail_payload(payload)
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = LocalCache(Path(temporary_directory))
            summary = solve_round(
                round_detail=round_detail,
                cache=cache,
                viewport_width=5,
                viewport_height=5,
                planned_queries_per_seed=1,
                rollout_count=8,
                random_seed=3,
                live_client=None,
                execute_live_queries=False,
                submit_predictions=False,
            )
            self.assertEqual(len(summary.seed_results), 5)
            first_prediction_path = Path(summary.seed_results[0].prediction_path)
            first_debug_dir = Path(summary.seed_results[0].debug_output_dir)
            self.assertTrue(first_prediction_path.exists())
            self.assertTrue((first_debug_dir / "index.html").exists())
            prediction_payload = json.loads(
                first_prediction_path.read_text(encoding="utf-8")
            )
            validate_prediction_tensor(
                prediction_payload["prediction"], width=5, height=5
            )


if __name__ == "__main__":
    unittest.main()
