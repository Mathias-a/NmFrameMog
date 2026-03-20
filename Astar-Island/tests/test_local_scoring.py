from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_cli_module = importlib.import_module("round_8_implementation.cli")
_emulator_module = importlib.import_module("round_8_implementation.solver.emulator")
_solver_module = importlib.import_module("round_8_implementation.solver")
_task_prediction_module = importlib.import_module("task_astar_island.prediction")

round_cli_main = _cli_module.main
DEFAULT_FIXTURE_PATHS = _emulator_module.DEFAULT_FIXTURE_PATHS
score_prediction_locally = _solver_module.score_prediction_locally
build_submission_body = _task_prediction_module.build_submission_body


def test_score_prediction_locally_accepts_submission_body() -> None:
    cell = [0.90, 0.02, 0.02, 0.02, 0.02, 0.02]
    prediction = [[cell[:] for _ in range(40)] for _ in range(40)]
    submission = build_submission_body(
        prediction,
        round_id="c5cdf100-a876-4fb7-b5d8-757162c97989",
        seed_index=0,
    )

    analysis = score_prediction_locally(
        prediction_payload=submission,
        fixture_paths=DEFAULT_FIXTURE_PATHS,
        analysis_rollout_count=16,
    )

    assert analysis["round_id"] == "c5cdf100-a876-4fb7-b5d8-757162c97989"
    assert analysis["seed_index"] == 0
    assert analysis["prediction"] == prediction
    score = analysis["score"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0


def test_score_local_cli_prints_analysis_payload(tmp_path: Path, capsys: Any) -> None:
    cell = [0.90, 0.02, 0.02, 0.02, 0.02, 0.02]
    prediction = [[cell[:] for _ in range(40)] for _ in range(40)]
    submission = build_submission_body(
        prediction,
        round_id="c5cdf100-a876-4fb7-b5d8-757162c97989",
        seed_index=1,
    )
    submission_path = tmp_path / "submission.json"
    submission_path.write_text(json.dumps(submission), encoding="utf-8")

    exit_code = round_cli_main(
        [
            "score-local",
            str(submission_path),
            "--fixture",
            str(DEFAULT_FIXTURE_PATHS[0]),
            "--fixture",
            str(DEFAULT_FIXTURE_PATHS[1]),
            "--analysis-rollouts",
            "16",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["round_id"] == "c5cdf100-a876-4fb7-b5d8-757162c97989"
    assert payload["seed_index"] == 1
    assert payload["score"] is not None
