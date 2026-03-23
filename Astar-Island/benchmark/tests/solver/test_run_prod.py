from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from astar_twin.solver.eval import run_prod


class _StubResult:
    def __init__(self) -> None:
        self.tensors = []
        self.total_queries_used = 7
        self.runtime_seconds = 1.5
        self.final_ess = 3.2
        self.contradiction_triggered = False


class _StubAdapter:
    def __init__(self) -> None:
        self.closed = False
        self.submissions: list[tuple[str, int]] = []
        self.round_status = "active"
        self.budget: tuple[int, int] = (0, 50)

    def close(self) -> None:
        self.closed = True

    def get_round_detail(self, round_id: str) -> object:
        del round_id

        class _Detail:
            status = self.round_status

        return _Detail()

    def get_budget(self, round_id: str) -> tuple[int, int]:
        del round_id
        return self.budget

    def submit(self, round_id: str, seed_index: int, prediction: object) -> object:
        del prediction
        self.submissions.append((round_id, seed_index))

        class _Response:
            def model_dump(self) -> dict[str, object]:
                return {
                    "status": "accepted",
                    "round_id": round_id,
                    "seed_index": seed_index,
                }

        return _Response()


def test_run_prod_refuses_submit_without_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="confirm-submit"):
        run_prod.main(["--round-id", "round-001", "--submit"])


def test_run_prod_writes_output_without_submitting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _StubAdapter()
    monkeypatch.setattr(run_prod.ProdAdapter, "from_environment", lambda **_: adapter)
    monkeypatch.setattr(run_prod, "solve", lambda *args, **kwargs: _StubResult())

    output_path = tmp_path / "out.json"
    run_prod.main(["--round-id", "round-001", "--output", str(output_path)])

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["round_id"] == "round-001"
    assert payload["variant"] == "particle"
    assert payload["submissions"] == []
    assert adapter.closed is True


def test_run_prod_submits_all_seed_tensors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _StubAdapter()
    result = _StubResult()
    result.tensors = [
        np.full((2, 2, 6), 1.0 / 6.0, dtype=np.float64),
        np.full((2, 2, 6), 1.0 / 6.0, dtype=np.float64),
    ]

    monkeypatch.setattr(run_prod.ProdAdapter, "from_environment", lambda **_: adapter)
    monkeypatch.setattr(run_prod, "solve_high_value_bidirectional", lambda *args, **kwargs: result)

    output_path = tmp_path / "submit.json"
    run_prod.main(
        [
            "--round-id",
            "round-001",
            "--variant",
            "high_value_bidirectional",
            "--submit",
            "--confirm-submit",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload["submissions"]) == 2
    assert adapter.submissions == [("round-001", 0), ("round-001", 1)]


def test_run_prod_blocks_partial_budget_without_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _StubAdapter()
    adapter.budget = (3, 50)
    monkeypatch.setattr(run_prod.ProdAdapter, "from_environment", lambda **_: adapter)
    with pytest.raises(RuntimeError, match="allow-resume"):
        run_prod.main(["--round-id", "round-001"])


def test_run_prod_rejects_inactive_round(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _StubAdapter()
    adapter.round_status = "completed"
    monkeypatch.setattr(run_prod.ProdAdapter, "from_environment", lambda **_: adapter)
    with pytest.raises(RuntimeError, match="not active"):
        run_prod.main(["--round-id", "round-001"])


def test_run_prod_allows_resume_when_explicit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _StubAdapter()
    adapter.budget = (3, 50)
    monkeypatch.setattr(run_prod.ProdAdapter, "from_environment", lambda **_: adapter)
    monkeypatch.setattr(run_prod, "solve", lambda *args, **kwargs: _StubResult())
    output_path = tmp_path / "resume.json"
    run_prod.main(
        ["--round-id", "round-001", "--allow-resume", "--output", str(output_path)]
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["queries_used"] == 7


def test_run_prod_submission_applies_safe_prediction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _StubAdapter()
    result = _StubResult()
    unsafe = np.zeros((2, 2, 6), dtype=np.float64)
    unsafe[:, :, 0] = 1.0
    result.tensors = [unsafe]

    monkeypatch.setattr(run_prod.ProdAdapter, "from_environment", lambda **_: adapter)
    monkeypatch.setattr(run_prod, "solve", lambda *args, **kwargs: result)

    output_path = tmp_path / "safe-submit.json"
    run_prod.main(
        [
            "--round-id",
            "round-001",
            "--submit",
            "--confirm-submit",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload["submissions"]) == 1
