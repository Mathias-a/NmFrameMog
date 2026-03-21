from __future__ import annotations

import importlib
import json
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_emulator_module = importlib.import_module("round_8_implementation.solver.emulator")
_server_module = importlib.import_module("round_8_implementation.solver.server")
_validator_module = importlib.import_module("round_8_implementation.solver.validator")
AstarIslandEmulator = _emulator_module.AstarIslandEmulator
AstarIslandHTTPServer = _server_module.AstarIslandHTTPServer
entropy_weighted_kl_score = _validator_module.entropy_weighted_kl_score


def _request_json(
    url: str, *, method: str = "GET", payload: object | None = None
) -> object:
    data: bytes | None = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url=url, method=method, headers=headers, data=data)
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AssertionError(f"Expected mapping payload, got {type(value).__name__}")
    return cast(dict[str, object], value)


def _as_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise AssertionError(f"Expected list payload, got {type(value).__name__}")
    return cast(list[object], value)


@contextmanager
def _running_server() -> Iterator[str]:
    emulator = AstarIslandEmulator.from_fixture_paths()
    server = AstarIslandHTTPServer(("127.0.0.1", 0), emulator)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        socket_name = server.socket.getsockname()
        host = str(socket_name[0])
        port = int(socket_name[1])
        yield f"http://{host}:{port}/astar-island"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_rounds_and_round_detail_are_served() -> None:
    with _running_server() as base_url:
        rounds = _as_list(_request_json(f"{base_url}/rounds"))
        assert len(rounds) == 2
        active_round = _as_mapping(rounds[0])
        assert active_round["status"] == "active"
        round_id = cast(str, active_round["id"])

        detail = _as_mapping(_request_json(f"{base_url}/rounds/{round_id}"))
        assert detail["id"] == round_id
        assert detail["map_width"] == 40
        assert detail["map_height"] == 40
        assert detail["seeds_count"] == 5
        initial_states = _as_list(detail["initial_states"])
        assert len(initial_states) == 5
        first_state = _as_mapping(initial_states[0])
        settlements = _as_list(first_state["settlements"])
        first_settlement = _as_mapping(settlements[0])
        assert set(first_settlement) == {
            "x",
            "y",
            "has_port",
            "alive",
        }


def test_simulate_clamps_viewport_and_preserves_static_terrain() -> None:
    with _running_server() as base_url:
        result = _as_mapping(
            _request_json(
                f"{base_url}/simulate",
                method="POST",
                payload={
                    "seed_index": 0,
                    "viewport_x": 39,
                    "viewport_y": 39,
                    "viewport_w": 5,
                    "viewport_h": 5,
                },
            )
        )
        viewport = _as_mapping(result["viewport"])
        grid = _as_list(result["grid"])
        assert viewport["x"] == 35
        assert viewport["y"] == 35
        assert viewport["w"] == 5
        assert viewport["width"] == 5
        assert len(grid) == 5
        typed_grid = [_as_list(row) for row in grid]
        assert all(len(row) == 5 for row in typed_grid)
        assert typed_grid[-1][-1] == 10
        assert result["queries_used"] == 1
        assert result["queries_max"] == 50


def test_submit_accepts_valid_tensor_and_overwrites_same_seed() -> None:
    with _running_server() as base_url:
        rounds = _as_list(_request_json(f"{base_url}/rounds"))
        active_round = cast(str, _as_mapping(rounds[0])["id"])
        cell = [0.90, 0.02, 0.02, 0.02, 0.02, 0.02]
        prediction = [[cell[:] for _ in range(40)] for _ in range(40)]

        first = _as_mapping(
            _request_json(
                f"{base_url}/submit",
                method="POST",
                payload={
                    "round_id": active_round,
                    "seed_index": 0,
                    "prediction": prediction,
                },
            )
        )
        second = _as_mapping(
            _request_json(
                f"{base_url}/submit",
                method="POST",
                payload={
                    "round_id": active_round,
                    "seed_index": 0,
                    "prediction": prediction,
                },
            )
        )

        assert first["status"] == "accepted"
        assert second["status"] == "accepted"
        assert second["seed_index"] == 0


def test_submit_rejects_invalid_tensor() -> None:
    with _running_server() as base_url:
        rounds = _as_list(_request_json(f"{base_url}/rounds"))
        active_round = cast(str, _as_mapping(rounds[0])["id"])
        bad_prediction = [
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(40)] for _ in range(40)
        ]
        request = Request(
            url=f"{base_url}/submit",
            method="POST",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            data=json.dumps(
                {
                    "round_id": active_round,
                    "seed_index": 0,
                    "prediction": bad_prediction,
                }
            ).encode("utf-8"),
        )

        try:
            urlopen(request)
        except HTTPError as error:
            assert error.code == 400
            body = json.loads(error.read().decode("utf-8"))
            assert "floor violation" in body["error"].lower()
        else:
            raise AssertionError("Expected invalid submission to fail.")


def test_budget_exhaustion_returns_429() -> None:
    with _running_server() as base_url:
        payload = {
            "seed_index": 0,
            "viewport_x": 0,
            "viewport_y": 0,
            "viewport_w": 5,
            "viewport_h": 5,
        }
        for _ in range(50):
            _request_json(f"{base_url}/simulate", method="POST", payload=payload)

        request = Request(
            url=f"{base_url}/simulate",
            method="POST",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8"),
        )
        try:
            urlopen(request)
        except HTTPError as error:
            assert error.code == 429
            body = json.loads(error.read().decode("utf-8"))
            assert "budget exhausted" in body["error"].lower()
        else:
            raise AssertionError("Expected budget exhaustion to fail.")


def test_analysis_returns_ground_truth_and_score_for_submission() -> None:
    with _running_server() as base_url:
        rounds = _as_list(_request_json(f"{base_url}/rounds"))
        active_round = cast(str, _as_mapping(rounds[0])["id"])
        cell = [0.90, 0.02, 0.02, 0.02, 0.02, 0.02]
        prediction = [[cell[:] for _ in range(40)] for _ in range(40)]
        _request_json(
            f"{base_url}/submit",
            method="POST",
            payload={
                "round_id": active_round,
                "seed_index": 0,
                "prediction": prediction,
            },
        )

        analysis = _as_mapping(_request_json(f"{base_url}/analysis/{active_round}/0"))
        assert analysis["width"] == 40
        assert analysis["height"] == 40
        assert analysis["prediction"] == prediction

        ground_truth = cast(list[list[list[float]]], analysis["ground_truth"])
        assert len(ground_truth) == 40
        assert all(len(row) == 40 for row in ground_truth)
        for row in ground_truth:
            for cell_distribution in row:
                assert len(cell_distribution) == 6
                assert abs(sum(cell_distribution) - 1.0) < 1e-9
                assert all(probability >= 0.0 for probability in cell_distribution)

        score = cast(float, analysis["score"])
        assert 0.0 <= score <= 100.0
        assert score == entropy_weighted_kl_score(prediction, ground_truth)


def test_analysis_without_submission_reports_null_score() -> None:
    with _running_server() as base_url:
        rounds = _as_list(_request_json(f"{base_url}/rounds"))
        active_round = cast(str, _as_mapping(rounds[0])["id"])

        analysis = _as_mapping(_request_json(f"{base_url}/analysis/{active_round}/1"))
        assert analysis["prediction"] is None
        assert analysis["score"] is None
        ground_truth = cast(list[list[list[float]]], analysis["ground_truth"])
        assert len(ground_truth) == 40
        assert len(ground_truth[0]) == 40
