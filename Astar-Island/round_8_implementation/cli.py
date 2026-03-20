from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from .solver.cache import LocalCache
from .solver.debug_visualization import load_trace_file, render_debug_bundle
from .solver.emulator import AstarIslandEmulator
from .solver.pipeline import (
    build_live_client_from_environment,
    parse_round_detail_payload,
    round_detail_to_payload,
    solve_round,
)
from .solver.server import run_emulator_server
from .solver.validator import validate_prediction_tensor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Astar Island baseline solver utilities."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve_parser = subparsers.add_parser(
        "solve-round", help="Build a legal baseline prediction bundle for a round."
    )
    solve_parser.add_argument(
        "--round-detail-file",
        type=Path,
        help="Local round detail JSON to use instead of fetching live data.",
    )
    solve_parser.add_argument(
        "--round-id",
        help="Round ID to fetch from the live API when --round-detail-file is omitted.",
    )
    solve_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".artifacts/astar-island"),
        help=(
            "Root directory for cached inputs, predictions, debug artifacts, "
            "and run summaries."
        ),
    )
    solve_parser.add_argument(
        "--base-url",
        default="https://api.ainm.no/astar-island",
        help="Base URL for the Astar Island API.",
    )
    solve_parser.add_argument(
        "--token-env-var",
        default="AINM_ACCESS_TOKEN",
        help="Environment variable that stores the Bearer token for live API access.",
    )
    solve_parser.add_argument(
        "--viewport-width", type=int, default=15, help="Planned viewport width (5-15)."
    )
    solve_parser.add_argument(
        "--viewport-height",
        type=int,
        default=15,
        help="Planned viewport height (5-15).",
    )
    solve_parser.add_argument(
        "--planned-queries-per-seed",
        type=int,
        default=2,
        help="How many high-entropy viewports to plan per seed for the baseline run.",
    )
    solve_parser.add_argument(
        "--rollouts",
        type=int,
        default=64,
        help="Number of stochastic rollouts used to aggregate the final tensor.",
    )
    solve_parser.add_argument(
        "--random-seed", type=int, default=7, help="Base random seed for rollouts."
    )
    solve_parser.add_argument(
        "--execute-live-queries",
        action="store_true",
        help=(
            "Execute live viewport queries using the planned viewports and "
            "cache the responses."
        ),
    )
    solve_parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit each generated tensor after validation. Requires live API access.",
    )

    validate_parser = subparsers.add_parser(
        "validate-prediction", help="Validate a saved prediction JSON file."
    )
    validate_parser.add_argument("prediction_file", type=Path)

    score_parser = subparsers.add_parser(
        "score-local",
        help="Score a prediction locally against the fixture-backed proxy simulator.",
    )
    score_parser.add_argument("prediction_file", type=Path)
    score_parser.add_argument("--round-id")
    score_parser.add_argument("--seed-index", type=int)
    score_parser.add_argument(
        "--fixture",
        dest="fixtures",
        action="append",
        type=Path,
        default=None,
        help="Round fixture JSON to load. Repeat to load multiple rounds.",
    )
    score_parser.add_argument(
        "--random-seed",
        type=int,
        default=20260320,
        help="Base random seed for local proxy scoring.",
    )
    score_parser.add_argument(
        "--analysis-rollouts",
        type=int,
        default=128,
        help="Monte Carlo rollout count used to build local ground truth.",
    )

    analysis_parser = subparsers.add_parser(
        "fetch-analysis",
        help="Fetch and cache post-round analysis for one seed using the live API.",
    )
    analysis_parser.add_argument("--round-id", required=True)
    analysis_parser.add_argument("--seed-index", type=int, required=True)
    analysis_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".artifacts/astar-island"),
        help="Root directory for cached analysis payloads.",
    )
    analysis_parser.add_argument(
        "--base-url",
        default="https://api.ainm.no/astar-island",
        help="Base URL for the Astar Island API.",
    )
    analysis_parser.add_argument(
        "--token-env-var",
        default="AINM_ACCESS_TOKEN",
        help="Environment variable that stores the Bearer token for live API access.",
    )

    debug_parser = subparsers.add_parser(
        "render-debug", help="Render static debug artifacts from a trace JSON file."
    )
    debug_parser.add_argument("--input", type=Path, required=True)
    debug_parser.add_argument("--output-dir", type=Path, required=True)

    serve_parser = subparsers.add_parser(
        "serve-emulator",
        help="Serve a local Astar Island emulator backed by cached round fixtures.",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for the local emulator HTTP server.",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port for the local emulator HTTP server.",
    )
    serve_parser.add_argument(
        "--fixture",
        dest="fixtures",
        action="append",
        type=Path,
        default=None,
        help="Round fixture JSON to load. Repeat to load multiple rounds.",
    )
    serve_parser.add_argument(
        "--active-round-id",
        help="Round ID to expose as the in-memory active round.",
    )
    serve_parser.add_argument(
        "--random-seed",
        type=int,
        default=20260320,
        help="Base random seed for stochastic /simulate responses.",
    )

    args = parser.parse_args(argv)
    command = _read_optional_str_arg(args, "command")
    if command == "solve-round":
        return _solve_round(args)
    if command == "validate-prediction":
        return _validate_prediction(_read_path_arg(args, "prediction_file"))
    if command == "score-local":
        return _score_local(args)
    if command == "fetch-analysis":
        return _fetch_analysis(args)
    if command == "render-debug":
        artifacts = render_debug_bundle(
            load_trace_file(_read_path_arg(args, "input")),
            _read_path_arg(args, "output_dir"),
        )
        print(artifacts.index_html)
        return 0
    if command == "serve-emulator":
        return _serve_emulator(args)
    raise AssertionError(f"Unsupported command: {command}")


def _solve_round(args: argparse.Namespace) -> int:
    cache_dir = _read_path_arg(args, "cache_dir")
    token_env_var = _read_str_arg(args, "token_env_var")
    base_url = _read_str_arg(args, "base_url")
    cache = LocalCache(cache_dir)
    client = build_live_client_from_environment(
        token_env_var=token_env_var,
        base_url=base_url,
    )

    round_detail_file = _read_optional_path_arg(args, "round_detail_file")
    payload: object
    if round_detail_file is not None:
        payload = _load_json_file(round_detail_file)
    else:
        round_id = _read_optional_str_arg(args, "round_id")
        if round_id is None:
            raise ValueError(
                "Either --round-detail-file or --round-id must be provided."
            )
        if client is None:
            raise ValueError(
                "Live round fetching requires the "
                f"{token_env_var} environment variable."
            )
        payload = client.get_round_detail(round_id)

    round_detail = parse_round_detail_payload(payload)
    summary = solve_round(
        round_detail=round_detail,
        cache=cache,
        viewport_width=_read_int_arg(args, "viewport_width"),
        viewport_height=_read_int_arg(args, "viewport_height"),
        planned_queries_per_seed=_read_int_arg(args, "planned_queries_per_seed"),
        rollout_count=_read_int_arg(args, "rollouts"),
        random_seed=_read_int_arg(args, "random_seed"),
        live_client=client,
        execute_live_queries=_read_bool_arg(args, "execute_live_queries"),
        submit_predictions=_read_bool_arg(args, "submit"),
    )
    print(
        json.dumps(
            {
                "round": round_detail_to_payload(round_detail),
                "run_id": summary.run_id,
                "output_root": summary.output_root,
                "seed_count": len(summary.seed_results),
            },
            indent=2,
        )
    )
    return 0


def _validate_prediction(prediction_file: Path) -> int:
    payload = _load_json_file(prediction_file)
    if not isinstance(payload, dict):
        raise ValueError("Prediction file must contain a JSON object.")
    prediction_payload = payload.get("prediction")
    prediction = _parse_prediction_tensor(prediction_payload)
    width = len(prediction[0])
    height = len(prediction)
    validate_prediction_tensor(prediction, width=width, height=height)
    print(prediction_file)
    return 0


def _score_local(args: argparse.Namespace) -> int:
    local_scoring = importlib.import_module(
        "round_8_implementation.solver.local_scoring"
    )
    load_json_payload = local_scoring.load_json_payload
    score_prediction_locally = local_scoring.score_prediction_locally

    prediction_file = _read_path_arg(args, "prediction_file")
    payload = load_json_payload(prediction_file)
    analysis = score_prediction_locally(
        prediction_payload=payload,
        fixture_paths=_read_path_list_arg(args, "fixtures"),
        round_id=_read_optional_str_arg(args, "round_id"),
        seed_index=_read_optional_int_arg(args, "seed_index"),
        random_seed=_read_int_arg(args, "random_seed"),
        analysis_rollout_count=_read_int_arg(args, "analysis_rollouts"),
    )
    print(json.dumps(analysis, indent=2))
    return 0


def _fetch_analysis(args: argparse.Namespace) -> int:
    cache_dir = _read_path_arg(args, "cache_dir")
    token_env_var = _read_str_arg(args, "token_env_var")
    base_url = _read_str_arg(args, "base_url")
    round_id = _read_str_arg(args, "round_id")
    seed_index = _read_int_arg(args, "seed_index")
    cache = LocalCache(cache_dir)
    cache.ensure()
    client = build_live_client_from_environment(
        token_env_var=token_env_var,
        base_url=base_url,
    )
    if client is None:
        raise ValueError(
            f"Live analysis fetching requires the {token_env_var} environment variable."
        )
    payload = client.get_analysis(round_id=round_id, seed_index=seed_index)
    output_path = cache.analysis_path(round_id, seed_index)
    cache.save_json(output_path, payload)
    print(output_path)
    return 0


def _serve_emulator(args: argparse.Namespace) -> int:
    emulator = AstarIslandEmulator.from_fixture_paths(
        _read_path_list_arg(args, "fixtures"),
        active_round_id=_read_optional_str_arg(args, "active_round_id"),
        random_seed=_read_int_arg(args, "random_seed"),
    )
    return run_emulator_server(
        emulator=emulator,
        host=_read_str_arg(args, "host"),
        port=_read_int_arg(args, "port"),
    )


def _read_path_arg(args: argparse.Namespace, field_name: str) -> Path:
    value: object = getattr(args, field_name)
    if not isinstance(value, Path):
        raise ValueError(f"Argument '{field_name}' must resolve to a path.")
    return value


def _read_optional_path_arg(args: argparse.Namespace, field_name: str) -> Path | None:
    value: object = getattr(args, field_name)
    if value is None:
        return None
    if not isinstance(value, Path):
        raise ValueError(f"Argument '{field_name}' must resolve to a path.")
    return value


def _read_path_list_arg(args: argparse.Namespace, field_name: str) -> list[Path] | None:
    value: object = getattr(args, field_name)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"Argument '{field_name}' must resolve to a path list.")
    paths: list[Path] = []
    for item in value:
        if not isinstance(item, Path):
            raise ValueError(f"Argument '{field_name}' must contain only paths.")
        paths.append(item)
    return paths


def _read_str_arg(args: argparse.Namespace, field_name: str) -> str:
    value: object = getattr(args, field_name)
    if not isinstance(value, str):
        raise ValueError(f"Argument '{field_name}' must be a string.")
    return value


def _read_optional_str_arg(args: argparse.Namespace, field_name: str) -> str | None:
    value: object = getattr(args, field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Argument '{field_name}' must be a string when provided.")
    return value


def _read_int_arg(args: argparse.Namespace, field_name: str) -> int:
    value: object = getattr(args, field_name)
    if not isinstance(value, int):
        raise ValueError(f"Argument '{field_name}' must be an integer.")
    return value


def _read_optional_int_arg(args: argparse.Namespace, field_name: str) -> int | None:
    value: object = getattr(args, field_name)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"Argument '{field_name}' must be an integer when provided.")
    return value


def _read_bool_arg(args: argparse.Namespace, field_name: str) -> bool:
    value: object = getattr(args, field_name)
    if not isinstance(value, bool):
        raise ValueError(f"Argument '{field_name}' must be a boolean flag.")
    return value


def _load_json_file(path: Path) -> object:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not _is_json_value(payload):
        raise ValueError(f"File {path} does not contain JSON-compatible data.")
    return payload


def _parse_prediction_tensor(payload: object) -> list[list[list[float]]]:
    if not isinstance(payload, list):
        raise ValueError("Prediction payload must be a nested list.")
    tensor: list[list[list[float]]] = []
    for row in payload:
        if not isinstance(row, list):
            raise ValueError("Prediction rows must be lists.")
        parsed_row: list[list[float]] = []
        for cell in row:
            if not isinstance(cell, list):
                raise ValueError("Prediction cells must be lists.")
            parsed_cell: list[float] = []
            for probability in cell:
                if isinstance(probability, bool) or not isinstance(
                    probability, (int, float)
                ):
                    raise ValueError("Prediction probabilities must be numeric.")
                parsed_cell.append(float(probability))
            parsed_row.append(parsed_cell)
        tensor.append(parsed_row)
    return tensor


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item) for key, item in value.items()
        )
    return False
