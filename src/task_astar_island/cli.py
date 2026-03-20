from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Literal, cast

from task_astar_island.client import DEFAULT_BASE_URL, AstarIslandClient
from task_astar_island.models import (
    AuthConfig,
    ensure_json_object,
    normalize_json_value,
)
from task_astar_island.prediction import (
    build_probability_grid,
    build_submission_body,
    extract_budget_hint,
    infer_grid_dimensions,
    validate_probability_grid,
)


class PredictArgs(argparse.Namespace):
    command: Literal["predict"]
    width: int | None
    height: int | None
    round_file: str | None
    budget: float | None
    budget_file: str | None
    output: str | None


class NetworkArgs(argparse.Namespace):
    command: Literal["budget", "rounds", "solve", "submit"]
    token: str
    base_url: str
    auth_header: str
    auth_scheme: str
    output: str | None
    round_id: str
    seed_index: int
    submission_output: str | None
    body_file: str | None


def _load_json_file(path: Path) -> object:
    return normalize_json_value(json.loads(path.read_text(encoding="utf-8")))


def _write_json(data: object, output_path: Path | None) -> None:
    serialized = json.dumps(data, indent=2)
    if output_path is None:
        print(serialized)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8")


def _select_round_payload(
    rounds_payload: object, round_id: str | None
) -> dict[str, object]:
    if isinstance(rounds_payload, dict):
        if "rounds" in rounds_payload and isinstance(rounds_payload["rounds"], list):
            rounds_payload = rounds_payload["rounds"]
        else:
            return rounds_payload

    if not isinstance(rounds_payload, list) or not rounds_payload:
        raise ValueError("Rounds payload did not contain a selectable round.")

    if round_id is None:
        first_round = rounds_payload[0]
        return ensure_json_object(first_round)

    for round_payload in rounds_payload:
        round_object = ensure_json_object(round_payload)
        for key in ("id", "roundId"):
            if key in round_object and str(round_object[key]) == round_id:
                return round_object

    raise ValueError(f"Could not find round {round_id} in the rounds payload.")


def _build_auth(args: NetworkArgs) -> AuthConfig:
    return AuthConfig(
        token=args.token, header_name=args.auth_header, scheme=args.auth_scheme
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Astar Island task CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser(
        "predict", help="Build a local probability-grid prediction."
    )
    predict_parser.add_argument("--width", type=int)
    predict_parser.add_argument("--height", type=int)
    predict_parser.add_argument("--round-file")
    predict_parser.add_argument("--budget", type=float)
    predict_parser.add_argument("--budget-file")
    predict_parser.add_argument("--output")

    for name in ("budget", "rounds", "solve", "submit"):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("--token", required=True)
        command_parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
        command_parser.add_argument("--auth-header", default="Authorization")
        command_parser.add_argument("--auth-scheme", default="Bearer")
        command_parser.add_argument("--output")

    solve_parser = subparsers.choices["solve"]
    submit_parser = subparsers.choices["submit"]
    solve_parser.add_argument("--round-id", required=True)
    solve_parser.add_argument("--seed-index", type=int, required=True)
    solve_parser.add_argument("--submission-output")
    submit_parser.add_argument("--round-id", required=True)
    submit_parser.add_argument("--seed-index", type=int, required=True)
    submit_parser.add_argument("--body-file", required=True)

    return parser


async def _run_network_command(args: NetworkArgs) -> int:
    auth = _build_auth(args)
    output_path = Path(args.output) if args.output else None

    async with AstarIslandClient(auth, base_url=args.base_url) as client:
        if args.command == "budget":
            _write_json(await client.get_budget(), output_path)
            return 0
        if args.command == "rounds":
            _write_json(await client.get_rounds(), output_path)
            return 0
        if args.command == "submit":
            if args.body_file is None:
                raise ValueError("submit requires --body-file.")
            payload = ensure_json_object(_load_json_file(Path(args.body_file)))
            prediction_raw = payload.get("prediction")
            if not isinstance(prediction_raw, list):
                raise ValueError("Submission body must contain a prediction grid list.")
            prediction = cast(list[list[list[float]]], prediction_raw)
            validate_probability_grid(prediction)
            submission = build_submission_body(
                prediction,
                round_id=args.round_id,
                seed_index=args.seed_index,
            )
            _write_json(await client.submit(submission), output_path)
            return 0

        budget_payload = await client.get_budget()
        rounds_payload = await client.get_rounds()
        selected_round = _select_round_payload(rounds_payload, args.round_id)
        width, height = infer_grid_dimensions(selected_round)
        prediction = build_probability_grid(
            width, height, extract_budget_hint(budget_payload)
        )
        submission = build_submission_body(
            prediction,
            round_id=args.round_id,
            seed_index=args.seed_index,
        )

        if args.submission_output:
            _write_json(submission, Path(args.submission_output))

        response = await client.submit(submission)
        _write_json(response, output_path)
        return 0


def _run_predict_command(args: PredictArgs) -> int:
    budget: float | None = args.budget
    if args.budget_file:
        budget = extract_budget_hint(_load_json_file(Path(args.budget_file)))

    if args.round_file:
        round_payload = ensure_json_object(_load_json_file(Path(args.round_file)))
        width, height = infer_grid_dimensions(round_payload)
    else:
        if args.width is None or args.height is None:
            raise ValueError(
                "predict requires either --round-file or both --width and --height."
            )
        width = args.width
        height = args.height

    prediction = build_probability_grid(width, height, budget)
    output_path = Path(args.output) if args.output else None
    _write_json(prediction, output_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = parser.parse_args(argv)
    command = cast(str, raw_args.command)

    if command == "predict":
        return _run_predict_command(cast(PredictArgs, raw_args))
    return asyncio.run(_run_network_command(cast(NetworkArgs, raw_args)))


if __name__ == "__main__":
    raise SystemExit(main())
