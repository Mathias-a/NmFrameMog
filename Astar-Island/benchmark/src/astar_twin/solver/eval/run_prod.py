from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from astar_twin.scoring import safe_prediction
from astar_twin.solver.adapters.prod import ProdAdapter, ProdAdapterError
from astar_twin.solver.high_value_bidirectional import solve_high_value_bidirectional
from astar_twin.solver.pipeline import solve


def _validate_round_is_active(adapter: ProdAdapter, round_id: str, allow_resume: bool) -> None:
    detail = adapter.get_round_detail(round_id)
    if detail.status != "active":
        raise RuntimeError(f"Round {round_id} is not active; status={detail.status}")

    queries_used, queries_max = adapter.get_budget(round_id)
    if queries_used >= queries_max:
        raise RuntimeError(
            f"Round {round_id} budget already exhausted: {queries_used}/{queries_max}"
        )
    if queries_used > 0 and not allow_resume:
        raise RuntimeError(
            f"Round {round_id} already used {queries_used}/{queries_max} queries. "
            "Pass --allow-resume to continue on a partial budget."
        )


def _validated_submission_tensors(result_tensors: list[np.ndarray]) -> list[np.ndarray]:
    validated: list[np.ndarray] = []
    for tensor in result_tensors:
        safe_tensor = safe_prediction(tensor)
        if safe_tensor.ndim != 3 or safe_tensor.shape[2] != 6:
            raise RuntimeError(f"Invalid prediction shape: {safe_tensor.shape}")
        if not np.all(np.isfinite(safe_tensor)):
            raise RuntimeError("Prediction tensor contains non-finite values")
        validated.append(safe_tensor)
    return validated


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Astar solver against production API")
    parser.add_argument("--round-id", required=True)
    parser.add_argument(
        "--variant",
        choices=("particle", "high_value_bidirectional"),
        default="particle",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--particles", type=int, default=24)
    parser.add_argument("--inner-runs", type=int, default=6)
    parser.add_argument("--sims-per-seed", type=int, default=64)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--allow-resume",
        action="store_true",
        help="Allow a run when the active round already consumed some query budget.",
    )
    parser.add_argument(
        "--confirm-submit",
        action="store_true",
        help="Required alongside --submit before any live submission occurs.",
    )
    args = parser.parse_args(argv)

    if args.submit and not args.confirm_submit:
        raise RuntimeError("Refusing live submission without --confirm-submit")

    adapter = ProdAdapter.from_environment(base_url=args.base_url, submit_enabled=args.submit)
    try:
        _validate_round_is_active(adapter, args.round_id, allow_resume=args.allow_resume)
        solve_fn = solve if args.variant == "particle" else solve_high_value_bidirectional
        result = solve_fn(
            adapter,
            args.round_id,
            n_particles=args.particles,
            n_inner_runs=args.inner_runs,
            sims_per_seed=args.sims_per_seed,
            base_seed=args.base_seed,
        )

        submissions: list[dict[str, object]] = []
        if args.submit:
            for seed_index, tensor in enumerate(_validated_submission_tensors(result.tensors)):
                response = adapter.submit(args.round_id, seed_index, tensor)
                submissions.append(response.model_dump())

        payload = {
            "round_id": args.round_id,
            "variant": args.variant,
            "queries_used": result.total_queries_used,
            "runtime_seconds": result.runtime_seconds,
            "final_ess": result.final_ess,
            "contradiction_triggered": result.contradiction_triggered,
            "submissions": submissions,
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
    except ProdAdapterError as exc:
        raise RuntimeError(f"Production adapter failure: {exc}") from exc
    finally:
        adapter.close()
