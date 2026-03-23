#!/usr/bin/env python3
"""Check the active Astar Island round, budget, and submitted predictions.
Evaluates the submitted predictions against a local strategy's predictions.
If no active round is found, checks the most recently completed or scoring round.
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARK_DIR.parent.parent

sys.path.insert(0, str(BENCHMARK_DIR / "src"))

import httpx  # noqa: E402
import numpy as np  # noqa: E402

from astar_twin.contracts.api_models import InitialSettlement, InitialState  # noqa: E402
from astar_twin.data.models import ParamsSource, RoundFixture  # noqa: E402
from astar_twin.solver.adapters.benchmark import BenchmarkAdapter  # noqa: E402
from astar_twin.solver.pipeline import solve  # noqa: E402
from astar_twin.scoring import compute_score, safe_prediction  # noqa: E402

BASE_URL = "https://api.ainm.no"


def _load_token() -> str:
    token = os.environ.get("ACCESS_TOKEN", "").strip()
    if token:
        return token

    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("access_token="):
                token = line.split("=", 1)[1].strip()
                if token:
                    return token

    raise RuntimeError("No API token found.")


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def main() -> None:
    try:
        token = _load_token()
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    with httpx.Client(timeout=30.0) as client:
        print("Fetching rounds...")
        resp = client.get(f"{BASE_URL}/astar-island/rounds")
        resp.raise_for_status()
        rounds = resp.json()

        active_rounds = [r for r in rounds if r.get("status") == "active"]
        
        target_round = None
        is_active = False
        
        if active_rounds:
            target_round = active_rounds[0]
            is_active = True
        else:
            print("No active rounds found. Looking for most recently completed or scoring round...")
            completed_rounds = [r for r in rounds if r.get("status") in ("completed", "scoring")]
            if not completed_rounds:
                print("No completed or scoring rounds found either.")
                return
            
            completed_rounds.sort(key=lambda x: x.get("round_number", 0), reverse=True)
            target_round = completed_rounds[0]

        round_id = target_round["id"]
        status = target_round["status"]
        print(f"\nTarget Round: {target_round['round_number']} (ID: {round_id}, Status: {status})")
        print(f"Map Size: {target_round['map_width']}x{target_round['map_height']}")
        if is_active:
            print(f"Closes At: {target_round.get('closes_at')}")

        print("\nFetching round details (initial states)...")
        resp = client.get(f"{BASE_URL}/astar-island/rounds/{round_id}")
        resp.raise_for_status()
        round_detail = resp.json()

        initial_states = []
        for s in round_detail["initial_states"]:
            initial_states.append(
                InitialState(
                    grid=s["grid"], settlements=[InitialSettlement(**st) for st in s["settlements"]]
                )
            )

        if is_active:
            print("\nFetching budget...")
            resp = client.get(f"{BASE_URL}/astar-island/budget", headers=_auth_headers(token))
            if resp.status_code == 200:
                budget = resp.json()
                print(f"Queries Used: {budget.get('queries_used')} / {budget.get('queries_max')}")
            else:
                print(f"Failed to fetch budget: {resp.status_code} {resp.text}")

        submitted_predictions = {}
        ground_truths = {}
        submitted_scores = {}
        
        if is_active:
            print("\nFetching submitted predictions...")
            url = f"{BASE_URL}/astar-island/my-predictions/{round_id}"
            resp = client.get(url, headers=_auth_headers(token))

            if resp.status_code == 200:
                data = resp.json()
                for item in data:
                    seed_idx = item["seed_index"]
                    argmax_grid = item.get("argmax_grid")
                    if argmax_grid:
                        submitted_predictions[seed_idx] = np.array(argmax_grid, dtype=np.int32)
                        shape_str = submitted_predictions[seed_idx].shape
                        print(f"  Seed {seed_idx}: Prediction submitted (shape: {shape_str})")
                    else:
                        print(f"  Seed {seed_idx}: No argmax_grid found in submission")
            else:
                print(f"Failed to fetch my-predictions: {resp.status_code} {resp.text}")
        else:
            print("\nFetching analysis for completed/scoring round...")
            for seed_idx in range(len(initial_states)):
                url = f"{BASE_URL}/astar-island/analysis/{round_id}/{seed_idx}"
                resp = client.get(url, headers=_auth_headers(token))
                if resp.status_code == 200:
                    data = resp.json()
                    if "prediction" in data and data["prediction"]:
                        pred_tensor = np.array(data["prediction"], dtype=np.float64)
                        submitted_predictions[seed_idx] = np.argmax(pred_tensor, axis=-1)
                        print(f"  Seed {seed_idx}: Prediction fetched")
                    if "ground_truth" in data and data["ground_truth"]:
                        ground_truths[seed_idx] = np.array(data["ground_truth"], dtype=np.float64)
                        print(f"  Seed {seed_idx}: Ground truth fetched")
                    if "score" in data and data["score"] is not None:
                        submitted_scores[seed_idx] = data["score"]
                        print(f"  Seed {seed_idx}: Score = {data['score']:.4f}")
                else:
                    print(f"  Seed {seed_idx}: Failed to fetch analysis: {resp.status_code}")

        if not submitted_predictions:
            print("\nNo submitted predictions to compare against.")
            return

        print("\nEvaluating submitted predictions against local 'calibrated_solver' strategy...")

        # Create a dummy fixture for the adapter
        fixture = RoundFixture(
            id=round_id,
            round_number=target_round["round_number"],
            status=target_round["status"],
            map_width=target_round["map_width"],
            map_height=target_round["map_height"],
            seeds_count=len(initial_states),
            initial_states=initial_states,
            params_source=ParamsSource.DEFAULT_PRIOR,
        )

        adapter = BenchmarkAdapter(fixture, require_calibrated_params=False)

        # Run the solver locally
        print("  Running calibrated_solver locally (this will take a moment)...")
        # Using fast params for quick check
        result = solve(
            adapter,
            round_id,
            n_particles=4,
            n_inner_runs=2,
            sims_per_seed=8,
            base_seed=42,
        )

        print(
            f"  Local solver finished in {result.runtime_seconds:.1f}s "
            f"using {result.total_queries_used} queries."
        )

        for seed_idx, submitted_argmax in submitted_predictions.items():
            if seed_idx >= len(result.tensors):
                continue

            local_tensor = result.tensors[seed_idx]
            local_argmax = np.argmax(local_tensor, axis=-1)

            # Compare argmax grids
            matches = np.sum(local_argmax == submitted_argmax)
            total_cells = local_argmax.size
            disagreement_rate = 1.0 - (matches / total_cells)

            print(
                f"\n  Seed {seed_idx}: Disagreement rate vs prod submission: "
                f"{disagreement_rate:.2%} ({total_cells - matches}/{total_cells} cells differ)"
            )
            
            if seed_idx in ground_truths:
                gt_tensor = ground_truths[seed_idx]
                safe_local_tensor = safe_prediction(local_tensor)
                local_score = compute_score(gt_tensor, safe_local_tensor)
                
                prod_score_str = f"{submitted_scores[seed_idx]:.4f}" if seed_idx in submitted_scores else "N/A"
                print(f"  Seed {seed_idx}: Local Score = {local_score:.4f} | Prod Score = {prod_score_str}")


if __name__ == "__main__":
    main()
