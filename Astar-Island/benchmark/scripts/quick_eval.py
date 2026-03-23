#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARK_DIR.parent.parent

sys.path.insert(0, str(BENCHMARK_DIR / "src"))

from astar_twin.contracts.api_models import AnalysisResponse, RoundSummary  # noqa: E402

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


def print_grid_corner(grid: list[list[int]], title: str, size: int = 5) -> None:
    print(f"  {title} (top-left {size}x{size}):")
    for y in range(min(size, len(grid))):
        row = grid[y][:size]
        print("    " + " ".join(f"{cell:2}" for cell in row))


def get_argmax_grid(prob_grid: list[list[list[float]]]) -> list[list[int]]:
    h = len(prob_grid)
    w = len(prob_grid[0])
    res = []
    for y in range(h):
        row = []
        for x in range(w):
            probs = prob_grid[y][x]
            max_idx = probs.index(max(probs))
            row.append(max_idx)
        res.append(row)
    return res


def main() -> None:
    token = _load_token()
    headers = {"Authorization": f"Bearer {token}"}

    with httpx.Client(timeout=60.0) as client:
        print("Fetching rounds...")
        resp = client.get(f"{BASE_URL}/astar-island/rounds")
        resp.raise_for_status()
        rounds = [RoundSummary.model_validate(r) for r in resp.json()]

        round_22 = next((r for r in rounds if r.round_number == 22), None)
        if not round_22:
            print("Round 22 not found.")
            return

        print(f"Found Round 22: {round_22.id}")

        resp = client.get(f"{BASE_URL}/astar-island/rounds/{round_22.id}")
        resp.raise_for_status()
        round_detail = resp.json()
        seeds_count = int(round_detail["seeds_count"])

        print("\n--- Round 22 Evaluation ---")
        print("1. Queries submitted: (Not exposed by API)")

        for seed_idx in range(seeds_count):
            print(f"\n=== Seed {seed_idx} ===")
            resp = client.get(
                f"{BASE_URL}/astar-island/analysis/{round_22.id}/{seed_idx}", headers=headers
            )
            if resp.status_code in (403, 404):
                print(f"Analysis not available for seed {seed_idx} (HTTP {resp.status_code})")
                continue
            resp.raise_for_status()
            analysis = AnalysisResponse.model_validate(resp.json())

            print(f"3. Score: {analysis.score}")

            print("2. Initial state vs Ground Truth:")
            if analysis.initial_grid:
                print_grid_corner(analysis.initial_grid, "Initial Grid")
            else:
                print("  Initial Grid: Not provided in analysis")

            gt_argmax = get_argmax_grid(analysis.ground_truth)
            print_grid_corner(gt_argmax, "Ground Truth (argmax)")

            if analysis.prediction:
                pred_argmax = get_argmax_grid(analysis.prediction)
                print_grid_corner(pred_argmax, "Our Prediction (argmax)")
            else:
                print("  Prediction: None")


if __name__ == "__main__":
    main()
