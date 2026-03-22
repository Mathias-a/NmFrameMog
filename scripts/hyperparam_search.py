"""Hyperparameter grid search using LOROCV.

Searches over (max_depth, learning_rate, max_iter, min_samples_leaf, l2_reg)
and writes results to results/hyperparam_search.csv sorted by mean score.

Usage:
  uv run python scripts/hyperparam_search.py
"""

from __future__ import annotations

import csv
import itertools
import time
from pathlib import Path

from astar_island.evaluation import run_lorocv
from astar_island.fixtures import load_all_fixtures
from astar_island.models import GBTHyperparams

# Focused grid — total combos kept manageable (~48 combos)
# We vary the most impactful params with sensible ranges
GRID = {
    "max_depth": [4, 5, 6, 8],
    "learning_rate": [0.03, 0.05, 0.08],
    "max_iter": [200, 400],
    "min_samples_leaf": [20, 30],
    "l2_regularization": [0.5, 1.0],
}

RESULTS_DIR = Path("results")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "hyperparam_search.csv"

    fixtures = load_all_fixtures()
    print(f"Loaded {len(fixtures)} fixtures")

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    total = len(combos)
    print(f"Total combos: {total}\n")

    results: list[dict[str, float | str]] = []
    best_score = -1.0
    best_params = ""

    for i, values in enumerate(combos, 1):
        param_dict = dict(zip(keys, values))
        params = GBTHyperparams(**param_dict)  # type: ignore[arg-type]

        desc = " ".join(f"{k}={v}" for k, v in param_dict.items())
        print(f"[{i}/{total}] {desc} ... ", end="", flush=True)

        t0 = time.time()
        result = run_lorocv(fixtures, params=params, verbose=False)
        elapsed = time.time() - t0

        row = {
            **param_dict,
            "mean_score": round(result.mean_score, 4),
            "std_score": round(result.std_score, 4),
            "min_score": round(result.min_score, 4),
            "max_score": round(result.max_score, 4),
            "elapsed_s": round(elapsed, 1),
        }
        results.append(row)

        marker = ""
        if result.mean_score > best_score:
            best_score = result.mean_score
            best_params = desc
            marker = " *** NEW BEST ***"

        print(
            f"mean={result.mean_score:.2f} std={result.std_score:.2f} "
            f"range=[{result.min_score:.2f},{result.max_score:.2f}] "
            f"({elapsed:.0f}s){marker}"
        )

    # Sort by mean_score descending and write CSV
    results.sort(key=lambda r: -r["mean_score"])  # type: ignore[arg-type]

    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {output_path}")
    print(f"\nBest: {best_params}")
    print(f"  mean={best_score:.4f}")

    # Print top 10
    print("\n--- Top 10 ---")
    for r in results[:10]:
        params_str = " ".join(f"{k}={r[k]}" for k in keys)
        print(
            f"  {params_str} → "
            f"mean={r['mean_score']:.2f} std={r['std_score']:.2f} "
            f"range=[{r['min_score']:.2f},{r['max_score']:.2f}]"
        )


if __name__ == "__main__":
    main()
