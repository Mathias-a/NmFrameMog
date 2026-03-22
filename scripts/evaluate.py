"""Run LOROCV evaluation and print results.

Usage:
  uv run python scripts/evaluate.py
"""

from __future__ import annotations

from astar_island.evaluation import run_lorocv
from astar_island.fixtures import load_all_fixtures
from astar_island.terrain import PredictionClass


def main() -> None:
    fixtures = load_all_fixtures()
    print(f"Loaded {len(fixtures)} fixtures\n")

    result = run_lorocv(fixtures, verbose=True)

    print("\n--- Per-fold breakdown ---")
    for fold in sorted(result.folds, key=lambda f: f.mean_score):
        seed_scores = ", ".join(f"{s:.1f}" for s in fold.scores)
        print(
            f"  {fold.round_id[:8]}... "
            f"mean={fold.mean_score:5.1f}  T={fold.temperature:.3f}  "
            f"seeds=[{seed_scores}]"
        )

    print(f"\nOverall: {result.mean_score:.2f} +/- {result.std_score:.2f}")
    print(f"Range: [{result.min_score:.2f}, {result.max_score:.2f}]")


if __name__ == "__main__":
    main()
