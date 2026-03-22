"""Benchmark tests predicting live round performance for the astar_island solver.

test_benchmark_prior_only: prior model alone, no queries
test_benchmark_full_pipeline: full pipeline with mocked simulate_query (slow)
"""

import pytest

from astar_island.api import load_all_rounds
from astar_island.config import Config
from astar_island.offline import benchmark_prior_only, benchmark_scores


def _fast_config() -> Config:
    return Config(
        prior_model="lightgbm",
        lgb_n_estimators=10,
        lgb_num_leaves=8,
        lgb_max_depth=4,
        lgb_min_data_in_leaf=1,
    )


def test_benchmark_prior_only() -> None:
    rounds = load_all_rounds()
    if len(rounds) < 2:
        pytest.skip("Need at least 2 cached rounds")

    config = _fast_config()
    results = benchmark_prior_only(rounds=rounds, config=config)

    assert len(results) == len(rounds), "Should have one entry per round"

    all_scores = [s for scores in results.values() for s in scores]
    assert len(all_scores) > 0, "No scores returned"
    assert all(s >= 0 for s in all_scores), "Scores must be non-negative"

    overall_avg = sum(all_scores) / len(all_scores)
    assert overall_avg > 0, "Overall average score should be > 0"

    print("\n--- Prior-only benchmark ---")
    for rnum in sorted(results):
        seed_scores = results[rnum]
        avg = sum(seed_scores) / len(seed_scores) if seed_scores else 0.0
        per_seed = ", ".join(f"{s:.1f}" for s in seed_scores)
        print(f"  Round {rnum:3d}: avg={avg:.1f}  seeds=[{per_seed}]")
    print(f"  Overall average: {overall_avg:.1f}")


@pytest.mark.slow
def test_benchmark_full_pipeline() -> None:
    rounds = load_all_rounds()
    if len(rounds) < 2:
        pytest.skip("Need at least 2 cached rounds")

    config = _fast_config()
    results = benchmark_scores(rounds=rounds, config=config, seed=0)

    assert len(results) == len(rounds), "Should have one entry per round"

    all_scores = [s for scores in results.values() for s in scores]
    assert len(all_scores) > 0, "No scores returned"
    assert all(s >= 0 for s in all_scores), "Scores must be non-negative"

    overall_avg = sum(all_scores) / len(all_scores)
    assert overall_avg > 0, "Overall average score should be > 0"

    print("\n--- Full-pipeline benchmark ---")
    for rnum in sorted(results):
        seed_scores = results[rnum]
        avg = sum(seed_scores) / len(seed_scores) if seed_scores else 0.0
        per_seed = ", ".join(f"{s:.1f}" for s in seed_scores)
        print(f"  Round {rnum:3d}: avg={avg:.1f}  seeds=[{per_seed}]")
    print(f"  Overall average: {overall_avg:.1f}")
