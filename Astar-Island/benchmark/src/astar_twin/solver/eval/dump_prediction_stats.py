"""Dump prediction statistics for a solver run.

CLI: uv run python -m astar_twin.solver.eval.dump_prediction_stats [fixture_path]
Library: from astar_twin.solver.eval.dump_prediction_stats import dump_stats
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.types import NUM_CLASSES


def dump_stats(
    tensors: list[NDArray[np.float64]],
    height: int,
    width: int,
) -> dict:
    stats: dict = {"seeds": [], "aggregate": {}}
    all_mins: list[float] = []
    all_maxs: list[float] = []
    all_entropies: list[float] = []

    for i, tensor in enumerate(tensors):
        assert tensor.shape == (height, width, NUM_CLASSES), (
            f"Seed {i}: shape {tensor.shape} != ({height}, {width}, {NUM_CLASSES})"
        )
        seed_min = float(tensor.min())
        seed_max = float(tensor.max())
        sums = tensor.sum(axis=2)
        sum_ok = bool(np.allclose(sums, 1.0, atol=1e-6))
        probabilities = np.clip(tensor, 1e-15, 1.0)
        entropy = -np.sum(probabilities * np.log(probabilities), axis=2)
        mean_entropy = float(entropy.mean())

        stats["seeds"].append(
            {
                "seed_index": i,
                "min_prob": seed_min,
                "max_prob": seed_max,
                "sum_check_passed": sum_ok,
                "mean_entropy": mean_entropy,
                "shape": list(tensor.shape),
            }
        )
        all_mins.append(seed_min)
        all_maxs.append(seed_max)
        all_entropies.append(mean_entropy)

    stats["aggregate"] = {
        "n_seeds": len(tensors),
        "global_min_prob": min(all_mins) if all_mins else None,
        "global_max_prob": max(all_maxs) if all_maxs else None,
        "mean_entropy": float(np.mean(all_entropies)) if all_entropies else None,
    }
    return stats


def main() -> None:
    from astar_twin.data.loaders import load_fixture
    from astar_twin.solver.baselines import uniform_baseline

    fixture_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/rounds/test-round-001")
    fixture = load_fixture(fixture_dir)

    height = fixture.map_height
    width = fixture.map_width
    n_seeds = len(fixture.initial_states)

    tensors = [uniform_baseline(height, width) for _ in range(n_seeds)]
    stats = dump_stats(tensors, height, width)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
