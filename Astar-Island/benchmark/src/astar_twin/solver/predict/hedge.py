"""Conservative hedge blending for prediction safety.

Gated hedge: only activated when candidate underperforms relative to
fixed_coverage baseline, or when calibration disagreement is high.

Hedge formula: q_final = 0.85 * q_particle + 0.15 * q_fixed_coverage
Applied per-cell then passed through the finalizer.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.solver.predict.finalize import finalize_tensor

# Hedge blend weights
PARTICLE_WEIGHT = 0.85
COVERAGE_WEIGHT = 0.15

# Score margin gate: hedge activates if candidate_mean <= fc_mean + MARGIN
SCORE_MARGIN = 5.0

# Calibration disagreement gate: hedge activates if >=2 seeds exceed threshold
CALIBRATION_THRESHOLD = 0.15
MIN_DISAGREEMENT_SEEDS = 2


def should_hedge(
    candidate_mean_score: float,
    fixed_coverage_mean_score: float,
    per_seed_disagreements: list[float] | None = None,
) -> bool:
    """Determine if hedge should activate.

    Gate conditions (either triggers hedge):
      1. candidate_mean <= fixed_coverage_mean + SCORE_MARGIN
      2. per_seed_disagreements has >= MIN_DISAGREEMENT_SEEDS entries > CALIBRATION_THRESHOLD

    Args:
        candidate_mean_score: Mean score of the particle solver across seeds.
        fixed_coverage_mean_score: Mean score of the fixed_coverage baseline.
        per_seed_disagreements: Optional list of calibration disagreement scores per seed.

    Returns:
        True if hedge should be applied.
    """
    # Gate 1: score margin
    if candidate_mean_score <= fixed_coverage_mean_score + SCORE_MARGIN:
        return True

    # Gate 2: calibration disagreement
    if per_seed_disagreements is not None:
        high_disagreement = sum(1 for d in per_seed_disagreements if d > CALIBRATION_THRESHOLD)
        if high_disagreement >= MIN_DISAGREEMENT_SEEDS:
            return True

    return False


def apply_hedge(
    particle_tensors: list[NDArray[np.float64]],
    coverage_tensors: list[NDArray[np.float64]],
    initial_states: list[InitialState],
    height: int,
    width: int,
) -> list[NDArray[np.float64]]:
    """Apply hedge blend and finalize.

    q_final = 0.85 * q_particle + 0.15 * q_fixed_coverage

    Both inputs should already be safe/normalized, but the result is
    re-finalized to guarantee safety.
    """
    hedged: list[NDArray[np.float64]] = []
    for i, (p_t, c_t) in enumerate(zip(particle_tensors, coverage_tensors)):
        blended = PARTICLE_WEIGHT * p_t + COVERAGE_WEIGHT * c_t
        initial_state = initial_states[i] if i < len(initial_states) else None
        finalized = finalize_tensor(blended, height, width, initial_state)
        hedged.append(finalized)
    return hedged
