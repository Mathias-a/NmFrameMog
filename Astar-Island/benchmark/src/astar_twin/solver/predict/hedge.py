"""Conservative hedge blending for prediction safety.

Two hedge modes are supported:

1. **Score-gated hedge** (original):  Blends particle predictions with
   ``fixed_coverage`` at a fixed 85/15 ratio when the particle candidate
   does not beat the baseline by a sufficient margin.

2. **Confidence-gated anchor hedge** (new):  Per-seed blending with the
   structural anchor tensor, where the blend weight is derived from the
   ``PosteriorConfidence`` score.  This mode is used by the hybrid solver.

Both modes finalize through ``finalize_tensor`` to guarantee safety.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import InitialState
from astar_twin.solver.inference.confidence import PosteriorConfidence
from astar_twin.solver.predict.finalize import finalize_tensor

# ── Original score-gated hedge constants ──────────────────────────────

PARTICLE_WEIGHT = 0.85
COVERAGE_WEIGHT = 0.15

SCORE_MARGIN = 5.0
CALIBRATION_THRESHOLD = 0.15
MIN_DISAGREEMENT_SEEDS = 2

# ── Confidence-gated anchor hedge constants ───────────────────────────

# Confidence thresholds for mode gating.
ANCHOR_HEDGE_CONFIDENCE_FLOOR: float = 0.30
# Maximum anchor weight when confidence = 0.
ANCHOR_MAX_WEIGHT: float = 0.85
# Minimum anchor weight (even when highly confident, keep a thin blend).
ANCHOR_MIN_WEIGHT: float = 0.0


# ── Score-gated hedge (existing API, unchanged) ──────────────────────


def should_hedge(
    candidate_mean_score: float,
    fixed_coverage_mean_score: float,
    per_seed_disagreements: list[float] | None = None,
) -> bool:
    """Determine if score-gated hedge should activate.

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
    """Apply score-gated hedge blend and finalize.

    q_final = 0.85 * q_particle + 0.15 * q_fixed_coverage

    Both inputs should already be safe/normalized, but the result is
    re-finalized to guarantee safety.
    """
    hedged: list[NDArray[np.float64]] = []
    for i, (p_t, c_t) in enumerate(zip(particle_tensors, coverage_tensors, strict=True)):
        blended = PARTICLE_WEIGHT * p_t + COVERAGE_WEIGHT * c_t
        initial_state = initial_states[i] if i < len(initial_states) else None
        finalized = finalize_tensor(blended, height, width, initial_state)
        hedged.append(finalized)
    return hedged


# ── Confidence-gated anchor hedge (new API) ──────────────────────────


def should_hedge_from_confidence(confidence: PosteriorConfidence) -> bool:
    """Determine if anchor hedge should activate based on confidence.

    The anchor hedge activates whenever the recommended mode is ``"blend"``
    or ``"anchor"`` — i.e. whenever the posterior is not fully trusted.

    Args:
        confidence: Confidence summary for a seed or round.

    Returns:
        True if the prediction should be blended with the structural anchor.
    """
    return confidence.recommended_mode != "particle"


def compute_blend_weight(confidence: PosteriorConfidence) -> float:
    """Compute the anchor blend weight from a confidence score.

    Returns the fraction of the structural anchor in the final blend:

    * ``confidence_score >= 1.0`` → ``ANCHOR_MIN_WEIGHT`` (pure particle)
    * ``confidence_score <= 0.0`` → ``ANCHOR_MAX_WEIGHT`` (pure anchor)

    The mapping is linear between those extremes.

    Args:
        confidence: Confidence summary.

    Returns:
        Float in [ANCHOR_MIN_WEIGHT, ANCHOR_MAX_WEIGHT] representing the
        anchor tensor's contribution to the blend.
    """
    # Invert confidence: higher confidence → lower anchor weight.
    t = max(0.0, min(1.0, 1.0 - confidence.confidence_score))
    return ANCHOR_MIN_WEIGHT + t * (ANCHOR_MAX_WEIGHT - ANCHOR_MIN_WEIGHT)


def apply_anchor_hedge(
    particle_tensors: list[NDArray[np.float64]],
    anchor_tensors: list[NDArray[np.float64]],
    confidences: list[PosteriorConfidence],
    initial_states: list[InitialState],
    height: int,
    width: int,
) -> list[NDArray[np.float64]]:
    """Apply per-seed confidence-gated blending with structural anchors.

    For each seed:

    * If the confidence recommends ``"particle"`` mode, the particle
      tensor is used directly (no anchor blending).
    * If the confidence recommends ``"blend"`` or ``"anchor"``, the
      anchor weight is computed from the confidence score and the tensors
      are linearly blended.

    All outputs are finalized to guarantee safety.

    Args:
        particle_tensors: Per-seed posterior-predictive tensors (H, W, 6).
        anchor_tensors: Per-seed structural anchor tensors (H, W, 6).
        confidences: Per-seed confidence summaries.
        initial_states: Per-seed initial states for static overrides.
        height: Map height.
        width: Map width.

    Returns:
        Per-seed finalized tensors.
    """
    result: list[NDArray[np.float64]] = []
    for i, (p_t, a_t, conf) in enumerate(
        zip(particle_tensors, anchor_tensors, confidences, strict=True)
    ):
        if should_hedge_from_confidence(conf):
            anchor_w = compute_blend_weight(conf)
            particle_w = 1.0 - anchor_w
            blended = particle_w * p_t + anchor_w * a_t
        else:
            blended = p_t.copy()

        initial_state = initial_states[i] if i < len(initial_states) else None
        finalized = finalize_tensor(blended, height, width, initial_state)
        result.append(finalized)
    return result
