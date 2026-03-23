"""Posterior confidence scoring for hybrid decision-making.

Summarises the solver's trust in posterior-driven predictions using
signals already available on ``PosteriorState``.  The confidence score
determines whether the final prediction should rely on the particle
posterior, blend with the structural anchor, or fall back entirely.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PosteriorConfidence:
    """Confidence summary for one seed or the overall round.

    Attributes:
        seed_index: The seed this confidence applies to (-1 for round-level).
        ess: Effective sample size of the particle posterior.
        top_particle_mass: Normalised weight of the dominant particle.
        disagreement: Mean calibration disagreement across observed viewports.
        entropy_mass: Mean per-cell entropy of the posterior-predictive tensor
            (0 = certain, log(6) ≈ 1.79 = maximum uncertainty for 6 classes).
        confidence_score: Scalar in [0, 1] summarising overall trust.
        recommended_mode: One of ``"particle"``, ``"blend"``, or ``"anchor"``.
    """

    seed_index: int
    ess: float
    top_particle_mass: float
    disagreement: float
    entropy_mass: float
    confidence_score: float
    recommended_mode: str  # "particle" | "blend" | "anchor"


# ── Thresholds ────────────────────────────────────────────────────────

# ESS below this is considered very weak evidence.
ESS_LOW: float = 3.0
# ESS above this is considered strong posterior.
ESS_HIGH: float = 8.0

# Top-particle mass above this suggests collapse / overconfidence.
TOP_MASS_COLLAPSE: float = 0.80

# Disagreement above this signals model-observation mismatch.
DISAGREEMENT_HIGH: float = 0.15

# Confidence thresholds for mode selection.
CONFIDENCE_PARTICLE_THRESHOLD: float = 0.65
CONFIDENCE_BLEND_THRESHOLD: float = 0.35


def _ess_score(ess: float, n_particles: int) -> float:
    """Map ESS to a [0, 1] score.

    Linear ramp from 0 at ESS_LOW to 1 at ESS_HIGH, clamped.
    """
    if n_particles <= 0:
        return 0.0
    if ess <= ESS_LOW:
        return 0.0
    if ess >= ESS_HIGH:
        return 1.0
    return float((ess - ESS_LOW) / (ESS_HIGH - ESS_LOW))


def _collapse_penalty(top_mass: float) -> float:
    """Penalty in [0, 1] for posterior collapse (high top-particle mass).

    Returns 0 when mass <= TOP_MASS_COLLAPSE, ramps to 1 at mass = 1.
    """
    if top_mass <= TOP_MASS_COLLAPSE:
        return 0.0
    return float(min((top_mass - TOP_MASS_COLLAPSE) / (1.0 - TOP_MASS_COLLAPSE), 1.0))


def _disagreement_penalty(disagreement: float) -> float:
    """Penalty in [0, 1] for calibration disagreement.

    Returns 0 when disagreement <= DISAGREEMENT_HIGH / 2,
    ramps to 1 at 2 * DISAGREEMENT_HIGH.
    """
    low = DISAGREEMENT_HIGH / 2.0
    high = DISAGREEMENT_HIGH * 2.0
    if disagreement <= low:
        return 0.0
    if disagreement >= high:
        return 1.0
    return float((disagreement - low) / (high - low))


def compute_confidence(
    seed_index: int,
    ess: float,
    top_particle_mass: float,
    disagreement: float,
    entropy_mass: float,
    n_particles: int,
) -> PosteriorConfidence:
    """Compute a deterministic confidence summary from posterior signals.

    The confidence score is a weighted combination of three components:

    * **ESS score** (weight 0.7) — how well-distributed the posterior mass is.
    * **Collapse penalty** (weight 0.2) — deducted when one particle dominates.
    * **Disagreement penalty** (weight 0.1) — deducted when observations
      conflict with the model.

    The resulting score is clamped to [0, 1] and mapped to a recommended
    mode: ``"particle"`` if confident, ``"blend"`` if moderate, or
    ``"anchor"`` if weak.

    Args:
        seed_index: Seed identifier (-1 for round-level).
        ess: Effective sample size.
        top_particle_mass: Normalised weight of the top particle.
        disagreement: Mean calibration disagreement.
        entropy_mass: Mean per-cell entropy of predictions.
        n_particles: Total number of particles in the posterior.

    Returns:
        A frozen ``PosteriorConfidence`` instance.
    """
    ess_s = _ess_score(ess, n_particles)
    collapse_p = _collapse_penalty(top_particle_mass)
    disagree_p = _disagreement_penalty(disagreement)

    raw = 0.7 * ess_s - 0.2 * collapse_p - 0.1 * disagree_p
    score = max(0.0, min(1.0, raw))

    if score >= CONFIDENCE_PARTICLE_THRESHOLD:
        mode = "particle"
    elif score >= CONFIDENCE_BLEND_THRESHOLD:
        mode = "blend"
    else:
        mode = "anchor"

    return PosteriorConfidence(
        seed_index=seed_index,
        ess=ess,
        top_particle_mass=top_particle_mass,
        disagreement=disagreement,
        entropy_mass=entropy_mass,
        confidence_score=score,
        recommended_mode=mode,
    )
