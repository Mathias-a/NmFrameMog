"""Tests for posterior confidence scoring."""

from __future__ import annotations

import pytest

from astar_twin.solver.inference.confidence import (
    CONFIDENCE_BLEND_THRESHOLD,
    CONFIDENCE_PARTICLE_THRESHOLD,
    PosteriorConfidence,
    compute_confidence,
)


class TestPosteriorConfidenceDataclass:
    """Verify the frozen dataclass basics."""

    def test_is_frozen(self) -> None:
        c = PosteriorConfidence(
            seed_index=0,
            ess=10.0,
            top_particle_mass=0.3,
            disagreement=0.05,
            entropy_mass=0.5,
            confidence_score=0.8,
            recommended_mode="particle",
        )
        with pytest.raises(AttributeError):
            c.confidence_score = 0.9  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        c = PosteriorConfidence(
            seed_index=2,
            ess=5.0,
            top_particle_mass=0.6,
            disagreement=0.1,
            entropy_mass=0.7,
            confidence_score=0.4,
            recommended_mode="blend",
        )
        assert c.seed_index == 2
        assert c.ess == 5.0
        assert c.recommended_mode == "blend"


class TestComputeConfidence:
    """Test compute_confidence determinism, boundary cases, and mode selection."""

    def test_deterministic(self) -> None:
        """Same inputs produce identical output."""
        kwargs = dict(
            seed_index=0,
            ess=6.0,
            top_particle_mass=0.4,
            disagreement=0.1,
            entropy_mass=0.5,
            n_particles=12,
        )
        c1 = compute_confidence(**kwargs)
        c2 = compute_confidence(**kwargs)
        assert c1 == c2

    def test_high_ess_low_mass_low_disagreement_gives_particle_mode(self) -> None:
        """Strong posterior → "particle" mode."""
        c = compute_confidence(
            seed_index=0,
            ess=10.0,
            top_particle_mass=0.3,
            disagreement=0.02,
            entropy_mass=0.3,
            n_particles=12,
        )
        assert c.recommended_mode == "particle"
        assert c.confidence_score >= CONFIDENCE_PARTICLE_THRESHOLD

    def test_low_ess_gives_anchor_mode(self) -> None:
        """Very weak posterior → "anchor" mode."""
        c = compute_confidence(
            seed_index=0,
            ess=1.5,
            top_particle_mass=0.9,
            disagreement=0.3,
            entropy_mass=1.5,
            n_particles=12,
        )
        assert c.recommended_mode == "anchor"
        assert c.confidence_score < CONFIDENCE_BLEND_THRESHOLD

    def test_moderate_ess_gives_blend_mode(self) -> None:
        """Moderate confidence → "blend" mode."""
        c = compute_confidence(
            seed_index=0,
            ess=6.0,
            top_particle_mass=0.4,
            disagreement=0.05,
            entropy_mass=0.8,
            n_particles=12,
        )
        assert c.recommended_mode == "blend"
        assert CONFIDENCE_BLEND_THRESHOLD <= c.confidence_score < CONFIDENCE_PARTICLE_THRESHOLD

    def test_high_collapse_lowers_confidence(self) -> None:
        """Top particle mass near 1.0 should penalise confidence."""
        c_normal = compute_confidence(
            seed_index=0,
            ess=8.0,
            top_particle_mass=0.3,
            disagreement=0.05,
            entropy_mass=0.5,
            n_particles=12,
        )
        c_collapsed = compute_confidence(
            seed_index=0,
            ess=8.0,
            top_particle_mass=0.95,
            disagreement=0.05,
            entropy_mass=0.5,
            n_particles=12,
        )
        assert c_collapsed.confidence_score < c_normal.confidence_score

    def test_high_disagreement_lowers_confidence(self) -> None:
        """High calibration disagreement should penalise confidence."""
        c_normal = compute_confidence(
            seed_index=0,
            ess=8.0,
            top_particle_mass=0.3,
            disagreement=0.02,
            entropy_mass=0.5,
            n_particles=12,
        )
        c_disagree = compute_confidence(
            seed_index=0,
            ess=8.0,
            top_particle_mass=0.3,
            disagreement=0.5,
            entropy_mass=0.5,
            n_particles=12,
        )
        assert c_disagree.confidence_score < c_normal.confidence_score

    def test_score_clamped_to_unit_interval(self) -> None:
        """Confidence score is always in [0, 1]."""
        # Best case: high ESS, no penalties
        c_best = compute_confidence(
            seed_index=0,
            ess=100.0,
            top_particle_mass=0.01,
            disagreement=0.0,
            entropy_mass=0.0,
            n_particles=200,
        )
        assert 0.0 <= c_best.confidence_score <= 1.0

        # Worst case: low ESS, max penalties
        c_worst = compute_confidence(
            seed_index=0,
            ess=0.5,
            top_particle_mass=1.0,
            disagreement=1.0,
            entropy_mass=1.8,
            n_particles=12,
        )
        assert 0.0 <= c_worst.confidence_score <= 1.0

    def test_zero_particles(self) -> None:
        """Edge case: no particles should produce anchor mode."""
        c = compute_confidence(
            seed_index=0,
            ess=0.0,
            top_particle_mass=0.0,
            disagreement=0.0,
            entropy_mass=0.0,
            n_particles=0,
        )
        assert c.recommended_mode == "anchor"
        assert c.confidence_score == 0.0

    def test_round_level_seed_index(self) -> None:
        """Verify seed_index=-1 (round-level) is preserved."""
        c = compute_confidence(
            seed_index=-1,
            ess=6.0,
            top_particle_mass=0.4,
            disagreement=0.1,
            entropy_mass=0.5,
            n_particles=12,
        )
        assert c.seed_index == -1

    def test_entropy_mass_is_stored(self) -> None:
        """entropy_mass is not used in scoring but must be stored for diagnostics."""
        c = compute_confidence(
            seed_index=0,
            ess=6.0,
            top_particle_mass=0.4,
            disagreement=0.1,
            entropy_mass=1.234,
            n_particles=12,
        )
        assert c.entropy_mass == 1.234
