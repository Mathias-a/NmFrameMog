from __future__ import annotations

from astar_twin.params import SimulationParams, sample_default_prior_params


def test_sampling_is_reproducible_for_same_seed() -> None:
    sampled_a = sample_default_prior_params(seed=17)
    sampled_b = sample_default_prior_params(seed=17)

    assert sampled_a == sampled_b


def test_sampling_changes_for_different_seeds() -> None:
    sampled_a = sample_default_prior_params(seed=17)
    sampled_b = sample_default_prior_params(seed=18)

    assert sampled_a != sampled_b


def test_sampling_preserves_non_inferred_defaults() -> None:
    defaults = SimulationParams()
    sampled = sample_default_prior_params(seed=17)

    assert sampled != defaults
    assert sampled.init_population_mean != defaults.init_population_mean
    assert sampled.food_base_yield != defaults.food_base_yield
    assert sampled.ruin_plain_rate != defaults.ruin_plain_rate


def test_sampling_keeps_integer_fields_integral() -> None:
    sampled = sample_default_prior_params(seed=17)

    assert isinstance(sampled.expansion_radius, int)
    assert isinstance(sampled.trade_range, int)
    assert isinstance(sampled.war_cooldown_years, int)
    assert isinstance(sampled.ruin_decay_delay, int)


def test_sampling_keeps_fractional_fields_in_valid_ranges() -> None:
    sampled = sample_default_prior_params(seed=17)

    assert 0.0 <= sampled.expansion_population_transfer_fraction <= 1.0
    assert 0.0 <= sampled.raid_loot_frac <= 1.0
    assert 0.0 <= sampled.collapse_dispersal_fraction <= 1.0
    assert 0.0 <= sampled.reclaim_inheritance_frac <= 1.0


def test_zero_spread_locks_numeric_fields_to_defaults() -> None:
    defaults = SimulationParams()
    sampled = sample_default_prior_params(seed=17, spread=0.0)

    assert sampled.init_population_mean == defaults.init_population_mean
    assert sampled.trade_range == defaults.trade_range
    assert sampled.ruin_plain_rate == defaults.ruin_plain_rate


def test_invalid_spread_raises() -> None:
    try:
        sample_default_prior_params(seed=17, spread=1.1)
        raise AssertionError("Expected invalid spread to raise")
    except ValueError as exc:
        assert "spread" in str(exc)
