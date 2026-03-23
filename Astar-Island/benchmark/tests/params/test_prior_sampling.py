from __future__ import annotations

from dataclasses import fields

import pytest

from astar_twin.params.prior_sampling import (
    DEFAULT_PRIOR_CONFIG,
    PriorConfig,
    sample_params,
    sample_params_batch,
)
from astar_twin.params.simulation_params import SimulationParams


def _numeric_field_names() -> list[str]:
    return [
        field.name
        for field in fields(SimulationParams)
        if field.name not in {"adjacency_mode", "distance_metric", "update_order_mode"}
    ]


def test_default_prior_config_returns_default_simulation_params() -> None:
    sampled = sample_params(DEFAULT_PRIOR_CONFIG)

    assert sampled == SimulationParams()


def test_nonzero_spread_changes_from_defaults() -> None:
    defaults = SimulationParams()
    sampled = sample_params(PriorConfig(spread=0.2, seed=42))

    changed_fields = [
        field_name
        for field_name in _numeric_field_names()
        if getattr(sampled, field_name) != pytest.approx(getattr(defaults, field_name))
    ]

    assert sampled != defaults
    assert changed_fields


def test_same_config_is_deterministic() -> None:
    config = PriorConfig(spread=0.2, seed=42)

    assert sample_params(config) == sample_params(config)


def test_different_seeds_produce_different_params() -> None:
    config_a = PriorConfig(spread=0.2, seed=42)
    config_b = PriorConfig(spread=0.2, seed=43)

    assert sample_params(config_a) != sample_params(config_b)


def test_perturbed_int_fields_are_clamped_and_enum_fields_unchanged() -> None:
    defaults = SimulationParams()
    sampled = sample_params(PriorConfig(spread=0.2, seed=42))

    for field in fields(SimulationParams):
        sampled_value = getattr(sampled, field.name)
        default_value = getattr(defaults, field.name)
        if field.name in {"adjacency_mode", "distance_metric", "update_order_mode"}:
            assert sampled_value == default_value
        elif field.type in ("int", int):
            assert sampled_value >= 1


def test_sample_params_batch_returns_five_distinct_params() -> None:
    batch = sample_params_batch(PriorConfig(spread=0.2, seed=42), 5)

    assert len(batch) == 5
    assert (
        len(
            {
                tuple(getattr(params, field.name) for field in fields(SimulationParams))
                for params in batch
            }
        )
        == 5
    )
