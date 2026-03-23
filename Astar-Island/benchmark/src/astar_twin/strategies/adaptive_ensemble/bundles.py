from __future__ import annotations

from dataclasses import dataclass

from astar_twin.params import SimulationParams

_DEFAULT_WEIGHT = 0.30
_VARIANT_WEIGHT = 0.0875


@dataclass(frozen=True)
class Bundle:
    name: str
    params: SimulationParams
    prior_weight: float


def get_bundles() -> list[Bundle]:
    base = SimulationParams()
    return [
        Bundle(name="DEFAULT", params=SimulationParams(), prior_weight=_DEFAULT_WEIGHT),
        Bundle(
            name="GROWTH_HIGH",
            params=SimulationParams(
                expansion_rate=base.expansion_rate * 1.15,
                population_growth_rate=base.population_growth_rate * 1.10,
                food_base_yield=base.food_base_yield * 1.10,
                carrying_capacity_base=base.carrying_capacity_base * 1.10,
            ),
            prior_weight=_VARIANT_WEIGHT,
        ),
        Bundle(
            name="GROWTH_LOW",
            params=SimulationParams(
                expansion_rate=base.expansion_rate * 0.85,
                population_growth_rate=base.population_growth_rate * 0.90,
                food_base_yield=base.food_base_yield * 0.90,
                carrying_capacity_base=base.carrying_capacity_base * 0.90,
            ),
            prior_weight=_VARIANT_WEIGHT,
        ),
        Bundle(
            name="CONFLICT_HARSH",
            params=SimulationParams(
                raid_base_prob=base.raid_base_prob * 1.35,
                raid_range_base=base.raid_range_base + 2,
                winter_severity_mean=base.winter_severity_mean * 1.20,
                collapse_threshold=base.collapse_threshold * 0.90,
            ),
            prior_weight=_VARIANT_WEIGHT,
        ),
        Bundle(
            name="CONFLICT_MILD",
            params=SimulationParams(
                raid_base_prob=base.raid_base_prob * 0.75,
                winter_severity_mean=base.winter_severity_mean * 0.85,
                collapse_threshold=base.collapse_threshold * 1.10,
            ),
            prior_weight=_VARIANT_WEIGHT,
        ),
        Bundle(
            name="RUIN_FAST_RECOVERY",
            params=SimulationParams(
                reclaim_rate=base.reclaim_rate * 1.20,
                ruin_decay_delay=max(2, base.ruin_decay_delay - 1),
                ruin_forest_rate=base.ruin_forest_rate * 0.80,
            ),
            prior_weight=_VARIANT_WEIGHT,
        ),
        Bundle(
            name="RUIN_PERSISTENT",
            params=SimulationParams(
                reclaim_rate=base.reclaim_rate * 0.80,
                ruin_decay_delay=base.ruin_decay_delay + 1,
                ruin_forest_rate=base.ruin_forest_rate * 1.25,
                ruin_plain_rate=base.ruin_plain_rate * 1.25,
            ),
            prior_weight=_VARIANT_WEIGHT,
        ),
        Bundle(
            name="TRADE_HIGH",
            params=SimulationParams(
                trade_range=base.trade_range + 2,
                trade_value_scale=base.trade_value_scale * 1.25,
                tech_diffusion_rate=base.tech_diffusion_rate * 1.15,
            ),
            prior_weight=_VARIANT_WEIGHT,
        ),
        Bundle(
            name="TRADE_LOW",
            params=SimulationParams(
                trade_range=max(3, base.trade_range - 2),
                trade_value_scale=base.trade_value_scale * 0.75,
                tech_diffusion_rate=base.tech_diffusion_rate * 0.85,
            ),
            prior_weight=_VARIANT_WEIGHT,
        ),
    ]


DEFAULT_BUNDLES = get_bundles()
