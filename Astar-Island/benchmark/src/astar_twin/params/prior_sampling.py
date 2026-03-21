from __future__ import annotations

from numpy.random import default_rng

from astar_twin.params.simulation_params import (
    AdjacencyMode,
    DistanceMetric,
    SimulationParams,
    UpdateOrderMode,
)

_DEFAULTS = SimulationParams()
_ADJACENCY_CHOICES = tuple(AdjacencyMode.__members__.values())
_DISTANCE_CHOICES = tuple(DistanceMetric.__members__.values())
_UPDATE_ORDER_CHOICES = tuple(UpdateOrderMode.__members__.values())

_CONTINUOUS_RANGES: dict[str, tuple[float, float, float]] = {
    "init_population_mean": (0.5, 3.0, _DEFAULTS.init_population_mean),
    "init_population_sigma": (0.05, 0.6, _DEFAULTS.init_population_sigma),
    "init_food_per_pop": (0.3, 1.4, _DEFAULTS.init_food_per_pop),
    "init_food_sigma": (0.02, 0.5, _DEFAULTS.init_food_sigma),
    "init_wealth_mean": (0.0, 1.0, _DEFAULTS.init_wealth_mean),
    "init_port_wealth_bonus": (0.0, 1.2, _DEFAULTS.init_port_wealth_bonus),
    "init_wealth_sigma": (0.01, 0.4, _DEFAULTS.init_wealth_sigma),
    "init_defense_per_pop": (0.0, 1.0, _DEFAULTS.init_defense_per_pop),
    "init_defense_sigma": (0.01, 0.4, _DEFAULTS.init_defense_sigma),
    "init_tech_base": (0.0, 0.2, _DEFAULTS.init_tech_base),
    "init_port_tech_bonus": (0.0, 0.6, _DEFAULTS.init_port_tech_bonus),
    "food_base_yield": (0.05, 1.0, _DEFAULTS.food_base_yield),
    "food_per_adjacent_forest": (0.0, 0.9, _DEFAULTS.food_per_adjacent_forest),
    "food_crowding_penalty": (0.0, 0.4, _DEFAULTS.food_crowding_penalty),
    "population_food_upkeep": (0.05, 0.6, _DEFAULTS.population_food_upkeep),
    "population_growth_rate": (0.01, 0.5, _DEFAULTS.population_growth_rate),
    "growth_food_buffer": (0.05, 1.0, _DEFAULTS.growth_food_buffer),
    "carrying_capacity_base": (1.0, 6.0, _DEFAULTS.carrying_capacity_base),
    "carrying_capacity_per_adjacent_forest": (
        0.0,
        2.0,
        _DEFAULTS.carrying_capacity_per_adjacent_forest,
    ),
    "carrying_capacity_port_bonus": (0.0, 3.0, _DEFAULTS.carrying_capacity_port_bonus),
    "prosperity_logit_scale": (0.1, 2.0, _DEFAULTS.prosperity_logit_scale),
    "prosperity_threshold_port": (0.5, 3.0, _DEFAULTS.prosperity_threshold_port),
    "prosperity_threshold_longship": (1.0, 4.0, _DEFAULTS.prosperity_threshold_longship),
    "prosperity_threshold_expand": (0.5, 3.0, _DEFAULTS.prosperity_threshold_expand),
    "expansion_rate": (0.05, 0.50, _DEFAULTS.expansion_rate),
    "expansion_radius": (1.0, 6.0, float(_DEFAULTS.expansion_radius)),
    "expansion_min_spacing": (1.0, 5.0, float(_DEFAULTS.expansion_min_spacing)),
    "expansion_site_temperature": (0.0, 1.0, _DEFAULTS.expansion_site_temperature),
    "expansion_site_forest_penalty": (0.0, 2.0, _DEFAULTS.expansion_site_forest_penalty),
    "expansion_site_coastal_bonus": (0.0, 1.0, _DEFAULTS.expansion_site_coastal_bonus),
    "expansion_site_distance_decay": (0.0, 1.0, _DEFAULTS.expansion_site_distance_decay),
    "expansion_population_transfer_fraction": (
        0.01,
        0.9,
        _DEFAULTS.expansion_population_transfer_fraction,
    ),
    "expansion_food_transfer_fraction": (
        0.01,
        0.9,
        _DEFAULTS.expansion_food_transfer_fraction,
    ),
    "expansion_wealth_transfer_fraction": (
        0.01,
        0.9,
        _DEFAULTS.expansion_wealth_transfer_fraction,
    ),
    "expansion_max_children_per_year": (
        0.0,
        3.0,
        float(_DEFAULTS.expansion_max_children_per_year),
    ),
    "raid_base_prob": (0.01, 0.20, _DEFAULTS.raid_base_prob),
    "raid_desperation_food_pc": (0.05, 0.95, _DEFAULTS.raid_desperation_food_pc),
    "raid_desperation_scale": (0.01, 0.5, _DEFAULTS.raid_desperation_scale),
    "raid_success_scale": (0.3, 1.5, _DEFAULTS.raid_success_scale),
    "raid_loot_frac": (0.01, 0.9, _DEFAULTS.raid_loot_frac),
    "raid_range_base": (2.0, 10.0, float(_DEFAULTS.raid_range_base)),
    "raid_range_longship_bonus": (
        2.0,
        14.0,
        float(_DEFAULTS.raid_range_longship_bonus),
    ),
    "raid_damage_frac": (0.05, 0.50, _DEFAULTS.raid_damage_frac),
    "raid_capture_threshold": (0.5, 3.0, _DEFAULTS.raid_capture_threshold),
    "trade_range": (3.0, 15.0, float(_DEFAULTS.trade_range)),
    "war_cooldown_years": (0.0, 6.0, float(_DEFAULTS.war_cooldown_years)),
    "trade_value_scale": (0.05, 0.50, _DEFAULTS.trade_value_scale),
    "tech_diffusion_rate": (0.0, 0.5, _DEFAULTS.tech_diffusion_rate),
    "tech_economic_bonus": (0.0, 0.6, _DEFAULTS.tech_economic_bonus),
    "tech_military_bonus": (0.0, 0.6, _DEFAULTS.tech_military_bonus),
    "winter_severity_mean": (0.1, 0.9, _DEFAULTS.winter_severity_mean),
    "winter_severity_concentration": (
        2.0,
        50.0,
        _DEFAULTS.winter_severity_concentration,
    ),
    "winter_severity_autocorr": (0.0, 0.8, _DEFAULTS.winter_severity_autocorr),
    "winter_food_loss_flat": (0.0, 0.6, _DEFAULTS.winter_food_loss_flat),
    "winter_food_loss_per_population": (
        0.02,
        0.30,
        _DEFAULTS.winter_food_loss_per_population,
    ),
    "winter_food_loss_severity_multiplier": (
        0.5,
        2.0,
        _DEFAULTS.winter_food_loss_severity_multiplier,
    ),
    "collapse_threshold": (0.3, 2.5, _DEFAULTS.collapse_threshold),
    "collapse_softness": (0.05, 0.80, _DEFAULTS.collapse_softness),
    "collapse_food_floor": (0.0, 0.5, _DEFAULTS.collapse_food_floor),
    "collapse_raid_stress_weight": (0.2, 3.0, _DEFAULTS.collapse_raid_stress_weight),
    "collapse_winter_severity_weight": (
        0.1,
        2.0,
        _DEFAULTS.collapse_winter_severity_weight,
    ),
    "collapse_defense_buffer_weight": (0.0, 2.0, _DEFAULTS.collapse_defense_buffer_weight),
    "raid_stress_decay": (0.0, 1.0, _DEFAULTS.raid_stress_decay),
    "collapse_dispersal_radius": (1.0, 12.0, float(_DEFAULTS.collapse_dispersal_radius)),
    "collapse_dispersal_fraction": (0.0, 0.99, _DEFAULTS.collapse_dispersal_fraction),
    "collapse_dispersal_distance_decay": (
        0.0,
        1.0,
        _DEFAULTS.collapse_dispersal_distance_decay,
    ),
    "reclaim_radius": (1.0, 8.0, float(_DEFAULTS.reclaim_radius)),
    "reclaim_threshold": (0.5, 3.5, _DEFAULTS.reclaim_threshold),
    "reclaim_rate": (0.01, 0.40, _DEFAULTS.reclaim_rate),
    "reclaim_inheritance_frac": (0.0, 0.9, _DEFAULTS.reclaim_inheritance_frac),
    "ruin_decay_delay": (1.0, 12.0, float(_DEFAULTS.ruin_decay_delay)),
    "ruin_forest_rate": (0.02, 0.30, _DEFAULTS.ruin_forest_rate),
    "ruin_plain_rate": (0.01, 0.30, _DEFAULTS.ruin_plain_rate),
}

_INTEGER_PARAMS = {
    "expansion_radius",
    "expansion_min_spacing",
    "expansion_max_children_per_year",
    "raid_range_base",
    "raid_range_longship_bonus",
    "trade_range",
    "war_cooldown_years",
    "collapse_dispersal_radius",
    "reclaim_radius",
    "ruin_decay_delay",
}


def sample_default_prior_params(
    seed: int,
    defaults: SimulationParams | None = None,
    spread: float = 1.0,
) -> SimulationParams:
    """Sample a reproducible benchmark prior around the default parameter set.

    The benchmark uses this only when a fixture still carries ``DEFAULT_PRIOR``
    metadata. The sampled parameter set is benchmark-wide for that fixture and
    deterministic for a fixed seed. ``spread`` only affects numeric fields;
    enum choices remain stochastic for a fixed seed.
    """

    if not 0.0 <= spread <= 1.0:
        raise ValueError("spread must be between 0.0 and 1.0")

    base = defaults or SimulationParams()
    rng = default_rng(seed)
    sampled = {
        **base.__dict__,
        "adjacency_mode": rng.choice(_ADJACENCY_CHOICES),
        "distance_metric": rng.choice(_DISTANCE_CHOICES),
        "update_order_mode": rng.choice(_UPDATE_ORDER_CHOICES),
    }

    for name, (lo, hi, mode) in _CONTINUOUS_RANGES.items():
        sampled_lo = mode - (mode - lo) * spread
        sampled_hi = mode + (hi - mode) * spread
        value = mode if sampled_lo == sampled_hi else rng.triangular(sampled_lo, mode, sampled_hi)
        if name in _INTEGER_PARAMS:
            sampled[name] = int(round(value))
        else:
            sampled[name] = float(value)

    return SimulationParams(**sampled)
