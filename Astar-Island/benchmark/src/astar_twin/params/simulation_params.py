from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AdjacencyMode(StrEnum):
    MOORE8 = "moore8"
    VON_NEUMANN4 = "von_neumann4"


class DistanceMetric(StrEnum):
    CHEBYSHEV = "chebyshev"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class UpdateOrderMode(StrEnum):
    STABLE = "stable"
    RANDOM_PER_YEAR = "random_per_year"
    RANDOM_PER_PHASE = "random_per_phase"


@dataclass
class SimulationParams:
    adjacency_mode: AdjacencyMode = AdjacencyMode.MOORE8
    distance_metric: DistanceMetric = DistanceMetric.CHEBYSHEV
    update_order_mode: UpdateOrderMode = UpdateOrderMode.RANDOM_PER_PHASE

    init_population_mean: float = 1.4
    init_population_sigma: float = 0.25
    init_food_per_pop: float = 0.80
    init_food_sigma: float = 0.20
    init_wealth_mean: float = 0.30
    init_port_wealth_bonus: float = 0.40
    init_wealth_sigma: float = 0.15
    init_defense_per_pop: float = 0.20
    init_defense_sigma: float = 0.10
    init_tech_base: float = 0.00
    init_port_tech_bonus: float = 0.15

    food_base_yield: float = 0.35
    food_per_adjacent_forest: float = 0.22
    food_crowding_penalty: float = 0.08
    population_food_upkeep: float = 0.18
    population_growth_rate: float = 0.16
    growth_food_buffer: float = 0.35
    carrying_capacity_base: float = 2.50
    carrying_capacity_per_adjacent_forest: float = 0.75
    carrying_capacity_port_bonus: float = 1.25

    prosperity_logit_scale: float = 0.60
    prosperity_threshold_port: float = 1.70
    prosperity_threshold_longship: float = 2.10
    prosperity_threshold_expand: float = 1.40
    expansion_rate: float = 0.18
    expansion_radius: int = 3
    expansion_min_spacing: int = 2
    expansion_site_temperature: float = 0.60
    expansion_site_forest_penalty: float = 0.80
    expansion_site_coastal_bonus: float = 0.20
    expansion_site_distance_decay: float = 0.45
    expansion_population_transfer_fraction: float = 0.30
    expansion_food_transfer_fraction: float = 0.20
    expansion_wealth_transfer_fraction: float = 0.20
    expansion_max_children_per_year: int = 1

    raid_range_base: int = 5
    raid_range_longship_bonus: int = 7
    raid_base_prob: float = 0.06
    raid_desperation_food_pc: float = 0.45
    raid_desperation_scale: float = 0.12
    raid_success_scale: float = 0.90
    raid_loot_frac: float = 0.25
    raid_damage_frac: float = 0.18
    raid_capture_threshold: float = 1.60

    trade_range: int = 8
    war_cooldown_years: int = 3
    trade_value_scale: float = 0.22
    tech_diffusion_rate: float = 0.08
    tech_economic_bonus: float = 0.15
    tech_military_bonus: float = 0.12

    winter_severity_mean: float = 0.45
    winter_severity_concentration: float = 12.0
    winter_severity_autocorr: float = 0.00
    winter_food_loss_flat: float = 0.18
    winter_food_loss_per_population: float = 0.10
    winter_food_loss_severity_multiplier: float = 1.00
    collapse_food_floor: float = 0.00
    collapse_threshold: float = 1.00
    collapse_softness: float = 0.25
    collapse_raid_stress_weight: float = 1.00
    collapse_winter_severity_weight: float = 0.80
    collapse_defense_buffer_weight: float = 0.40
    raid_stress_decay: float = 0.50
    collapse_dispersal_radius: int = 5
    collapse_dispersal_fraction: float = 0.60
    collapse_dispersal_distance_decay: float = 0.60

    reclaim_radius: int = 4
    reclaim_threshold: float = 1.80
    reclaim_rate: float = 0.12
    reclaim_inheritance_frac: float = 0.30
    ruin_decay_delay: int = 5
    ruin_forest_rate: float = 0.10
    ruin_plain_rate: float = 0.06
