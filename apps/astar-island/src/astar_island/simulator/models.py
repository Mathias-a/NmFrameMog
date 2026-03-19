"""Data models for the Norse world simulator."""

from __future__ import annotations

from dataclasses import dataclass

from astar_island.terrain import TerrainType


@dataclass(slots=True)
class Settlement:
    """A Norse settlement on the map."""

    x: int
    y: int
    population: int
    food: int
    wealth: int
    defense: int
    tech_level: int
    faction_id: int
    has_port: bool
    has_longship: bool
    alive: bool


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Configuration for the Norse world simulation."""

    # Growth
    food_per_forest: int = 10
    food_per_plains: int = 5
    expansion_threshold: int = 150
    port_development_prob: float = 0.3
    longship_build_prob: float = 0.2
    # Conflict
    raid_range: int = 3
    longship_range_bonus: int = 5
    raid_aggression_threshold: int = 30
    raid_success_base: float = 0.4
    # Trade
    trade_range: int = 4
    trade_food_bonus: int = 15
    trade_wealth_bonus: int = 10
    tech_diffusion_rate: float = 0.5
    # Winter
    winter_severity_mean: float = 0.5
    winter_severity_std: float = 0.15
    collapse_food_threshold: int = 0
    # Environment
    ruin_reclaim_prob: float = 0.15
    ruin_to_forest_prob: float = 0.10
    ruin_to_plains_prob: float = 0.05
    # Mapgen
    fjord_count: int = 3
    fjord_max_length: int = 12
    mountain_chains: int = 2
    mountain_walk_length: int = 8
    forest_patch_count: int = 6
    forest_patch_radius: int = 3
    initial_settlement_count: int = 5
    min_settlement_spacing: int = 6


@dataclass(slots=True)
class WorldState:
    """The full state of the simulated Norse world."""

    width: int
    height: int
    grid: list[list[TerrainType]]  # grid[y][x]
    settlements: list[Settlement]
    year: int
