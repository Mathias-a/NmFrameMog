from __future__ import annotations

from astar_twin.harness.protocol import Strategy
from astar_twin.strategies.filter_baseline.strategy import FilterBaselineStrategy
from astar_twin.strategies.initial_prior.strategy import InitialPriorStrategy
from astar_twin.strategies.mc_oracle.strategy import MCOracleStrategy
from astar_twin.strategies.monte_carlo.strategy import MonteCarlStrategy
from astar_twin.strategies.naive_baseline.strategy import NaiveBaselineStrategy
from astar_twin.strategies.terrain_prior.strategy import TerrainPriorStrategy
from astar_twin.strategies.uniform.strategy import UniformStrategy

REGISTRY: dict[str, type[Strategy]] = {
    "uniform": UniformStrategy,
    "naive_baseline": NaiveBaselineStrategy,
    "initial_prior": InitialPriorStrategy,
    "filter_baseline": FilterBaselineStrategy,
    "terrain_prior": TerrainPriorStrategy,
    "monte_carlo": MonteCarlStrategy,
    "mc_oracle": MCOracleStrategy,
}

__all__ = [
    "REGISTRY",
    "UniformStrategy",
    "NaiveBaselineStrategy",
    "InitialPriorStrategy",
    "FilterBaselineStrategy",
    "TerrainPriorStrategy",
    "MonteCarlStrategy",
    "MCOracleStrategy",
]
