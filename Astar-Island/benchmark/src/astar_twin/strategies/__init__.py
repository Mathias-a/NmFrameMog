from __future__ import annotations

from astar_twin.harness.protocol import Strategy
from astar_twin.strategies.adaptive_entropy_mc.strategy import AdaptiveEntropyMCStrategy
from astar_twin.strategies.filter_baseline.strategy import FilterBaselineStrategy
from astar_twin.strategies.initial_prior.strategy import InitialPriorStrategy
from astar_twin.strategies.mc_oracle.strategy import MCOracleStrategy
from astar_twin.strategies.terrain_aware_mc.strategy import TerrainAwareMCStrategy
from astar_twin.strategies.uniform.strategy import UniformStrategy

REGISTRY: dict[str, type[Strategy]] = {
    "uniform": UniformStrategy,
    "initial_prior": InitialPriorStrategy,
    "filter_baseline": FilterBaselineStrategy,
    "mc_oracle": MCOracleStrategy,
    "terrain_aware_mc": TerrainAwareMCStrategy,
    "adaptive_entropy_mc": AdaptiveEntropyMCStrategy,
}

__all__ = [
    "REGISTRY",
    "UniformStrategy",
    "InitialPriorStrategy",
    "FilterBaselineStrategy",
    "MCOracleStrategy",
    "TerrainAwareMCStrategy",
    "AdaptiveEntropyMCStrategy",
]
