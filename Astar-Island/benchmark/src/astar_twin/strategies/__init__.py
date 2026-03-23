from __future__ import annotations

from astar_twin.harness.protocol import Strategy
from astar_twin.strategies.filter_baseline.strategy import FilterBaselineStrategy
from astar_twin.strategies.ids_bandit.strategy import IDSBanditStrategy
from astar_twin.strategies.initial_prior.strategy import InitialPriorStrategy
from astar_twin.strategies.mc_challenger.strategy import MCChallengerStrategy
from astar_twin.strategies.mc_oracle.strategy import MCOracleStrategy
from astar_twin.strategies.spatial_heuristic.strategy import SpatialHeuristicStrategy
from astar_twin.strategies.spatial_heuristic_v2.strategy import SpatialHeuristicV2Strategy
from astar_twin.strategies.uniform.strategy import UniformStrategy

REGISTRY: dict[str, type[Strategy]] = {
    "uniform": UniformStrategy,
    "initial_prior": InitialPriorStrategy,
    "filter_baseline": FilterBaselineStrategy,
    "mc_oracle": MCOracleStrategy,
    "mc_challenger": MCChallengerStrategy,
    "spatial_heuristic": SpatialHeuristicStrategy,
    "spatial_heuristic_v2": SpatialHeuristicV2Strategy,
    "ids_bandit": IDSBanditStrategy,
}

__all__ = [
    "REGISTRY",
    "UniformStrategy",
    "InitialPriorStrategy",
    "FilterBaselineStrategy",
    "MCOracleStrategy",
    "MCChallengerStrategy",
    "SpatialHeuristicStrategy",
    "SpatialHeuristicV2Strategy",
    "IDSBanditStrategy",
]
