from __future__ import annotations

from astar_twin.harness.protocol import Strategy
from astar_twin.strategies.adaptive_entropy_mc.strategy import AdaptiveEntropyMCStrategy
from astar_twin.strategies.active_sensing_ensemble_mc.strategy import (
    ActiveSensingEnsembleMCStrategy,
)
from astar_twin.strategies.ensemble_adaptive_mc.strategy import EnsembleAdaptiveMCStrategy
from astar_twin.strategies.filter_baseline.strategy import FilterBaselineStrategy
from astar_twin.strategies.initial_prior.strategy import InitialPriorStrategy
from astar_twin.strategies.mc_oracle.strategy import MCOracleStrategy
from astar_twin.strategies.reduced_rollout_planner.strategy import ReducedRolloutPlannerStrategy
from astar_twin.strategies.terrain_aware_mc.strategy import TerrainAwareMCStrategy
from astar_twin.strategies.uniform.strategy import UniformStrategy

REGISTRY: dict[str, type[Strategy]] = {
    "uniform": UniformStrategy,
    "initial_prior": InitialPriorStrategy,
    "filter_baseline": FilterBaselineStrategy,
    "mc_oracle": MCOracleStrategy,
    "terrain_aware_mc": TerrainAwareMCStrategy,
    "adaptive_entropy_mc": AdaptiveEntropyMCStrategy,
    "ensemble_adaptive_mc": EnsembleAdaptiveMCStrategy,
    "active_sensing_ensemble_mc": ActiveSensingEnsembleMCStrategy,
    "reduced_rollout_planner": ReducedRolloutPlannerStrategy,
}

__all__ = [
    "REGISTRY",
    "UniformStrategy",
    "InitialPriorStrategy",
    "FilterBaselineStrategy",
    "MCOracleStrategy",
    "TerrainAwareMCStrategy",
    "AdaptiveEntropyMCStrategy",
    "EnsembleAdaptiveMCStrategy",
    "ActiveSensingEnsembleMCStrategy",
    "ReducedRolloutPlannerStrategy",
]
