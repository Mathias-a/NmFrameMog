from __future__ import annotations

from astar_twin.harness.protocol import Strategy
from astar_twin.strategies.filter_baseline.strategy import FilterBaselineStrategy
from astar_twin.strategies.hybrid_solver.strategy import HybridSolverStrategy
from astar_twin.strategies.initial_prior.strategy import InitialPriorStrategy
from astar_twin.strategies.mc_oracle.strategy import MCOracleStrategy
from astar_twin.strategies.smc_particle_filter.strategy import SMCParticleFilterStrategy
from astar_twin.strategies.uniform.strategy import UniformStrategy

# REGISTRY contains only strategies with zero-arg constructors.
# HybridSolverStrategy requires a RoundFixture and must be instantiated manually.
REGISTRY: dict[str, type[Strategy]] = {
    "uniform": UniformStrategy,
    "initial_prior": InitialPriorStrategy,
    "filter_baseline": FilterBaselineStrategy,
    "mc_oracle": MCOracleStrategy,
    "smc_particle_filter": SMCParticleFilterStrategy,
}

__all__ = [
    "REGISTRY",
    "UniformStrategy",
    "InitialPriorStrategy",
    "FilterBaselineStrategy",
    "MCOracleStrategy",
    "SMCParticleFilterStrategy",
    "HybridSolverStrategy",
]
