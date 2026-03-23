from __future__ import annotations

from astar_twin.harness.protocol import Strategy
from astar_twin.strategies.filter_baseline.strategy import FilterBaselineStrategy
from astar_twin.strategies.initial_prior.strategy import InitialPriorStrategy
from astar_twin.strategies.mc_oracle.strategy import MCOracleStrategy
from astar_twin.strategies.uniform.strategy import UniformStrategy
from astar_twin.strategies.wrong.strategy import WrongStrategy

REGISTRY: dict[str, type[Strategy]] = {
    "uniform": UniformStrategy,
    "initial_prior": InitialPriorStrategy,
    "filter_baseline": FilterBaselineStrategy,
    "mc_oracle": MCOracleStrategy,
    "wrong": WrongStrategy,
}

__all__ = [
    "REGISTRY",
    "UniformStrategy",
    "InitialPriorStrategy",
    "FilterBaselineStrategy",
    "MCOracleStrategy",
    "WrongStrategy",
]
