from __future__ import annotations

from numpy.random import Generator, default_rng

from astar_twin.contracts.api_models import InitialState
from astar_twin.contracts.types import SIM_YEARS, TerrainCode
from astar_twin.engine.invariants import check_invariants
from astar_twin.params.simulation_params import SimulationParams
from astar_twin.phases import (
    apply_conflict,
    apply_environment,
    apply_growth,
    apply_trade,
    apply_winter,
)
from astar_twin.state.grid import Grid
from astar_twin.state.round_state import RoundState
from astar_twin.state.settlement import Settlement


def _sync_settlement_cells(state: RoundState) -> RoundState:
    grid = state.grid.copy()
    for settlement in state.settlements:
        if not settlement.alive:
            continue
        grid.set(
            settlement.y,
            settlement.x,
            TerrainCode.PORT if settlement.has_port else TerrainCode.SETTLEMENT,
        )
    result = state.copy()
    result.grid = grid
    return result


class Simulator:
    def __init__(self, params: SimulationParams | None = None) -> None:
        self.params = params or SimulationParams()

    def init_state(self, initial_state: InitialState, rng: Generator) -> RoundState:
        grid = Grid.from_list(initial_state.grid)
        settlements: list[Settlement] = []

        for owner_id, initial_settlement in enumerate(initial_state.settlements):
            population = max(
                0.1,
                float(
                    rng.lognormal(
                        mean=self.params.init_population_mean,
                        sigma=self.params.init_population_sigma,
                    )
                ),
            )
            food = max(
                0.0,
                float(
                    rng.normal(
                        loc=population * self.params.init_food_per_pop,
                        scale=self.params.init_food_sigma,
                    )
                ),
            )
            wealth = max(
                0.0,
                float(
                    rng.normal(
                        loc=self.params.init_wealth_mean
                        + (
                            self.params.init_port_wealth_bonus
                            if initial_settlement.has_port
                            else 0.0
                        ),
                        scale=self.params.init_wealth_sigma,
                    )
                ),
            )
            defense = max(
                0.0,
                float(
                    rng.normal(
                        loc=population * self.params.init_defense_per_pop,
                        scale=self.params.init_defense_sigma,
                    )
                ),
            )
            tech = self.params.init_tech_base + (
                self.params.init_port_tech_bonus if initial_settlement.has_port else 0.0
            )

            settlements.append(
                Settlement(
                    x=initial_settlement.x,
                    y=initial_settlement.y,
                    owner_id=owner_id,
                    alive=initial_settlement.alive,
                    has_port=initial_settlement.has_port,
                    population=population,
                    food=food,
                    wealth=wealth,
                    defense=defense,
                    tech=tech,
                )
            )
            grid.set(
                initial_settlement.y,
                initial_settlement.x,
                TerrainCode.PORT if initial_settlement.has_port else TerrainCode.SETTLEMENT,
            )

        return RoundState(grid=grid, settlements=settlements, year=0)

    def run(self, initial_state: InitialState, sim_seed: int) -> RoundState:
        rng = default_rng(sim_seed)
        state = self.init_state(initial_state, rng)
        initial_grid = state.grid.copy()
        war_registry: dict[tuple[int, int], int] = {}
        prev_severity = 0.0

        for year in range(SIM_YEARS):
            state = apply_growth(state, self.params, rng)
            state = apply_conflict(state, self.params, rng, war_registry, year)
            state = apply_trade(state, self.params, rng, war_registry, year)
            state, prev_severity = apply_winter(state, self.params, rng, prev_severity)
            state = apply_environment(state, self.params, rng)
            state = _sync_settlement_cells(state)
            state.year = year + 1
            check_invariants(state, initial_grid)

        return state
