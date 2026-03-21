from __future__ import annotations

import copy
from dataclasses import dataclass


@dataclass
class Settlement:
    x: int
    y: int
    owner_id: int
    alive: bool = True
    has_port: bool = False
    has_longship: bool = False
    population: float = 1.0
    food: float = 0.5
    wealth: float = 0.3
    defense: float = 0.2
    tech: float = 0.0
    raid_stress: float = 0.0
    ruin_age: int = 0
    port_qualifying_years: int = 0
    longship_qualifying_years: int = 0

    def copy(self) -> Settlement:
        return copy.copy(self)

    def is_ruin(self) -> bool:
        return not self.alive

    def prosperity(self, tech_economic_bonus: float) -> float:
        food_pc = self.food / max(self.population, 0.5)
        return food_pc + self.wealth + tech_economic_bonus * self.tech
