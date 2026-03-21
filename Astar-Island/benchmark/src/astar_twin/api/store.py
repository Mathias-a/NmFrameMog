from __future__ import annotations

from astar_twin.contracts.types import MAX_QUERIES
from astar_twin.data.models import RoundFixture


class RoundStore:
    def __init__(self) -> None:
        self._rounds: dict[str, RoundFixture] = {}

    def add(self, f: RoundFixture) -> None:
        self._rounds[f.id] = f

    def get(self, id: str) -> RoundFixture | None:
        return self._rounds.get(id)

    def list(self) -> list[RoundFixture]:
        return list(self._rounds.values())

    def get_active(self) -> RoundFixture | None:
        for fixture in self._rounds.values():
            if fixture.status == "active":
                return fixture
        return None


class SubmissionStore:
    def __init__(self) -> None:
        self._submissions: dict[tuple[str, int], list[list[list[float]]]] = {}

    def upsert(self, round_id: str, seed_index: int, prediction: list[list[list[float]]]) -> None:
        self._submissions[(round_id, seed_index)] = prediction

    def get(self, round_id: str, seed_index: int) -> list[list[list[float]]] | None:
        return self._submissions.get((round_id, seed_index))


class BudgetStore:
    def __init__(self) -> None:
        self._queries_used: dict[str, int] = {}

    def used(self, round_id: str) -> int:
        return self._queries_used.get(round_id, 0)

    def increment(self, round_id: str) -> None:
        self._queries_used[round_id] = self.used(round_id) + 1

    @property
    def max_queries(self) -> int:
        return MAX_QUERIES
