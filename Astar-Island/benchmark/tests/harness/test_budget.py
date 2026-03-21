"""Unit tests for the Budget shared query-pool object."""

from __future__ import annotations

import pytest

from astar_twin.harness.budget import Budget


class TestBudgetInitialization:
    def test_initial_remaining_equals_total(self) -> None:
        budget = Budget(total=50)
        assert budget.remaining == 50

    def test_initial_used_is_zero(self) -> None:
        budget = Budget(total=50)
        assert budget.used == 0

    def test_total_stored(self) -> None:
        budget = Budget(total=10)
        assert budget.total == 10


class TestBudgetConsume:
    def test_consume_decrements_remaining(self) -> None:
        budget = Budget(total=10)
        budget.consume()
        assert budget.remaining == 9
        assert budget.used == 1

    def test_consume_n_decrements_by_n(self) -> None:
        budget = Budget(total=10)
        budget.consume(3)
        assert budget.remaining == 7
        assert budget.used == 3

    def test_consume_exactly_to_zero(self) -> None:
        budget = Budget(total=5)
        for _ in range(5):
            budget.consume()
        assert budget.remaining == 0
        assert budget.used == 5

    def test_consume_whole_budget_at_once(self) -> None:
        budget = Budget(total=5)
        budget.consume(5)
        assert budget.remaining == 0

    def test_consume_beyond_budget_raises_runtime_error(self) -> None:
        budget = Budget(total=3)
        budget.consume(3)
        with pytest.raises(RuntimeError, match="budget exhausted"):
            budget.consume()

    def test_consume_partially_then_overshoot_raises(self) -> None:
        budget = Budget(total=5)
        budget.consume(3)
        with pytest.raises(RuntimeError):
            budget.consume(3)  # only 2 remain

    def test_consume_zero_raises_value_error(self) -> None:
        budget = Budget(total=10)
        with pytest.raises(ValueError, match="n must be >= 1"):
            budget.consume(0)

    def test_consume_negative_raises_value_error(self) -> None:
        budget = Budget(total=10)
        with pytest.raises(ValueError, match="n must be >= 1"):
            budget.consume(-1)


class TestBudgetDunderInt:
    def test_int_returns_remaining(self) -> None:
        budget = Budget(total=10)
        budget.consume(3)
        assert int(budget) == 7

    def test_int_after_full_consumption_is_zero(self) -> None:
        budget = Budget(total=5)
        budget.consume(5)
        assert int(budget) == 0


class TestBudgetRepr:
    def test_repr_contains_key_fields(self) -> None:
        budget = Budget(total=50)
        budget.consume(5)
        r = repr(budget)
        assert "used=5" in r
        assert "total=50" in r
        assert "remaining=45" in r


class TestBudgetMutability:
    def test_same_instance_shared_across_references(self) -> None:
        budget = Budget(total=10)
        ref = budget
        budget.consume(4)
        assert ref.used == 4  # same object — mutation is visible through ref

    def test_two_separate_budgets_are_independent(self) -> None:
        b1 = Budget(total=10)
        b2 = Budget(total=10)
        b1.consume(5)
        assert b2.remaining == 10  # b2 unaffected
