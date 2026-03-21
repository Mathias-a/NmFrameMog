---
name: benchmark-twin
description: Evaluate a prediction strategy against the Astar Island digital twin. Enforces strict rules against mutating the simulator.
allowed-tools: Bash
context: fork
---

Evaluate the strategy described in: $ARGUMENTS

## Setup

All benchmark work happens inside `Astar-Island/benchmark/`:

```bash
cd Astar-Island/benchmark
```

## Running a full benchmark

```bash
uv run python -c "
from astar_twin.harness.runner import BenchmarkRunner
from astar_twin.strategies import REGISTRY
from astar_twin.data.loaders import load_fixture
from pathlib import Path

fixture = load_fixture(Path('data/rounds/test-round-001'))
strategies = [cls() for cls in REGISTRY.values()]
report = BenchmarkRunner(fixture=fixture, base_seed=42).run(strategies)

for sr in report.strategy_reports:
    print(f'{sr.strategy_name}: mean={sr.mean_score:.2f}, per_seed={sr.scores}')
best = report.best_strategy()
if best:
    print(f'Best: {best.strategy_name} @ {best.mean_score:.2f}')
"
```

## Adding a new strategy

1. Create `src/astar_twin/strategies/<name>/strategy.py` implementing the `Strategy` protocol
2. Register it in `src/astar_twin/strategies/__init__.py` under `REGISTRY`
3. Add tests in `tests/strategies/`
4. Run quality checks (see `quality-check` skill)

## ⛔ ABSOLUTE PROHIBITIONS

You are NOT allowed to:
- Modify `SimulationParams` default field values in `params/simulation_params.py`
- Edit ANY file in `src/astar_twin/engine/`, `src/astar_twin/phases/`, or `src/astar_twin/mc/`
- Import from `astar_twin.phases`, `astar_twin.api`, or `astar_twin.data` inside strategy files
- Remove or alter invariant tests in `tests/invariants/`
- Alter the `safe_prediction()` function or `compute_score()` function
- Change the ground truth generation logic in `BenchmarkRunner`

**Rationale**: The digital twin is a shared evaluation oracle. Any modification to its behavior invalidates comparisons between strategies developed by different agents.

## ✅ Allowed actions

- Create new strategies in `src/astar_twin/strategies/<name>/`
- Pass custom `SimulationParams` instances to `Simulator(params=...)` inside your strategy constructor
- Add new fixtures to `data/rounds/`
- Add tests in `tests/strategies/` and `tests/harness/`
- Tune `n_runs` and `base_seed` arguments to strategies and runner

## Verification checklist

Before completing, verify:

```bash
# All tests pass
uv run pytest tests/ -v

# Quality gates
uv run ruff check .
uv run ruff format --check .
uv run mypy

# Benchmark runs end-to-end without error
uv run python -c "
from astar_twin.harness.runner import BenchmarkRunner
from astar_twin.strategies import REGISTRY
from astar_twin.data.loaders import load_fixture
from pathlib import Path
fixture = load_fixture(Path('data/rounds/test-round-001'))
report = BenchmarkRunner(fixture=fixture, base_seed=0).run([cls() for cls in REGISTRY.values()])
print('Benchmark complete:', [(r.strategy_name, r.mean_score) for r in report.strategy_reports])
"
```
