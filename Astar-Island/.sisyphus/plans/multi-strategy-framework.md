# Multi-Strategy Benchmarking Framework

## TL;DR
> **Summary**: Add a `strategies/` layer and `benchmark/` harness that let multiple prediction approaches be developed and evaluated in parallel against the digital twin, without any strategy being able to corrupt or mutate the simulator core. Also extend `AGENTS.md` with usage guidance and create a `benchmark-twin` skill.
> **Deliverables**: `strategies/` package with Protocol + explicit registry + two starter strategies; `benchmark/` harness with `BenchmarkRunner` + `BenchmarkReport`; `AGENTS.md` new section; `.claude/skills/benchmark-twin/SKILL.md`
> **Effort**: Medium
> **Parallel**: YES — 3 waves
> **Critical Path**: Task 1 (Strategy Protocol) → Task 2 (Registry + Naive Baseline) → Task 3 (Benchmark Harness) → Task 4 (MC Strategy) → Task 5 (Tests) → Task 6 (AGENTS.md) → Task 7 (Skill)

---

## Context

### Original Request
Add an infrastructure layer that lets multiple prediction strategies be developed and benchmarked in parallel, without strategies interfering with each other or the digital twin. Extend AGENTS.md and create a `benchmark-twin` skill that strictly prohibits mutation of the simulator.

### Interview Summary
- **Strategy interface**: `predict(initial_state, budget) -> NDArray[float64]` (H×W×6), fixture-based only
- **Registry**: explicit `dict[str, type[Strategy]]` — no auto-discovery
- **Budget param**: passed as metadata only; strategies do NOT allocate API queries
- **Determinism**: same `base_seed` argument → identical output, required for CI reproducibility
- **v1 scope**: Python API + tests + docs. No CLI entrypoint.
- **Allowed imports in strategy code**: only `astar_twin.engine.Simulator`, `astar_twin.mc.*`, `astar_twin.scoring.*`, `astar_twin.contracts.*`, `astar_twin.params.SimulationParams`, `astar_twin.state.*`
- **Forbidden imports in strategy code**: anything from `astar_twin.phases.*`, `astar_twin.api.*`, `astar_twin.data.*` — strategies must treat the engine as a black box

### Metis Review (gaps addressed)
- **Contract underspecification** → resolved: full `Strategy` Protocol defined with explicit shape, dtype, normalization, seeding requirements
- **Benchmark determinism** → resolved: `base_seed: int` required arg on `BenchmarkRunner.run()`; same seed = reproducible
- **No mutation — enforceable** → resolved: import boundary acceptance criteria + test that imports are clean + AGENTS.md hard rule
- **Malformed predictions** → resolved: harness validates shape + `safe_prediction()` applied before scoring; test for wrong-shape strategies
- **Scope creep risk** → resolved: explicit "Must NOT" list below

---

## Work Objectives

### Core Objective
Provide a stable, non-interfering multi-strategy evaluation layer on top of the existing digital twin.

### Deliverables
1. `benchmark/src/astar_twin/strategies/__init__.py` — exports `Strategy` Protocol + `REGISTRY`
2. `benchmark/src/astar_twin/strategies/base.py` — `Strategy` Protocol definition
3. `benchmark/src/astar_twin/strategies/naive_baseline/` — uniform distribution strategy
4. `benchmark/src/astar_twin/strategies/monte_carlo/` — MC aggregation strategy (wraps `MCRunner`)
5. `benchmark/src/astar_twin/harness/__init__.py`
6. `benchmark/src/astar_twin/harness/runner.py` — `BenchmarkRunner`
7. `benchmark/src/astar_twin/harness/report.py` — `BenchmarkReport`, `SeedResult`
8. `benchmark/tests/strategies/` — tests for protocol + registry + each strategy
9. `benchmark/tests/harness/` — tests for runner + report + determinism
10. `AGENTS.md` — new "## Digital Twin Benchmark" section
11. `.claude/skills/benchmark-twin/SKILL.md` — benchmark skill

### Definition of Done (verifiable)
```bash
# All tests pass
cd benchmark && uv run pytest tests/ -v

# Quality gates pass
cd benchmark && uv run ruff check . && uv run ruff format --check . && uv run mypy

# Naive baseline produces valid H×W×6 tensors
cd benchmark && uv run python -c "
from astar_twin.strategies import REGISTRY
from astar_twin.contracts.api_models import InitialState
# registry has at least 2 entries
assert len(REGISTRY) >= 2
"
```

### Must Have
- `Strategy` Protocol with `predict(initial_state: InitialState, budget: int, base_seed: int) -> NDArray[np.float64]` signature
- `safe_prediction()` applied by harness (not by strategy) before scoring
- Per-seed scores in `BenchmarkReport`
- Aggregate mean score in `BenchmarkReport`
- Existing twin tests still pass unchanged

### Must NOT Have (guardrails)
- No strategy may modify `SimulationParams` field defaults — parameters may only be passed in as constructor args
- No strategy may import from `astar_twin.phases`, `astar_twin.api`, `astar_twin.data`
- No CLI entrypoint (v1)
- No auto-discovery / entry_points plugin system
- No async strategies (sync-only in v1)
- No shared mutable state between strategies in the same `BenchmarkRunner.run()` call
- No fixture loading code inside strategy files

---

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: **tests-after** (implementations driven by spec, then tests confirm)
- Framework: `pytest` in `benchmark/` via `uv run pytest tests/`
- QA policy: every task has executable acceptance criteria + named QA scenarios
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.txt`

---

## Execution Strategy

### Parallel Execution Waves

**Wave 1** (foundation — no dependencies):
- Task 1: `Strategy` Protocol + `base.py`

**Wave 2** (depends on Wave 1):
- Task 2: `strategies/__init__.py` + `REGISTRY` + naive baseline strategy
- Task 3: `harness/report.py` — `BenchmarkReport` + `SeedResult` dataclasses

**Wave 3** (depends on Wave 2):
- Task 4: `monte_carlo` strategy (uses `MCRunner`)
- Task 5: `harness/runner.py` — `BenchmarkRunner`

**Wave 4** (depends on Wave 3):
- Task 6: `tests/strategies/` — protocol + registry + each strategy
- Task 7: `tests/harness/` — runner + report + determinism

**Wave 5** (depends on Wave 4):
- Task 8: `AGENTS.md` benchmark section
- Task 9: `.claude/skills/benchmark-twin/SKILL.md`

### Dependency Matrix
| Task | Blocks | Blocked By |
|------|--------|------------|
| 1: Strategy Protocol | 2, 4 | — |
| 2: Naive Baseline + Registry | 5, 6 | 1 |
| 3: Report dataclasses | 5, 7 | — |
| 4: MC Strategy | 5, 6 | 1 |
| 5: BenchmarkRunner | 6, 7 | 2, 3, 4 |
| 6: Strategy tests | 8 | 2, 4, 5 |
| 7: Harness tests | 8 | 3, 5 |
| 8: AGENTS.md | — | 6, 7 |
| 9: Skill | — | 6, 7 |

### Agent Dispatch Summary
| Wave | Tasks | Category |
|------|-------|----------|
| 1 | 1 | quick |
| 2 | 2, 3 | quick (parallel) |
| 3 | 4, 5 | quick (parallel) |
| 4 | 6, 7 | quick (parallel) |
| 5 | 8, 9 | writing (parallel) |

---

## TODOs

- [ ] 1. Define `Strategy` Protocol in `benchmark/src/astar_twin/strategies/base.py`

  **What to do**:
  1. Create `benchmark/src/astar_twin/strategies/` directory
  2. Create `benchmark/src/astar_twin/strategies/base.py` with:
     ```python
     from __future__ import annotations
     from typing import Protocol
     import numpy as np
     from numpy.typing import NDArray
     from astar_twin.contracts.api_models import InitialState

     class Strategy(Protocol):
         """Prediction strategy contract.

         All strategies MUST:
         - Return NDArray[float64] with shape (H, W, 6) where H, W come from initial_state
         - Return values in [0, 1] per class (harness applies safe_prediction before scoring)
         - Be deterministic given the same base_seed
         - NOT import from astar_twin.phases, astar_twin.api, or astar_twin.data
         - NOT mutate SimulationParams defaults
         """

         @property
         def name(self) -> str: ...

         def predict(
             self,
             initial_state: InitialState,
             budget: int,
             base_seed: int,
         ) -> NDArray[np.float64]: ...
     ```
  3. Create `benchmark/src/astar_twin/strategies/__init__.py` (empty for now — Task 2 fills it)

  **Must NOT do**: Import any phase modules, API modules, or data loaders here. No implementation code.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: single small file, clear spec
  - Skills: [`quality-check`] — run after writing

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 2, 4 | Blocked By: —

  **References**:
  - Pattern: `benchmark/src/astar_twin/generation/interface.py:1-9` — existing Protocol pattern in this codebase
  - Type: `benchmark/src/astar_twin/contracts/api_models.py` — `InitialState` type
  - Type: `benchmark/src/astar_twin/contracts/types.py:73-74` — `DEFAULT_MAP_WIDTH`, `DEFAULT_MAP_HEIGHT`

  **Acceptance Criteria**:
  - [ ] `benchmark/src/astar_twin/strategies/base.py` exists
  - [ ] `from astar_twin.strategies.base import Strategy` imports without error: `cd benchmark && uv run python -c "from astar_twin.strategies.base import Strategy; print('ok')"`
  - [ ] `uv run mypy` passes with no errors in `strategies/base.py`
  - [ ] `uv run ruff check .` passes

  **QA Scenarios**:
  ```
  Scenario: Protocol can be used as type annotation
    Tool: Bash
    Steps: cd benchmark && uv run python -c "from astar_twin.strategies.base import Strategy; import inspect; assert inspect.isclass(Strategy)"
    Expected: exits 0
    Evidence: .sisyphus/evidence/task-1-strategy-protocol.txt

  Scenario: Protocol is a Protocol (not ABC)
    Tool: Bash
    Steps: cd benchmark && uv run python -c "from typing import Protocol; from astar_twin.strategies.base import Strategy; import typing; assert issubclass(Strategy, Protocol) or hasattr(Strategy, '__protocol_attrs__')"
    Expected: exits 0, no AttributeError
    Evidence: .sisyphus/evidence/task-1-strategy-protocol.txt
  ```

  **Commit**: YES | Message: `feat(strategies): add Strategy protocol contract` | Files: `benchmark/src/astar_twin/strategies/`

---

- [ ] 2. Create `strategies/__init__.py` with `REGISTRY` + implement `NaiveBaselineStrategy`

  **What to do**:
  1. Create `benchmark/src/astar_twin/strategies/naive_baseline/` directory
  2. Create `benchmark/src/astar_twin/strategies/naive_baseline/__init__.py`
  3. Create `benchmark/src/astar_twin/strategies/naive_baseline/strategy.py`:
     - Class `NaiveBaselineStrategy` implements `Strategy` protocol
     - `name` property returns `"naive_baseline"`
     - `predict()` returns uniform distribution: `np.full((H, W, 6), 1.0/6.0, dtype=np.float64)` where H, W from `len(initial_state.grid)` and `len(initial_state.grid[0])`
     - Constructor: `def __init__(self) -> None: ...` (no params)
     - No RNG needed (deterministic by definition)
  4. Update `benchmark/src/astar_twin/strategies/__init__.py`:
     ```python
     from astar_twin.strategies.base import Strategy
     from astar_twin.strategies.naive_baseline.strategy import NaiveBaselineStrategy

     REGISTRY: dict[str, type[Strategy]] = {
         "naive_baseline": NaiveBaselineStrategy,
     }

     __all__ = ["Strategy", "REGISTRY", "NaiveBaselineStrategy"]
     ```

  **Must NOT do**: No imports from `astar_twin.phases`, `astar_twin.api`, `astar_twin.data`. No fixture loading.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: trivial implementation
  - Skills: [`quality-check`]

  **Parallelization**: Can Parallel: YES (with Task 3) | Wave 2 | Blocks: 5, 6 | Blocked By: 1

  **References**:
  - Protocol: `benchmark/src/astar_twin/strategies/base.py` (Task 1 output)
  - Shape reference: `benchmark/src/astar_twin/contracts/types.py:73-74` — H=40, W=40
  - Existing pattern: `benchmark/src/astar_twin/mc/aggregate.py:10-11` — how H×W×6 tensors are created
  - Import boundary rule: strategies ONLY use `astar_twin.engine`, `astar_twin.mc`, `astar_twin.scoring`, `astar_twin.contracts`, `astar_twin.params`, `astar_twin.state`

  **Acceptance Criteria**:
  - [ ] `from astar_twin.strategies import REGISTRY` works
  - [ ] `"naive_baseline" in REGISTRY` is True
  - [ ] `NaiveBaselineStrategy().predict(initial_state, budget=50, base_seed=0).shape == (H, W, 6)`
  - [ ] All values in output tensor ≈ 1/6
  - [ ] `uv run mypy` passes

  **QA Scenarios**:
  ```
  Scenario: Naive baseline produces correct shape and values
    Tool: Bash
    Steps: cd benchmark && uv run python -c "
  import numpy as np
  from astar_twin.strategies import REGISTRY, NaiveBaselineStrategy
  from astar_twin.contracts.api_models import InitialState
  from astar_twin.data.loaders import load_fixture
  from pathlib import Path
  fixture = load_fixture(Path('data/rounds/test-round-001'))
  s = NaiveBaselineStrategy()
  pred = s.predict(fixture.initial_states[0], budget=50, base_seed=0)
  assert pred.shape == (40, 40, 6), f'shape={pred.shape}'
  assert np.allclose(pred, 1/6), f'values not uniform'
  print('ok')
  "
    Expected: prints 'ok', exits 0
    Evidence: .sisyphus/evidence/task-2-naive-baseline.txt

  Scenario: REGISTRY has expected keys
    Tool: Bash
    Steps: cd benchmark && uv run python -c "from astar_twin.strategies import REGISTRY; assert 'naive_baseline' in REGISTRY; print(list(REGISTRY.keys()))"
    Expected: prints list containing 'naive_baseline', exits 0
    Evidence: .sisyphus/evidence/task-2-naive-baseline.txt
  ```

  **Commit**: YES | Message: `feat(strategies): add NaiveBaselineStrategy and REGISTRY` | Files: `benchmark/src/astar_twin/strategies/`

---

- [ ] 3. Create `harness/report.py` with `BenchmarkReport` and `SeedResult` dataclasses

  **What to do**:
  1. Create `benchmark/src/astar_twin/harness/` directory
  2. Create `benchmark/src/astar_twin/harness/__init__.py`
  3. Create `benchmark/src/astar_twin/harness/report.py`:
     ```python
     from __future__ import annotations
     from dataclasses import dataclass, field

     @dataclass
     class SeedResult:
         seed_index: int
         score: float | None  # None if ground_truth unavailable
         prediction_shape: tuple[int, int, int]  # (H, W, 6)

     @dataclass
     class StrategyReport:
         strategy_name: str
         seed_results: list[SeedResult] = field(default_factory=list)

         @property
         def mean_score(self) -> float | None:
             scored = [r.score for r in self.seed_results if r.score is not None]
             return sum(scored) / len(scored) if scored else None

         @property
         def scores(self) -> list[float | None]:
             return [r.score for r in self.seed_results]

     @dataclass
     class BenchmarkReport:
         round_id: str
         strategy_reports: list[StrategyReport] = field(default_factory=list)

         def best_strategy(self) -> StrategyReport | None:
             scored = [r for r in self.strategy_reports if r.mean_score is not None]
             return max(scored, key=lambda r: r.mean_score or 0.0) if scored else None
     ```

  **Must NOT do**: No simulation logic here. No numpy. Pure data container.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: pure dataclasses
  - Skills: [`quality-check`]

  **Parallelization**: Can Parallel: YES (with Task 2) | Wave 2 | Blocks: 5, 7 | Blocked By: —

  **References**:
  - Existing dataclass pattern: `benchmark/src/astar_twin/params/simulation_params.py` — `@dataclass` usage style

  **Acceptance Criteria**:
  - [ ] `from astar_twin.harness.report import BenchmarkReport, StrategyReport, SeedResult` imports without error
  - [ ] `SeedResult(seed_index=0, score=55.2, prediction_shape=(40,40,6)).score == 55.2`
  - [ ] `StrategyReport("test", [SeedResult(0, 50.0, (40,40,6)), SeedResult(1, 60.0, (40,40,6))]).mean_score == 55.0`
  - [ ] `uv run mypy` passes

  **QA Scenarios**:
  ```
  Scenario: Report dataclasses instantiate correctly
    Tool: Bash
    Steps: cd benchmark && uv run python -c "
  from astar_twin.harness.report import BenchmarkReport, StrategyReport, SeedResult
  r = StrategyReport('s1', [SeedResult(0, 70.0, (40,40,6)), SeedResult(1, 80.0, (40,40,6))])
  assert r.mean_score == 75.0
  b = BenchmarkReport('r1', [r])
  assert b.best_strategy().strategy_name == 's1'
  print('ok')
  "
    Expected: prints 'ok', exits 0
    Evidence: .sisyphus/evidence/task-3-report.txt

  Scenario: mean_score returns None when no scored seeds
    Tool: Bash
    Steps: cd benchmark && uv run python -c "
  from astar_twin.harness.report import StrategyReport, SeedResult
  r = StrategyReport('s1', [SeedResult(0, None, (40,40,6))])
  assert r.mean_score is None
  print('ok')
  "
    Expected: prints 'ok', exits 0
    Evidence: .sisyphus/evidence/task-3-report.txt
  ```

  **Commit**: YES | Message: `feat(harness): add BenchmarkReport dataclasses` | Files: `benchmark/src/astar_twin/harness/`

---

- [ ] 4. Implement `MonteCarloStrategy` in `strategies/monte_carlo/`

  **What to do**:
  1. Create `benchmark/src/astar_twin/strategies/monte_carlo/` directory
  2. Create `benchmark/src/astar_twin/strategies/monte_carlo/__init__.py`
  3. Create `benchmark/src/astar_twin/strategies/monte_carlo/strategy.py`:
     ```python
     class MonteCarloStrategy:
         """Runs n_runs simulations and aggregates to H×W×6 probability tensor."""

         def __init__(self, n_runs: int = 200, params: SimulationParams | None = None) -> None:
             self._n_runs = n_runs
             self._simulator = Simulator(params=params)
             self._runner = MCRunner(self._simulator)

         @property
         def name(self) -> str:
             return f"monte_carlo_n{self._n_runs}"

         def predict(self, initial_state: InitialState, budget: int, base_seed: int) -> NDArray[np.float64]:
             runs = self._runner.run_batch(initial_state, self._n_runs, base_seed=base_seed)
             H = len(initial_state.grid)
             W = len(initial_state.grid[0])
             return aggregate_runs(runs, H, W)
     ```
  4. Update `benchmark/src/astar_twin/strategies/__init__.py` to add:
     ```python
     from astar_twin.strategies.monte_carlo.strategy import MonteCarloStrategy
     REGISTRY["monte_carlo"] = MonteCarloStrategy
     ```

  **Must NOT do**: Do NOT modify `SimulationParams` default field values. Do NOT import `astar_twin.phases` directly. Do NOT read fixture data from disk.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: wraps existing `MCRunner` + `aggregate_runs`
  - Skills: [`quality-check`]

  **Parallelization**: Can Parallel: YES (with Task 5) | Wave 3 | Blocks: 6 | Blocked By: 1, 2

  **References**:
  - Pattern: `benchmark/src/astar_twin/mc/runner.py:8-17` — `MCRunner.run_batch()` signature
  - Pattern: `benchmark/src/astar_twin/mc/aggregate.py:10-23` — `aggregate_runs()` usage
  - Engine: `benchmark/src/astar_twin/engine/simulator.py:36-38` — `Simulator.__init__(params)` signature
  - Params: `benchmark/src/astar_twin/params/simulation_params.py:25` — `SimulationParams` dataclass
  - Import boundary: only `astar_twin.engine`, `astar_twin.mc`, `astar_twin.contracts`, `astar_twin.params`, `astar_twin.state`

  **Acceptance Criteria**:
  - [ ] `"monte_carlo" in REGISTRY` after import
  - [ ] `MonteCarloStrategy(n_runs=10).predict(initial_state, budget=50, base_seed=0).shape == (40, 40, 6)`
  - [ ] Two calls with `base_seed=42` return identical arrays (determinism)
  - [ ] Two calls with different `base_seed` return different arrays (sensitivity)
  - [ ] `uv run mypy` passes

  **QA Scenarios**:
  ```
  Scenario: MC strategy is deterministic with same seed
    Tool: Bash
    Steps: cd benchmark && uv run python -c "
  import numpy as np
  from astar_twin.strategies.monte_carlo.strategy import MonteCarloStrategy
  from astar_twin.data.loaders import load_fixture
  from pathlib import Path
  fixture = load_fixture(Path('data/rounds/test-round-001'))
  s = MonteCarloStrategy(n_runs=5)
  p1 = s.predict(fixture.initial_states[0], budget=50, base_seed=7)
  p2 = s.predict(fixture.initial_states[0], budget=50, base_seed=7)
  assert np.array_equal(p1, p2), 'not deterministic'
  print('ok')
  "
    Expected: prints 'ok', exits 0
    Evidence: .sisyphus/evidence/task-4-mc-strategy.txt

  Scenario: MC strategy shape is H×W×6
    Tool: Bash
    Steps: cd benchmark && uv run python -c "
  from astar_twin.strategies.monte_carlo.strategy import MonteCarloStrategy
  from astar_twin.data.loaders import load_fixture
  from pathlib import Path
  fixture = load_fixture(Path('data/rounds/test-round-001'))
  s = MonteCarloStrategy(n_runs=3)
  pred = s.predict(fixture.initial_states[0], 50, 0)
  assert pred.shape == (40, 40, 6), f'bad shape: {pred.shape}'
  assert pred.dtype == 'float64'
  print('ok')
  "
    Expected: prints 'ok', exits 0
    Evidence: .sisyphus/evidence/task-4-mc-strategy.txt
  ```

  **Commit**: YES | Message: `feat(strategies): add MonteCarloStrategy` | Files: `benchmark/src/astar_twin/strategies/monte_carlo/`, `benchmark/src/astar_twin/strategies/__init__.py`

---

- [ ] 5. Implement `BenchmarkRunner` in `harness/runner.py`

  **What to do**:
  1. Create `benchmark/src/astar_twin/harness/runner.py`:
     ```python
     @dataclass
     class BenchmarkRunner:
         """Evaluates a list of strategies against a round fixture."""

         fixture: RoundFixture          # from astar_twin.data.models
         base_seed: int = 0

         def run(self, strategies: list[Strategy]) -> BenchmarkReport:
             """Run each strategy against every seed in the fixture.

             For each strategy and each seed_index:
             1. Call strategy.predict(initial_state, budget=MAX_QUERIES, base_seed=self.base_seed)
             2. Apply safe_prediction() to the raw output
             3. If ground_truth available: compute_score(ground_truth, safe_pred)
             4. Append SeedResult to StrategyReport
             5. Append StrategyReport to BenchmarkReport
             """
     ```
  2. Ground truth: only available if `fixture.ground_truths is not None` (aggregate over `n_gt_runs=500` MC runs using `base_seed + 10_000` offset to avoid seed collision with strategies)
  3. `BenchmarkRunner` must be pure — no side effects, no file I/O

  **Must NOT do**: Do NOT call `safe_prediction()` inside strategies — only the harness does this. Do NOT allow strategies to share mutable state across seeds.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: orchestration glue, no novel logic
  - Skills: [`quality-check`]

  **Parallelization**: Can Parallel: YES (with Task 4 if tasks 2+3 done) | Wave 3 | Blocks: 7 | Blocked By: 2, 3

  **References**:
  - Data model: `benchmark/src/astar_twin/data/models.py` — `RoundFixture` type (check for `ground_truths` field)
  - Scoring: `benchmark/src/astar_twin/scoring/kl.py:9-18` — `compute_score(ground_truth, prediction)`
  - Safe pred: `benchmark/src/astar_twin/scoring/safe_prediction.py:7-10` — `safe_prediction(tensor)`
  - Constants: `benchmark/src/astar_twin/contracts/types.py:79` — `MAX_QUERIES = 50`
  - Report: `benchmark/src/astar_twin/harness/report.py` (Task 3 output)
  - Protocol: `benchmark/src/astar_twin/strategies/base.py` (Task 1 output)
  - MC aggregate: `benchmark/src/astar_twin/mc/aggregate.py` — for generating ground truth from MC runs

  **Acceptance Criteria**:
  - [ ] `BenchmarkRunner(fixture=fixture).run([NaiveBaselineStrategy()])` returns a `BenchmarkReport`
  - [ ] Report has one `StrategyReport` with `strategy_name == "naive_baseline"`
  - [ ] Each `StrategyReport.seed_results` has `len == fixture.seeds_count`
  - [ ] If ground truth is available, all `SeedResult.score` values are in [0.0, 100.0]
  - [ ] Same `base_seed` → same scores (deterministic)

  **QA Scenarios**:
  ```
  Scenario: Full end-to-end benchmark with naive baseline
    Tool: Bash
    Steps: cd benchmark && uv run python -c "
  from astar_twin.harness.runner import BenchmarkRunner
  from astar_twin.strategies import NaiveBaselineStrategy
  from astar_twin.data.loaders import load_fixture
  from pathlib import Path
  fixture = load_fixture(Path('data/rounds/test-round-001'))
  runner = BenchmarkRunner(fixture=fixture, base_seed=0)
  report = runner.run([NaiveBaselineStrategy()])
  assert len(report.strategy_reports) == 1
  sr = report.strategy_reports[0]
  assert sr.strategy_name == 'naive_baseline'
  assert len(sr.seed_results) == fixture.seeds_count
  print('mean_score:', sr.mean_score)
  print('ok')
  "
    Expected: prints mean_score value and 'ok', exits 0
    Evidence: .sisyphus/evidence/task-5-runner.txt

  Scenario: Benchmark is deterministic
    Tool: Bash
    Steps: cd benchmark && uv run python -c "
  from astar_twin.harness.runner import BenchmarkRunner
  from astar_twin.strategies import NaiveBaselineStrategy
  from astar_twin.data.loaders import load_fixture
  from pathlib import Path
  fixture = load_fixture(Path('data/rounds/test-round-001'))
  runner = BenchmarkRunner(fixture=fixture, base_seed=42)
  r1 = runner.run([NaiveBaselineStrategy()])
  r2 = runner.run([NaiveBaselineStrategy()])
  s1 = [sr.score for sr in r1.strategy_reports[0].seed_results]
  s2 = [sr.score for sr in r2.strategy_reports[0].seed_results]
  assert s1 == s2, f'{s1} != {s2}'
  print('ok')
  "
    Expected: prints 'ok', exits 0
    Evidence: .sisyphus/evidence/task-5-runner-determinism.txt

  Scenario: Strategy returning wrong shape raises ValueError
    Tool: Bash
    Steps: cd benchmark && uv run python -c "
  import numpy as np
  from astar_twin.harness.runner import BenchmarkRunner
  from astar_twin.strategies.base import Strategy
  from astar_twin.contracts.api_models import InitialState
  from astar_twin.data.loaders import load_fixture
  from pathlib import Path
  class BadStrategy:
      name = 'bad'
      def predict(self, initial_state, budget, base_seed):
          return np.zeros((5, 5, 3))  # wrong shape
  fixture = load_fixture(Path('data/rounds/test-round-001'))
  runner = BenchmarkRunner(fixture=fixture, base_seed=0)
  try:
      runner.run([BadStrategy()])
      print('FAIL: should have raised')
  except ValueError as e:
      print('ok:', e)
  "
    Expected: prints 'ok:' with error message, exits 0
    Evidence: .sisyphus/evidence/task-5-runner-bad-shape.txt
  ```

  **Commit**: YES | Message: `feat(harness): add BenchmarkRunner` | Files: `benchmark/src/astar_twin/harness/runner.py`

---

- [ ] 6. Write `tests/strategies/` — protocol compliance + registry + per-strategy tests

  **What to do**:
  1. Create `benchmark/tests/strategies/__init__.py`
  2. Create `benchmark/tests/strategies/test_protocol.py`:
     - Test: every strategy in `REGISTRY` has a `name` property that returns a non-empty string
     - Test: every strategy in `REGISTRY` is instantiable with no required args
     - Test: calling `strategy.predict(fixture_initial_state, 50, 0)` returns `NDArray[float64]` of shape `(H, W, 6)`
     - Test: all values in prediction are in [0.0, 1.0]
     - Test: no strategy imports from forbidden modules (`astar_twin.phases`, `astar_twin.api`, `astar_twin.data`)
  3. Create `benchmark/tests/strategies/test_naive_baseline.py`:
     - Test: output is exactly uniform (all values == 1/6)
     - Test: deterministic across calls (base_seed ignored by design)
  4. Create `benchmark/tests/strategies/test_monte_carlo.py`:
     - Test: determinism with same `base_seed`
     - Test: variation with different `base_seed`
     - Test: shape correct
     - Test: values in [0.0, 1.0]

  **Must NOT do**: Do NOT import from `astar_twin.phases` or `astar_twin.api` in test files.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: standard pytest test writing
  - Skills: [`quality-check`]

  **Parallelization**: Can Parallel: YES (with Task 7) | Wave 4 | Blocks: 8 | Blocked By: 2, 4, 5

  **References**:
  - Test pattern: `benchmark/tests/mc/test_tensor_shape.py` — shape assertion pattern
  - Test pattern: `benchmark/tests/scoring/test_kl_formula.py` — value assertion pattern
  - Fixture loading: `benchmark/tests/conftest.py` — how fixtures are loaded in existing tests
  - Import boundary check: use `importlib` to inspect strategy module `__file__` + grep imports

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/strategies/ -v` passes with 0 failures
  - [ ] Import boundary test catches a mock strategy that imports `astar_twin.phases`
  - [ ] Determinism test confirms `base_seed=7 → base_seed=7` identical

  **QA Scenarios**:
  ```
  Scenario: All strategy tests pass
    Tool: Bash
    Steps: cd benchmark && uv run pytest tests/strategies/ -v
    Expected: All tests PASSED, 0 failures
    Evidence: .sisyphus/evidence/task-6-strategy-tests.txt

  Scenario: Import boundary enforcement test exists and passes
    Tool: Bash
    Steps: cd benchmark && uv run pytest tests/strategies/test_protocol.py -v -k "import"
    Expected: test_no_forbidden_imports passes for all registry strategies
    Evidence: .sisyphus/evidence/task-6-import-boundary.txt
  ```

  **Commit**: YES | Message: `test(strategies): add protocol compliance and strategy tests` | Files: `benchmark/tests/strategies/`

---

- [ ] 7. Write `tests/harness/` — runner + report + determinism + bad-shape handling

  **What to do**:
  1. Create `benchmark/tests/harness/__init__.py`
  2. Create `benchmark/tests/harness/test_report.py`:
     - Test: `SeedResult` / `StrategyReport` / `BenchmarkReport` instantiate correctly
     - Test: `mean_score` returns `None` when all seeds have `score=None`
     - Test: `best_strategy()` returns strategy with highest mean_score
  3. Create `benchmark/tests/harness/test_runner.py`:
     - Test: `BenchmarkRunner.run([naive])` returns `BenchmarkReport` with correct shape
     - Test: `StrategyReport.seed_results` has exactly `seeds_count` entries
     - Test: determinism: two runs with same `base_seed` produce identical scores
     - Test: `safe_prediction()` is applied — strategy returning `np.zeros(...)` doesn't cause infinity
     - Test: wrong-shape prediction raises `ValueError` with informative message
     - Test: multiple strategies in one `run()` call produces multiple `StrategyReport`s

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: standard pytest tests
  - Skills: [`quality-check`]

  **Parallelization**: Can Parallel: YES (with Task 6) | Wave 4 | Blocks: 8 | Blocked By: 3, 5

  **References**:
  - Test pattern: `benchmark/tests/contracts/test_simulate.py` — FastAPI test client patterns
  - Test pattern: `benchmark/tests/mc/test_tensor_shape.py` — numpy assertion patterns
  - Existing conftest: `benchmark/tests/conftest.py`

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/harness/ -v` passes with 0 failures
  - [ ] Test for zero-floor safety passes (strategy returning all-zeros doesn't crash)
  - [ ] Determinism test passes

  **QA Scenarios**:
  ```
  Scenario: All harness tests pass
    Tool: Bash
    Steps: cd benchmark && uv run pytest tests/harness/ -v
    Expected: All PASSED, 0 failures
    Evidence: .sisyphus/evidence/task-7-harness-tests.txt

  Scenario: Existing twin tests still pass (regression)
    Tool: Bash
    Steps: cd benchmark && uv run pytest tests/ -v --ignore=tests/strategies --ignore=tests/harness
    Expected: All pre-existing tests still pass unchanged
    Evidence: .sisyphus/evidence/task-7-regression.txt
  ```

  **Commit**: YES | Message: `test(harness): add BenchmarkRunner and report tests` | Files: `benchmark/tests/harness/`

---

- [x] 8. Add "## Digital Twin Benchmark" section to `AGENTS.md`

  **What to do**: Append a new top-level section to `Astar-Island/AGENTS.md` (after existing content):

  ```markdown
  ## Digital Twin Benchmark

  ### Overview
  The `benchmark/` directory contains a local digital twin of the remote Astar Island simulator.
  Strategies are evaluated offline against this twin using the `BenchmarkRunner` harness.
  Use it to compare approaches before burning real API budget.

  ### Running the benchmark
  ```bash
  cd Astar-Island/benchmark
  uv run python -c "
  from astar_twin.harness.runner import BenchmarkRunner
  from astar_twin.strategies import REGISTRY
  from astar_twin.data.loaders import load_fixture
  from pathlib import Path

  fixture = load_fixture(Path('data/rounds/test-round-001'))
  strategies = [cls() for cls in REGISTRY.values()]
  report = BenchmarkRunner(fixture=fixture, base_seed=42).run(strategies)

  for sr in report.strategy_reports:
      print(f'{sr.strategy_name}: mean={sr.mean_score:.2f}, per-seed={sr.scores}')
  "
  ```

  ### Adding a new strategy
  1. Create `benchmark/src/astar_twin/strategies/<your_strategy>/strategy.py`
  2. Implement the `Strategy` protocol: `name` property + `predict(initial_state, budget, base_seed) -> NDArray[float64]`
  3. Register it in `benchmark/src/astar_twin/strategies/__init__.py`'s `REGISTRY`
  4. Add tests in `benchmark/tests/strategies/`

  ### ⛔ Hard Rules for Strategy Authors
  - **DO NOT** modify `SimulationParams` default field values
  - **DO NOT** import from `astar_twin.phases`, `astar_twin.api`, or `astar_twin.data`
  - **DO NOT** write to any files in `benchmark/src/astar_twin/engine/`, `benchmark/src/astar_twin/phases/`, or `benchmark/src/astar_twin/mc/`
  - The simulator is a shared, stable black box. Your strategy passes params **in** via constructor; it never changes the engine.
  - Violating these rules corrupts the benchmark for all agents working in this branch.

  ### Strategy contract
  ```python
  class Strategy(Protocol):
      @property
      def name(self) -> str: ...
      def predict(self, initial_state: InitialState, budget: int, base_seed: int) -> NDArray[np.float64]: ...
  ```
  - Output shape: `(H, W, 6)` where H, W from `initial_state.grid`
  - The harness applies `safe_prediction()` before scoring — do NOT floor zeros yourself
  - Deterministic: same `base_seed` MUST produce identical output
  ```

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: documentation editing
  - Skills: []

  **Parallelization**: Can Parallel: YES (with Task 9) | Wave 5 | Blocks: — | Blocked By: 6, 7

  **References**:
  - File: `Astar-Island/AGENTS.md` — existing sections and tone to match
  - Strategy contract: `benchmark/src/astar_twin/strategies/base.py` (Task 1 output)

  **Acceptance Criteria**:
  - [ ] `AGENTS.md` contains a "## Digital Twin Benchmark" section
  - [ ] Section includes the "⛔ Hard Rules" list
  - [ ] Section includes a "Running the benchmark" code block
  - [ ] Section includes "Adding a new strategy" steps
  - [ ] No existing AGENTS.md content is modified or removed

  **QA Scenarios**:
  ```
  Scenario: AGENTS.md contains required benchmark section
    Tool: Bash
    Steps: grep -q "Digital Twin Benchmark" Astar-Island/AGENTS.md && grep -q "Hard Rules" Astar-Island/AGENTS.md && echo ok
    Expected: prints 'ok'
    Evidence: .sisyphus/evidence/task-8-agents-md.txt

  Scenario: Existing AGENTS.md content preserved
    Tool: Bash
    Steps: grep -q "Challenge in One Sentence" Astar-Island/AGENTS.md && grep -q "API Quick Reference" Astar-Island/AGENTS.md && echo ok
    Expected: prints 'ok'
    Evidence: .sisyphus/evidence/task-8-agents-md.txt
  ```

  **Commit**: YES | Message: `docs(agents): add Digital Twin Benchmark section` | Files: `Astar-Island/AGENTS.md`

---

- [x] 9. Create `.claude/skills/benchmark-twin/SKILL.md`

  **What to do**: Create `/Users/mathias/ai-fun/NmFrameMog/worktree-2/.claude/skills/benchmark-twin/SKILL.md` with this exact content:

  ```markdown
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
  ```

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: skill file documentation
  - Skills: []

  **Parallelization**: Can Parallel: YES (with Task 8) | Wave 5 | Blocks: — | Blocked By: 6, 7

  **References**:
  - Pattern: `/Users/mathias/ai-fun/NmFrameMog/worktree-2/.claude/skills/quality-check/SKILL.md` — skill front-matter format
  - Pattern: `/Users/mathias/ai-fun/NmFrameMog/worktree-2/.claude/skills/solve-challenge/SKILL.md` — workflow structure

  **Acceptance Criteria**:
  - [ ] File exists at `.claude/skills/benchmark-twin/SKILL.md`
  - [ ] Front-matter has `name`, `description`, `allowed-tools` fields
  - [ ] Contains "⛔ ABSOLUTE PROHIBITIONS" section
  - [ ] Contains "✅ Allowed actions" section
  - [ ] Contains "Verification checklist" with runnable bash commands
  - [ ] PROHIBITIONS list explicitly names `simulation_params.py`, `engine/`, `phases/`, `mc/`

  **QA Scenarios**:
  ```
  Scenario: Skill file exists with required sections
    Tool: Bash
    Steps: grep -q "ABSOLUTE PROHIBITIONS" /Users/mathias/ai-fun/NmFrameMog/worktree-2/.claude/skills/benchmark-twin/SKILL.md && grep -q "Allowed actions" /Users/mathias/ai-fun/NmFrameMog/worktree-2/.claude/skills/benchmark-twin/SKILL.md && echo ok
    Expected: prints 'ok'
    Evidence: .sisyphus/evidence/task-9-skill.txt
  ```

  **Commit**: YES | Message: `docs(skills): add benchmark-twin skill` | Files: `.claude/skills/benchmark-twin/SKILL.md`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback → fix → re-run → present again → wait for okay.

- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
  - `cd benchmark && uv run ruff check . && uv run ruff format --check . && uv run mypy`
- [ ] F3. Real Manual QA — unspecified-high
  - `cd benchmark && uv run pytest tests/ -v` (all tests including pre-existing)
  - End-to-end benchmark with both strategies against test-round-001
- [ ] F4. Scope Fidelity Check — deep
  - Verify no strategy imports forbidden modules
  - Verify `SimulationParams` defaults unchanged
  - Verify existing twin test suite passes unmodified

---

## Commit Strategy
1. Task 1: `feat(strategies): add Strategy protocol contract`
2. Task 2: `feat(strategies): add NaiveBaselineStrategy and REGISTRY`
3. Task 3: `feat(harness): add BenchmarkReport dataclasses`
4. Task 4: `feat(strategies): add MonteCarloStrategy`
5. Task 5: `feat(harness): add BenchmarkRunner`
6. Task 6: `test(strategies): add protocol compliance and strategy tests`
7. Task 7: `test(harness): add BenchmarkRunner and report tests`
8. Task 8: `docs(agents): add Digital Twin Benchmark section`
9. Task 9: `docs(skills): add benchmark-twin skill`

---

## Success Criteria
- `uv run pytest tests/ -v` — 0 failures, pre-existing tests unchanged
- `uv run ruff check . && uv run ruff format --check . && uv run mypy` — all pass
- `REGISTRY` contains at least `"naive_baseline"` and `"monte_carlo"`
- End-to-end benchmark produces `BenchmarkReport` with per-seed scores for all fixture seeds
- Two runs with same `base_seed` produce identical `BenchmarkReport`
- `AGENTS.md` contains "Digital Twin Benchmark" section with Hard Rules
- `.claude/skills/benchmark-twin/SKILL.md` exists with PROHIBITIONS and ALLOWED sections
- No engine, phases, or mc source files are modified
