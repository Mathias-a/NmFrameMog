# Astar Island — Agent Briefing

## Challenge in One Sentence

Observe a stochastic Norse civilisation simulator through limited viewports (50 queries, 15x15 max) and predict the probability distribution of 6 terrain classes across the full 40x40 map for each of 5 seeds.

## What We're Predicting

A **H x W x 6 probability tensor** per seed. Each cell gets probabilities for: Empty(0), Settlement(1), Port(2), Ruin(3), Forest(4), Mountain(5). Scored via entropy-weighted KL divergence. Score = 100 * exp(-3 * weighted_kl), range [0, 100].

## Core Constraints

- **50 queries total** per round, shared across all 5 seeds
- **Viewport**: 5-15 cells wide/tall per query
- **Stochastic**: same map + params produce different outcomes each run
- **Hidden parameters**: control world behavior, same for all seeds in a round
- **Map seed visible**: initial terrain layout is reconstructable
- **Time window**: ~2h45m per round

## Critical Rules

- **NEVER output probability 0.0** — floor at 0.01, renormalize. One zero = infinite KL = destroyed score.
- **Always submit all 5 seeds** — missing seed scores 0.
- Uniform baseline scores ~1-5. Any model beats that.

## Strategy Levers

| Lever | Why It Matters |
|-------|---------------|
| Query allocation | 50 queries across 5 seeds — which areas/seeds to observe? |
| Viewport placement | Cover dynamic regions (settlements), skip static (ocean, mountains) |
| Monte Carlo aggregation | Multiple queries on same seed = empirical distribution |
| Cross-seed transfer | Same hidden params across seeds — learn rules once, apply to all |
| Digital twin | Simulate locally to generate unlimited samples without burning queries |

## The Winning Path

1. **Reverse-engineer the simulation rules** from observations
2. **Build a local digital twin** that approximates the real simulator
3. **Run thousands of local Monte Carlo sims** to generate probability tensors
4. **Use API queries strategically** to calibrate/validate the twin, not as primary data source

## Agent Roles

| Agent | Task |
|-------|------|
| **researcher** | Analyze observations, classify simulation mechanics, identify hidden parameter effects |
| **architect** | Design digital twin data model, Monte Carlo pipeline, query strategy |
| **implementer** | Build twin simulator, API client, prediction pipeline (type-safe, quality-gated) |
| **tester** | Validate twin against real API observations, edge cases, scoring math |
| **reviewer** | Verify output format (H x W x 6), probability constraints, KL safety |
| **optimizer** | Tune twin parameters, query allocation strategy, runtime performance |
| **debugger** | Root-cause when twin diverges from observations |

## Key Data Structures

```
Initial state (visible):  grid[H][W] of terrain codes, settlements[{x, y, has_port, alive}]
Sim response (viewport):  grid[vh][vw], settlements[{x, y, pop, food, wealth, defense, has_port, alive, owner_id}]
Prediction (submit):       prediction[H][W][6] — probabilities summing to 1.0 per cell
```

## API Quick Reference

| Endpoint | Purpose | Auth |
|----------|---------|------|
| `GET /astar-island/rounds` | List rounds, find active | Public |
| `GET /astar-island/rounds/{id}` | Initial states for all seeds | Public |
| `GET /astar-island/budget` | Remaining queries | Team |
| `POST /astar-island/simulate` | Observe one viewport (costs 1 query) | Team |
| `POST /astar-island/submit` | Submit H x W x 6 tensor for one seed | Team |
| `GET /astar-island/analysis/{round_id}/{seed}` | Post-round ground truth (after completion) | Team |

## Digital Twin Benchmark

### Overview
The `benchmark/` directory contains a local digital twin of the remote Astar Island simulator.
Strategies are evaluated offline against this twin using the `BenchmarkRunner` harness.
Use it to compare approaches before burning real API budget.

### Running the benchmark

Use the most recent real round (round 18, 40×40 map with real ground truths from the API):

```bash
cd Astar-Island/benchmark
uv run python -c "
from astar_twin.harness.runner import BenchmarkRunner
from astar_twin.strategies import REGISTRY
from astar_twin.data.loaders import load_fixture
from pathlib import Path

fixture = load_fixture(Path('data/rounds/b0f9d1bf-4b71-4e6e-816c-19c718d29056'))
strategies = [cls() for cls in REGISTRY.values()]
report = BenchmarkRunner(fixture=fixture, base_seed=42).run(strategies)

for sr in report.strategy_reports:
    print(f'{sr.strategy_name}: mean={sr.mean_score:.2f}, per-seed={sr.scores}')
"
```

All 18 completed real rounds (rounds 1–18, each 40×40) are available under
`benchmark/data/rounds/`. The synthetic `test-round-001` (10×10) remains for
unit tests but should not be used as a benchmark reference.

To refresh the local fixture database (e.g. after new rounds complete):
```bash
cd Astar-Island/benchmark
uv run python scripts/fetch_real_rounds.py
```

Use `--prior-spread <float>` when local fallback ground truths must be derived
from `DEFAULT_PRIOR` parameters and you want a narrower or wider hyperparameter
range than the default `1.0` spread.

If you regenerate local ground truths or benchmark uncached fixtures, keep the
same prior-spread setting across fixture prep and on-demand benchmark
evaluation so scores stay comparable.

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
