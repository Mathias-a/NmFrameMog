# Astar Island: Backtesting & Iteration Workflow

## Overview

After each round completes, we can fetch the real ground truth and backtest
any prediction strategy locally. This is the fastest way to improve scores.

**Loop:** submit prediction → round completes → fetch GT → backtest → improve → submit again

## 1. Setup

### API Token

Get your JWT token from [app.ainm.no](https://app.ainm.no) (browser cookies or auth flow):

```bash
export ASTAR_API_TOKEN="eyJhbG..."
```

### Quick Start

```python
from astar_island.backtesting import AstarClient, fetch_and_freeze_all

# Fetch all available ground truth
with AstarClient(token="your-jwt-here") as client:
    saved = fetch_and_freeze_all(client)
    print(f"Saved {len(saved)} fixtures")
```

## 2. Fetching Ground Truth

After a round's status changes to `completed`, the `/analysis` endpoint
returns the real ground truth for each seed.

```python
from astar_island.backtesting import AstarClient

with AstarClient() as client:
    # See all rounds
    rounds = client.get_rounds()
    for r in rounds:
        print(f"Round {r.round_number}: {r.status}")

    # Fetch GT for a specific seed
    analysis = client.get_analysis("round-uuid", seed_index=0)
    print(f"Official score: {analysis.official_score}")
    print(f"GT shape: {analysis.ground_truth.shape}")
```

Fixtures are saved as JSON in `apps/astar-island/data/fixtures/`.

## 3. Running Backtests

A `PredictionStrategy` is any function that takes an initial grid and
returns a prediction tensor:

```python
import numpy as np
from astar_island.prediction import PredictionTensor, make_uniform_prediction

# Simplest strategy: uniform distribution
def uniform_strategy(
    initial_grid: list[list[int]], width: int, height: int
) -> PredictionTensor:
    return make_uniform_prediction(width, height)
```

Run it against all saved fixtures:

```python
from astar_island.backtesting import backtest_strategy

summary = backtest_strategy(uniform_strategy)
print(f"Mean score: {summary.mean_local_score:.2f}/100")
print(f"Mean delta vs official: {summary.mean_score_delta:.4f}")

for r in summary.results:
    print(f"  Round {r.round_id} seed {r.seed_index}: "
          f"local={r.local_score:.2f} official={r.official_score:.2f} "
          f"delta={r.score_delta:.4f}")
```

## 4. Interpreting Results

| Metric | What it means |
|--------|---------------|
| `local_score` | Our `competition_score()` against real GT |
| `official_score` | What the server computed when we submitted |
| `score_delta` | `local - official`. Should be ~0 if our scorer is correct |
| `weighted_kl` | Raw KL divergence (lower = better prediction) |
| `mean_local_score` | Average across all fixtures |

**If `score_delta` is consistently non-zero**, our scoring formula may differ
from the server's. Investigate the constant (`-3` in `exp(-3 * wkl)`),
log base, or entropy calculation.

## 5. Using the Simulator as a Strategy

The local simulator can generate predictions via ensemble:

```python
from astar_island.simulator import run_ensemble, SimConfig

def simulator_strategy(
    initial_grid: list[list[int]], width: int, height: int
) -> PredictionTensor:
    # TODO: use initial_grid to set the map seed or initial state
    return run_ensemble(seed=42, n_runs=100)

summary = backtest_strategy(simulator_strategy)
```

## 6. Calibrating the Simulator

Use `dataclasses.replace()` to sweep parameters:

```python
from dataclasses import replace
from astar_island.simulator import SimConfig, run_ensemble
from astar_island.scoring import competition_score
from astar_island.backtesting import load_fixture

fixture = load_fixture("round-uuid", 0)
gt = fixture.ground_truth

# Try different winter severity
for severity in [0.3, 0.5, 0.7, 0.9]:
    config = replace(SimConfig(), winter_severity_mean=severity)
    pred = run_ensemble(seed=42, config=config, n_runs=50)
    score = competition_score(gt, pred)
    print(f"severity={severity:.1f} → score={score:.2f}")
```

## 7. Adding New Strategies

1. Write a function matching `PredictionStrategy` signature
2. Run `backtest_strategy(your_strategy)`
3. Compare `mean_local_score` against the uniform baseline
4. If it beats uniform → worth submitting

Strategies to try:
- **Static-aware**: predict 1.0 for the initial terrain class on static cells (ocean/mountain/forest), uniform on dynamic cells
- **Settlement heuristic**: settlements near forest/coast more likely to survive
- **Simulator ensemble**: run the local sim N times
- **Observation-based**: use viewport data from `/simulate` to build empirical distributions
