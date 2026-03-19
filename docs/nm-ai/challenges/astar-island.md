# Astar Island

Sources:

- `challenge://astar-island/overview`
- `challenge://astar-island/mechanics`
- `challenge://astar-island/endpoint`
- `challenge://astar-island/scoring`
- `challenge://astar-island/quickstart`

## MCP excerpts

### Overview

- "Astar Island is a machine learning challenge where you observe a black-box Norse civilisation simulator through a limited viewport and predict the final world state."
- The simulator runs a procedurally generated Norse world for 50 years.
- Goal excerpt: "predict the probability distribution of terrain types across the entire map."
- "Task type: Observation + probabilistic prediction"
- Workflow excerpt:
  5. Submit predictions — for each seed, submit a `W×H×6` probability tensor predicting terrain type probabilities per cell.
  6. Scoring — predictions are compared against the ground truth using entropy-weighted KL divergence.

### Mechanics

- The mechanics docs are titled **Astar Island Simulation Mechanics**.
- "The world is a rectangular grid (default 40×40) with 8 terrain types that map to 6 prediction classes."
- The page includes a terrain table mapping internal code, terrain, class index, and description.

### Endpoint

- The endpoint docs expose at least these routes:
  - `GET /astar-island/my-predictions/{round_id}`
  - `GET /astar-island/analysis/{round_id}/{seed_index}`
  - `GET /astar-island/leaderboard`
- Analysis endpoint excerpt: returns your prediction alongside the ground truth for a specific seed after the round completes.

### Scoring

- "Your score is based on entropy-weighted KL divergence between your prediction and the ground truth."
- The page includes **Ground Truth** and **KL Divergence** sections.
- Critical pitfall:

> Never assign probability `0.0` to any class. If ground truth has non-zero mass on a class with zero predicted probability, KL divergence goes to infinity for that cell.

- Recommended mitigation from the docs: enforce a minimum floor such as `0.01` and renormalize.

### Quickstart

- The quickstart includes **Authentication** and **Using the MCP Server**.
- The same zero-probability warning is repeated here, signaling it is one of the highest-value operational rules in the task.

## What matters for implementation

- This is a forecasting and calibration challenge, not a plain classification task.
- A strong solution needs simulation understanding, uncertainty estimation, and careful probability-floor handling.
- The analysis endpoint suggests a tight evaluation loop: submit, inspect ground-truth deltas, recalibrate, repeat.
