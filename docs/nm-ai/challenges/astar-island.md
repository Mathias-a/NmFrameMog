# Astar Island

Sources:

- `challenge://astar-island/overview`
- `challenge://astar-island/mechanics`
- `challenge://astar-island/endpoint`
- `challenge://astar-island/scoring`
- `challenge://astar-island/quickstart`

## Live MCP summary

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

- The live endpoint docs now expose the full public/team REST surface:
  - `GET /astar-island/rounds`
  - `GET /astar-island/rounds/{round_id}`
  - `GET /astar-island/budget`
  - `POST /astar-island/simulate`
  - `POST /astar-island/submit`
  - `GET /astar-island/my-rounds`
  - `GET /astar-island/my-predictions/{round_id}`
  - `GET /astar-island/analysis/{round_id}/{seed_index}`
  - `GET /astar-island/leaderboard`
- Authentication is fully documented in the live resources:
  - `access_token` cookie, or
  - `Authorization: Bearer <token>` header
- The simulator endpoint is now fully described:
  - `POST /astar-island/simulate`
  - shared budget of **50 queries per round**
  - viewport size `5–15`
  - response includes `grid`, `settlements`, `viewport`, `queries_used`, `queries_max`
- Submission validation is also explicit in the live docs:
  - prediction tensor must be `H × W × 6`
  - probabilities must sum to `1.0` within tolerance
  - probabilities must be non-negative
- Analysis endpoint excerpt: returns your prediction alongside the ground truth for a specific seed after the round completes.
- Important live inconsistency to preserve: some Astar docs use `seeds_count = 5` and `seed_index 0–4`, while endpoint prose also says “submit all 15 seeds.”

### Scoring

- "Your score is based on entropy-weighted KL divergence between your prediction and the ground truth."
- The page includes **Ground Truth** and **KL Divergence** sections.
- Critical pitfall:

> Never assign probability `0.0` to any class. If ground truth has non-zero mass on a class with zero predicted probability, KL divergence goes to infinity for that cell.

- Recommended mitigation from the docs: enforce a minimum floor such as `0.01` and renormalize.

### Quickstart

- The quickstart includes **Authentication** and **Using the MCP Server**.
- The same zero-probability warning is repeated here, signaling it is one of the highest-value operational rules in the task.
- The live quickstart now also exposes an end-to-end workflow:
  1. get the active round
  2. fetch round details including `initial_states`
  3. query the simulator with viewport coordinates
  4. submit one probability tensor per seed
- The quickstart includes a concrete MCP setup command:

```bash
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp
```

## What matters for implementation

- This is a forecasting and calibration challenge, not a plain classification task.
- A strong solution needs simulation understanding, uncertainty estimation, and careful probability-floor handling.
- The analysis endpoint suggests a tight evaluation loop: submit, inspect ground-truth deltas, recalibrate, repeat.
- The live API docs make this much more operationally concrete than before: there is a strict query budget, explicit submission validation, and a team-scoped set of leaderboard and post-round analysis endpoints.
