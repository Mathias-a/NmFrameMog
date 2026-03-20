# Astar Island baseline solver

This package implements a cache-first, stdlib-only baseline for the NM i AI Astar Island challenge. It is organized around a legal first submission path: parse round detail, build a validated `prediction[y][x][class]` tensor, save artifacts, optionally execute live queries, and optionally submit.

## What is implemented

- canonical 8-to-6 terrain mapping artifact
- challenge state container per seed
- cache-first filesystem persistence for rounds, queries, predictions, debug artifacts, and run summaries
- baseline prior over the `W×H×6` tensor with a hard `0.01` floor and per-cell renormalization
- query planner that ranks `5–15 × 5–15` viewports by entropy and unresolved coverage
- stochastic rollout aggregation to convert the current belief state into a submission tensor
- strict tensor validation and entropy-weighted KL scoring helper
- static debug visualizer that saves:
  - `start-state.svg`
  - `query-overlay.svg`
  - `screenshots/*.svg`
  - `index.html`

## Package layout

- `solver/contract.py` — verified mechanics and canonical terrain mapping
- `solver/models.py` — round, query, seed-state, and run-summary dataclasses
- `solver/cache.py` — local artifact and cache layout
- `solver/baseline.py` — baseline prior and observation assimilation
- `solver/planner.py` — viewport ranking by entropy and unresolved coverage
- `solver/rollouts.py` — Monte Carlo aggregation over the current belief tensor
- `solver/validator.py` — hard validation gate plus entropy-weighted KL helper
- `solver/api.py` — stdlib HTTP client for the live API
- `solver/pipeline.py` — end-to-end orchestration and debug artifact generation
- `solver/debug_visualization.py` — local HTML/SVG debugger and saved viewport screenshots
- `cli.py` — command-line entrypoints

## Verified contract used as ground truth

- 5 seeds per round
- 50 total queries per round, shared across seeds
- viewport width/height must stay between `5` and `15`
- prediction tensor shape is `prediction[y][x][class]`
- 8 internal terrain codes map to 6 classes
- scoring uses entropy-weighted KL divergence
- do not emit `0.0`; apply a `0.01` floor and renormalize
- analysis is available post-round

## Assumptions intentionally kept config-driven

- hidden settlement dynamics and parameter values
- the exact simulator rates behind growth, raids, trade, winter severity, and reclamation
- any future changes to round timing or endpoint payload details beyond the verified fields used here

## Commands

### 1. Solve a round from a local round detail file

```bash
python -m nmframemog.astar_island solve-round \
  --round-detail-file round-detail.json \
  --cache-dir .artifacts/astar-island
```

This will:

- save the round detail and mapping artifact
- build baseline tensors for each seed
- plan viewports for each seed
- run stochastic aggregation
- validate each prediction tensor
- save a debug bundle for each seed, including viewport screenshots
- write a run summary

### 2. Fetch a round live and optionally execute queries

```bash
export AINM_ACCESS_TOKEN=...

python -m nmframemog.astar_island solve-round \
  --round-id your-round-id \
  --execute-live-queries \
  --cache-dir .artifacts/astar-island
```

Live query responses are cached locally so the same viewports can be replayed offline.

### 3. Submit generated predictions

```bash
export AINM_ACCESS_TOKEN=...

python -m nmframemog.astar_island solve-round \
  --round-id your-round-id \
  --execute-live-queries \
  --submit
```

### 4. Validate a saved prediction

```bash
python -m nmframemog.astar_island validate-prediction \
  .artifacts/astar-island/predictions/<run-id>/seed-00.json
```

### 5. Render debug artifacts from a trace file directly

```bash
python -m nmframemog.astar_island render-debug \
  --input src/nmframemog/astar_island/example_trace.json \
  --output-dir .artifacts/astar-island-debug
```

### 6. Fetch post-round analysis for calibration

```bash
export AINM_ACCESS_TOKEN=...

python -m nmframemog.astar_island fetch-analysis \
  --round-id your-round-id \
  --seed-index 0 \
  --cache-dir .artifacts/astar-island
```

This stores the analysis payload under `analysis/<round-id>/seed-XX.json` so the prediction can be compared against ground truth offline.

## Artifact layout

All generated files live under the cache root, which defaults to `.artifacts/astar-island`.

- `rounds/<round-id>.json`
- `queries/<round-id>/seed-XX/*.json`
- `analysis/<round-id>/seed-XX.json`
- `predictions/<run-id>/seed-XX.json`
- `debug/<run-id>/seed-XX/`
- `runs/<run-id>/summary.json`
- `mapping/terrain_mapping.json`

Viewport screenshot artifacts use stable names that encode query order and viewport bounds, for example:

```text
query-000_step-000_x-001_y-001_w-015_h-015.svg
```

## Notes on the baseline

This is a reliability-first implementation, not a full simulator clone. The baseline uses local terrain and neighborhood heuristics to form a calibrated prior, then converts that belief state into a legal tensor via Monte Carlo aggregation. The planner prioritizes high-entropy, unresolved regions so live queries improve the tensor where the score is most sensitive.
