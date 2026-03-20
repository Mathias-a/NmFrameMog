# Astar Island

Implementation-oriented notes for this repository's Astar Island package.

Reference competition docs: [docs/astar-island/](./astar-island/README.md)

This package provides:

- an authenticated async client for `/budget`, `/rounds`, `/simulate`, and `/submit`
- a CLI for fetching data, building predictions, and submitting them
- a valid baseline probability-grid generator

## CLI examples

Fetch budget:

```bash
PYTHONPATH=src uv run python -m task_astar_island.cli budget --token "$ASTAR_TOKEN"
```

Fetch rounds:

```bash
PYTHONPATH=src uv run python -m task_astar_island.cli rounds --token "$ASTAR_TOKEN"
```

Build a local prediction grid:

```bash
PYTHONPATH=src uv run python -m task_astar_island.cli predict --width 8 --height 8 --output prediction.json
```

Fetch budget and rounds, build a prediction, and submit it:

```bash
PYTHONPATH=src uv run python -m task_astar_island.cli solve --token "$ASTAR_TOKEN" --round-id "$ROUND_ID" --seed-index 0 --submission-output submission.json
```

Submit an existing prediction body:

```bash
PYTHONPATH=src uv run python -m task_astar_island.cli submit --token "$ASTAR_TOKEN" --round-id "$ROUND_ID" --seed-index 0 --body-file submission.json
```

## Prediction guarantees

- tensor shape is `prediction[y][x][class]`
- exactly 6 classes per cell
- every probability is strictly greater than zero
- every cell sums to 1.0

## Notes

The upstream simulation and scoring rules are not fully documented here, so the prediction builder is intentionally a deterministic baseline rather than a fabricated game-specific solver. The client and CLI focus on reliable transport, validation, and valid submission structure.
