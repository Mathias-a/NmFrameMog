---
name: astar-solve-round
description: Run the solver on one frozen round detail artifact.
allowed-tools: Bash
context: fork
agent: implementer
---

Run one offline solve pass from a frozen round detail file.

## Purpose

Use this skill only for the solve lane. It runs the solver against one round detail JSON from a frozen dataset.

## Allowed inputs

- `round_detail_file`, one file under `.artifacts/astar-island/datasets/<version>/rounds/`
- Optional `cache_dir`, default `.artifacts/astar-island`

## Exact commands

```bash
python -m nmframemog.astar_island solve-round --round-detail-file .artifacts/astar-island/datasets/<version>/rounds/<round-id>.json --cache-dir .artifacts/astar-island
python -m nmframemog.astar_island validate-prediction <prediction-file>
PYTHONPATH=. uv run --no-project --with pytest pytest tests/astar -q
```

## Artifacts produced

- `predictions/<run-id>/`
- `runs/<run-id>/summary.json`
- `debug/<run-id>/seed-XX/`

## Evidence paths

- Solver run notes can be stored under `.sisyphus/evidence/`

## Refusal conditions

- Refuse if the caller wants live fetching by round id instead of a frozen round detail file
- Refuse if the round detail file is outside one frozen dataset version
- Refuse if the caller asks to benchmark, promote, or bless the candidate in this skill
- Refuse if the solve step depends on live API access during offline replay work
