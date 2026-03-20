---
name: astar-contract-check
description: Validate one saved prediction artifact before evaluation handoff.
allowed-tools: Bash
context: fork
agent: tester
---

Run the Astar contract check for one saved prediction bundle.

## Purpose

Use this skill to confirm that a candidate prediction file is legal before any benchmark or promotion step.

## Allowed inputs

- `prediction_file`, a single saved prediction JSON file

## Exact commands

```bash
python -m nmframemog.astar_island validate-prediction <prediction-file>
PYTHONPATH=. uv run --no-project --with pytest pytest tests/astar -q
```

## Artifacts produced

- No new solver artifacts
- Validation output to stdout for the checked file

## Evidence paths

- Optional handoff note under `.sisyphus/evidence/`

## Refusal conditions

- Refuse if more than one prediction file is supplied
- Refuse if the input is not a saved JSON prediction artifact
- Refuse if the caller asks for benchmarking, promotion, dataset refresh, or debug rendering
- Refuse if the check is meant to replace the evaluation lane verdict
