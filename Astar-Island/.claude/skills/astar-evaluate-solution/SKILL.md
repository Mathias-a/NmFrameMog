---
name: astar-evaluate-solution
description: Run the promotion verdict for one frozen candidate on one frozen dataset.
allowed-tools: Bash
context: fork
agent: reviewer
---

Run the promotion decision for one frozen candidate and one frozen dataset version.

## Purpose

Use this skill only for the evaluation lane promotion step. This is the only skill that may promote a candidate.
The first command below is the only operator-facing promotion command for Astar Island.

## Allowed inputs

- `dataset_version`, one frozen dataset version under `.artifacts/astar-island/datasets/<version>/`
- `candidate`, one candidate id with frozen prediction artifacts in the dataset version
- Optional `cache_dir`, default `.artifacts/astar-island`

## Exact commands

```bash
uv run --no-project python -m nmframemog.astar_island evaluate-solution promote --cache-dir .artifacts/astar-island --dataset-version <version> --candidate <candidate-id>
PYTHONPATH=. uv run --no-project --with pytest pytest tests/astar -q
```

The promote command is the sole promotion entrypoint. The pytest command is the existing offline quality and CI gate agents should run before handoff; it stays offline and lets the evaluation surface be checked without live API access.

Wrapped module target used by the runnable wrapper in this checkout:

```bash
python -m nmframemog.astar_island evaluate-solution promote --cache-dir .artifacts/astar-island --dataset-version <version> --candidate <candidate-id>
```

## Artifacts produced

- Machine-readable promotion verdict artifacts
- Reference-comparison evidence written by the evaluation lane, without changing frozen dataset contents

## Evidence paths

- `.sisyphus/evidence/`

## Refusal conditions

- Refuse if the caller requires a workflow where benchmarking has not already been completed separately and forbids `astar-evaluate-solution promote` from rerunning the same offline suite itself
- Refuse if the candidate, references, and reports do not all point to the same frozen dataset version
- Refuse if the caller asks for live API access during promotion or replay
- Refuse if the caller asks capture, solve, or report lanes to bless or replace references
- Refuse if the caller asks for ad hoc promotion logic outside `astar-evaluate-solution promote`
- Refuse if `uv run --no-project python -m nmframemog.astar_island evaluate-solution promote` is not available in the current checkout, report the missing wrapper instead of inventing another command
