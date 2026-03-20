---
name: astar-refresh-dataset
description: Freeze one completed round into one versioned dataset snapshot.
allowed-tools: Bash
context: fork
agent: implementer
---

Freeze one completed Astar round into a single immutable dataset version.

## Purpose

Use this skill only for the capture lane. It freezes one completed round into `.artifacts/astar-island/datasets/<version>/`.
The first command below is the only operator-facing dataset refresh command for Astar Island.

## Allowed inputs

- `round_id`, one completed round id
- `dataset_version`, one new dataset version string
- Optional `cache_dir`, default `.artifacts/astar-island`

## Exact commands

```bash
uv run --no-project python -m nmframemog.astar_island refresh-dataset --round-id <completed-round-id> --cache-dir .artifacts/astar-island --dataset-version <version>
PYTHONPATH=. uv run --no-project --with pytest pytest tests/astar -q
```

The refresh command is the sole capture entrypoint. The pytest command is the existing offline quality and CI gate for the frozen Astar lane surface and does not require live replay, benchmarking, or promotion access.

Wrapped module target used by the runnable wrapper in this checkout:

```bash
python -m nmframemog.astar_island refresh-dataset --round-id <completed-round-id> --cache-dir .artifacts/astar-island --dataset-version <version>
```

## Artifacts produced

- `.artifacts/astar-island/datasets/<version>/manifest.json`
- `.artifacts/astar-island/datasets/<version>/hashes.json`
- `.artifacts/astar-island/datasets/<version>/query-trace.json`
- `.artifacts/astar-island/datasets/<version>/rounds/<round-id>.json`
- `.artifacts/astar-island/datasets/<version>/queries/`
- `.artifacts/astar-island/datasets/<version>/analysis/<round-id>/seed-XX.json`
- `.artifacts/astar-island/datasets/<version>/predictions/`
- `.artifacts/astar-island/datasets/<version>/mapping/`
- `.artifacts/astar-island/datasets/<version>/seed-metadata/`

## Evidence paths

- Capture evidence can be summarized under `.sisyphus/evidence/`

## Refusal conditions

- Refuse if the round is still live or incomplete
- Refuse if the dataset version already exists or would rewrite frozen artifacts
- Refuse if any required seed analysis, hash, or query trace artifact is missing
- Refuse if the caller asks to mix artifacts from multiple dataset versions
- Refuse if the caller asks for offline replay, benchmarking, or promotion in the same step
- Refuse if `uv run --no-project python -m nmframemog.astar_island refresh-dataset` is not available in the current checkout, report the missing wrapper instead of inventing another command
