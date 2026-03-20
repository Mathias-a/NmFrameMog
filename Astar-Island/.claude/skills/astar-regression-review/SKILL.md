---
name: astar-regression-review
description: Render one saved trace into a debug bundle for regression review.
allowed-tools: Bash
context: fork
agent: reviewer
---

Render one saved trace into a report bundle for regression review.

## Purpose

Use this skill only for the report lane. It packages saved traces and benchmark context into a reviewable debug bundle.

## Allowed inputs

- `trace_file`, one saved trace JSON file
- `output_dir`, one target directory under `.artifacts/astar-island/debug/<run-id>/seed-XX/`

## Exact commands

```bash
python -m nmframemog.astar_island render-debug --input <trace.json> --output-dir .artifacts/astar-island/debug/<run-id>/seed-00
PYTHONPATH=. uv run --no-project --with pytest pytest tests/astar -q
```

## Artifacts produced

- `.artifacts/astar-island/debug/<run-id>/seed-XX/`
- HTML or SVG debug bundles linked to a specific dataset version and candidate or run id
- Summary evidence under `.sisyphus/evidence/`

## Evidence paths

- `.artifacts/astar-island/debug/<run-id>/seed-XX/`
- `.sisyphus/evidence/`

## Refusal conditions

- Refuse if the input trace is not already saved to disk
- Refuse if the caller asks this skill to compute new predictions, benchmark results, or promotion verdicts
- Refuse if the caller asks for live API access during offline regression review
- Refuse if the review bundle would mix traces or reports from different frozen dataset versions
