# ng-submission

Use this skill when preparing NorgesGruppen submission packaging.

## Focus

- preserve the `run.py` at zip-root contract from `docs/submission.md`
- keep submission contents sandbox-safe and offline
- minimize files, dependencies, and moving parts

## Guardrails

- no runtime installs
- avoid blocked imports like `os`, `subprocess`, and `yaml`
- prefer `pathlib` and `json`

## Done when

- the package structure clearly maps to a root-level `run.py`
- submission helpers do not assume network access
- the detector-first path remains the default shipped path
