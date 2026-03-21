# ng-data-eval

Use this skill when working on NorgesGruppen dataset layout, data checks, or evaluation helpers.

## Focus

- keep code inside `src/ng_data/data/` and `src/ng_data/eval/`
- prefer small typed utilities and deterministic local checks
- treat missing real dataset archives as a normal local-development condition

## Guardrails

- do not add training infrastructure here
- keep configs and fixtures sandbox-safe
- match the repo's small `argparse` + `Path` style for any CLI surface

## Done when

- required directories/files exist
- local checks are testable without challenge data downloads
- evaluation helpers stay detector-first and narrow in scope
