2026-03-20 — Task 8
- `AGENTS.md` defines `python -m nmframemog.astar_island refresh-dataset` and `python -m nmframemog.astar_island evaluate-solution ...` as the lane-facing wrappers, but `idk_2/astar_island_dr_plan_1/cli.py` in this checkout currently exposes only `solve-round`, `validate-prediction`, `fetch-analysis`, and `render-debug`.
- Minimal interpretation used for this task: document the wrapper interface exactly as the lane contract requires, add refusal text that reports a missing wrapper when the current checkout does not expose it yet, and avoid changing CLI code because that belongs to a later task.

2026-03-20 — Task 10
- Temporary bridge for Task 10: `evaluate-solution` reads optional comparison references from `.artifacts/astar-island/evaluation/<dataset-version>/references.json` with `candidate_id`, `solver_id`, and `prediction_run_id`, but long-term blessed-reference storage policy and mutation rules still belong to Task 11.

2026-03-20 — Task 12
- `idk_2/astar_island_dr_plan_1/cli.py` still visibly exposes `evaluate-solution benchmark|promote` but not the documented `refresh-dataset` parser in this subtree, so Task 12 kept the command-surface lock at the operator-doc and offline-test level within the allowed file set instead of widening scope into CLI implementation changes.
- Follow-up resolved in the same task thread: adding the missing `refresh-dataset` parser/handler was enough to make `uv run --no-project python -m nmframemog.astar_island refresh-dataset --help` succeed without touching dataset refresh business logic.
