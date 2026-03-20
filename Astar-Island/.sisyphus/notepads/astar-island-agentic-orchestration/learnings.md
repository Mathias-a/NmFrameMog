2026-03-20 — Task 1
- The solver already defines canonical tensor legality in `idk_2/astar_island_dr_plan_1/solver/validator.py`; evaluation artifacts should reuse `validate_prediction_tensor()` instead of duplicating floor or normalization rules.
- The challenge query budget is round-shared, so the offline contract should model a single `RoundQueryTrace` with global query indices rather than unrelated per-seed traces.
- Organizer analysis ground truth can legitimately contain zero probabilities even though candidate predictions must not, so evaluation contracts need separate validation paths for candidate tensors vs. reference tensors.
2026-03-20 — Task 2
- Frozen benchmark datasets can reuse the live cache layout (`rounds/`, `queries/`, `analysis/`, `predictions/`, `mapping/`) inside `datasets/<version>/`, which keeps replay/loading logic aligned with the existing solver lane.
- Reconstructing the shared round query trace from cached simulate payloads depends on `queries_used`; if cached payloads omit `queries_used` or `queries_max`, refresh must fail closed because global trace identity cannot be recovered safely.
2026-03-20 — Task 3
- Offline replay must validate `query-trace.json` using the same logical hash payload that Task 2 used during freezing, not the raw serialized file body, because the stored trace identity intentionally excludes `trace_artifact_hash`.
- Proving replay is frozen-only is easiest by deleting the mutable cache trees outside `datasets/<version>/` after snapshot creation and confirming replay still rebuilds the benchmark input from the frozen dataset alone.
2026-03-20 — Task 5
- The benchmark metric layer can reuse the existing `BenchmarkReport` for canonical per-seed score accounting, then wrap it with richer summaries for calibration, stability, fallback detection, and reference-delta structure without changing the Task 1 contract.
- Calibration should be computed over the full class distribution on dynamic cells only; averaging cross-entropy, Brier score, and total variation catches overconfident shape errors that top-class accuracy would miss.

2026-03-20 — Task 4
- Keeping `AGENTS.md` at 70 lines or fewer is enough to force the file to stay operational, while tests still cover required lane labels, commands, artifact paths, and stop conditions.
- The lane doc should lock promotion to `astar-evaluate-solution promote` and leave capture, solve, and report as artifact producers only, so later skill docs inherit one clear authority boundary.

2026-03-20 — Task 6
- Historical backfill works best as an append-only raw capture tree under `history/raw/<snapshot-version>/`, because it can accumulate completed rounds without rewriting the curated benchmark datasets under `datasets/<version>/`.
- Reusing per-round `FrozenDatasetManifest` and `RoundQueryTrace` payloads inside raw history snapshots preserves Task 2 hash coverage and query identity invariants while still allowing curated multi-round corpora to restamp `dataset_version` later.

2026-03-20 — Task 7
- The benchmark suite can preserve canonical `BenchmarkReport` scoring while layering robustness checks as separate labeled results; hard-gate integrity failures still raise immediately, while adversarial signals like edge-clamped regressions and overconfidence fit best as explicit report-only checks in the suite payload.
- For immutable evaluation-contract fixtures, tests should rebuild `BenchmarkInput` and `RoundQueryTrace` objects instead of mutating frozen dataclass fields directly; using `object.__setattr__` is only useful when intentionally simulating corrupted in-memory artifacts after construction.
- Mypy will treat helper predicates as returning `Any` if a structured argument like `viewport` is left untyped; importing and annotating the concrete `Viewport` model is enough to satisfy strict return checking without changing the robustness logic.

2026-03-20 — Task 8
- The repo skill style is easiest to keep consistent by using short YAML frontmatter plus fixed sections for purpose, allowed inputs, exact commands, artifacts, evidence paths, and refusal conditions.
- For Astar lane docs, the cleanest split is one skill per lane action, with `astar-evaluate-solution` holding the only promotion authority and the benchmark skills explicitly tied to one frozen dataset version.

2026-03-20 — Task 9
- AGENTS drift tests stay simpler and harder to game when they derive skill names and allowed `python -m nmframemog.astar_island ...` commands directly from `.claude/skills/astar-*/SKILL.md`, then assert the lane doc mentions each one without documenting any extra workflow.

2026-03-20 — Task 10
- `evaluate-solution` stays narrowest and safest when it reuses `load_offline_replay()` for frozen artifact validation and `evaluate_benchmark_suite()` for all scoring, delta, and hard-gate logic, then only adds mode-specific report/verdict framing on top.
- Reusing replay for blessed baseline and last blessed candidate comparisons means reference bundles must also be present in the frozen dataset hash registry; that keeps offline evaluation honest and avoids a second unverified reference-loading path.
- For offline evaluation, candidate selection cannot be inferred from `round_id` alone; a frozen candidate-to-`prediction_run_id` registry is the smallest reliable way to ensure the requested candidate actually controls which prediction bundle gets benchmarked.

2026-03-20 — Task 11
- Blessed evaluation references are safer as separate role-specific records under `evaluation/<dataset-version>/` with explicit `reference_key` and `dataset_version_compatibility` fields; that lets promote-mode fail closed on missing or incompatible evidence without changing the frozen replay path.

2026-03-20 — Task 12
- When `AGENTS.md` is intentionally capped at 70 lines by tests, the safest place to document final operator examples is the lane skill docs; that keeps the agent lane contract minimal while still giving operators one exact refresh, benchmark, and promote command.
- The isolated command `PYTHONPATH=. uv run --no-project --with pytest pytest tests/astar -q` already functions as the offline Astar quality and CI gate in this checkout, so the minimal wiring is to make that gate explicit and test-locked in the skill docs instead of adding broader workspace automation that would hit the known broken uv workspace member.
- In this checkout, the operator-runnable surface is `uv run --no-project python -m nmframemog.astar_island ...`, while AGENTS drift checks still benefit from preserving the underlying `python -m nmframemog.astar_island ...` module target text separately for lane-command alignment.
- For `evaluate-solution promote`, the cleanest command-surface behavior is to keep printing the machine-readable JSON payload on both pass and fail, then switch only the process exit code to non-zero when `promotion_verdict` is `fail`; that preserves automation-readable output without widening evaluation logic.
