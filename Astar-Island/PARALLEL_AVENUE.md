# Worktree 9 Avenue — Benchmark Diagnostics, Replay Validation, and Comparability

## Mission Anchor

The benchmark is the control room for this whole challenge: if it is noisy, incomplete, or poorly calibrated, every solver change becomes harder to trust. The docs and current branch already point at `prior_spread`, replay validation, and multi-fixture evaluation as important comparability tools.

## Why This Avenue Is High-Value

The current evaluation surface is spread across:

- `benchmark/src/astar_twin/solver/eval/run_benchmark_suite.py`
- `benchmark/src/astar_twin/solver/eval/run_replay_validation.py`
- `benchmark/src/astar_twin/solver/eval/run_multi_fixture_suite.py`
- `benchmark/src/astar_twin/solver/eval/dump_prediction_stats.py`
- `benchmark/src/astar_twin/params/prior_sampling.py`
- `benchmark/src/astar_twin/fixture_prep/ground_truth.py`
- `benchmark/src/astar_twin/data/models.py`

Right now the suite reports useful high-level stats, but it can still be hard to tell whether a change improved parameter inference, query placement, calibration, or merely one easy round. Stronger diagnostics and comparability discipline will help every other avenue. `prior_spread` is relevant here, but as one subtopic inside fair evaluation rather than the main theme.

## Core Hypothesis

Better benchmark visibility can unlock faster solver improvements. The plan should explore:

1. Per-round and per-seed failure decomposition.
2. Replay-driven diagnostics that explain where predictions diverge from ground truth.
3. Prior-spread and parameter-sampling experiments that keep comparisons fair when uncached `DEFAULT_PRIOR` ground truths are involved.
4. A tighter offline evaluation protocol that resists overfitting to a single fixture.

## Primary Files To Analyze

- `Astar-Island/docs/overview.md`
- `Astar-Island/docs/scoring.md`
- `Astar-Island/docs/RULESET.md`
- `Astar-Island/AGENTS.md`
- `Astar-Island/benchmark/src/astar_twin/solver/eval/run_benchmark_suite.py`
- `Astar-Island/benchmark/src/astar_twin/solver/eval/run_replay_validation.py`
- `Astar-Island/benchmark/src/astar_twin/solver/eval/run_multi_fixture_suite.py`
- `Astar-Island/benchmark/src/astar_twin/solver/eval/dump_prediction_stats.py`
- `Astar-Island/benchmark/src/astar_twin/params/prior_sampling.py`
- `Astar-Island/benchmark/src/astar_twin/fixture_prep/ground_truth.py`
- `Astar-Island/benchmark/src/astar_twin/data/models.py`

## Questions You Should Answer In The Plan

1. Which rounds or seeds dominate regressions, and are they tied to specific mechanics?
2. How should replay validation be used to separate model error from benchmark noise?
3. Are current prior-spread assumptions too broad or too narrow for fair offline comparison?
4. What extra metrics would make future solver work materially faster and safer?

## Constraints

- Preserve benchmark comparability across runs.
- Do not corrupt fixture provenance or cached ground-truth semantics.
- Keep the simulator itself stable.
- Any prior-shaping recommendation must be benchmarked transparently and documented.

## Scope Boundary

This avenue owns **measurement, replay, and fairness of comparison**, not solver logic. That means:

- In scope: benchmark outputs, replay validation, error decomposition, run-to-run comparability, per-round diagnostics, fair use of prior-spread.
- Out of scope: new query policies, likelihood math changes, hedge tuning, or new strategy behavior except where needed for evaluation support.

## Success Criteria

- A plan for turning the benchmark into a stronger decision tool.
- Explicit diagnostic outputs that help the other four worktrees compare ideas fairly.
- A reproducible evaluation protocol across single-fixture and multi-fixture runs.

## Expected Output From The Planning Agent

Produce a plan that prioritizes measurement, replay, and comparability discipline. The end result should make it easier to tell which solver changes are real wins and which are just noise or overfit.
