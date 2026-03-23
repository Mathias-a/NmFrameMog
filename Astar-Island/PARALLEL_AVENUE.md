# Worktree 10 Avenue — New Strategy Challenger and Ensemble Benchmarking

## Mission Anchor

The local digital twin and harness are meant to compare strategies before spending real API budget. The current registry is still small, which makes this a good place to explore an intentionally different challenger instead of only refining the main pipeline.

## Why This Avenue Is High-Value

The current benchmark registry is in `benchmark/src/astar_twin/strategies/__init__.py` and currently exposes:

- `uniform`
- `initial_prior`
- `filter_baseline`
- `mc_oracle`

That leaves room for a serious new strategy track: an ensemble, a solver-wrapped benchmark strategy, or a deliberately simpler but more robust challenger. This avenue creates an independent way to test whether the current end-to-end solver architecture is actually the best use of the twin.

## Core Hypothesis

An alternate strategy packaged through the official benchmark registry may outperform the current baseline set or reveal a simpler path to competitive scores. The plan should explore one **concrete challenger-first** direction:

1. A new benchmark strategy that wraps selected pieces of the current solver, or a deliberately simpler robust heuristic challenger.
2. A challenger focused on robustness and variance reduction rather than only peak mean score.
3. Strategy-level testing and comparison through the official harness.
4. Only consider ensemble behavior if it is clearly distinct from the hedge/calibration work owned by worktree 8.

## Primary Files To Analyze

- `Astar-Island/docs/overview.md`
- `Astar-Island/docs/scoring.md`
- `Astar-Island/AGENTS.md`
- `Astar-Island/benchmark/src/astar_twin/harness/protocol.py`
- `Astar-Island/benchmark/src/astar_twin/harness/runner.py`
- `Astar-Island/benchmark/src/astar_twin/strategies/__init__.py`
- `Astar-Island/benchmark/src/astar_twin/strategies/uniform/strategy.py`
- `Astar-Island/benchmark/src/astar_twin/strategies/initial_prior/strategy.py`
- `Astar-Island/benchmark/src/astar_twin/strategies/filter_baseline/strategy.py`
- `Astar-Island/benchmark/src/astar_twin/strategies/mc_oracle/strategy.py`
- `Astar-Island/benchmark/tests/strategies/`

## Questions You Should Answer In The Plan

1. What gap exists between the current registry and the full solver pipeline?
2. Which alternate strategy concept is most worth testing first: wrapped solver or robust heuristic challenger?
3. How should the new strategy be evaluated against both mean score and variance?
4. Which minimal tests are needed so a new strategy can be trusted in the benchmark harness?

## Constraints

- Respect the strategy-author rules in `AGENTS.md`.
- Do not modify engine, phase, or Monte Carlo internals from this avenue.
- Keep any new strategy deterministic for a given `base_seed`.
- Preserve benchmark comparability with the existing registry.

## Scope Boundary

This avenue owns **new benchmarkable challenger strategies**, not output calibration. That means:

- In scope: one or more new `strategies/*` implementations, registry integration, harness-level comparisons, strategy tests.
- Out of scope: hedge-weight tuning, final tensor calibration experiments, likelihood redesign, or benchmark diagnostics redesign unless strictly needed to compare the challenger.

## Success Criteria

- A plan for one or more new benchmarkable strategies with clear evaluation rules.
- A comparison method against the current registry and the main solver pipeline.
- A packaging and test strategy that keeps this avenue mergeable if it wins.

## Expected Output From The Planning Agent

Produce a plan that defines the best challenger strategy to build first, explains why it is distinct from the existing registry, and lays out how to benchmark it rigorously without drifting into simulator changes or into worktree 8's calibration scope.
