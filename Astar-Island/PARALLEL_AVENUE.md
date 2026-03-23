# Worktree 6 Avenue — Likelihood and Posterior Calibration

## Mission Anchor

The challenge rewards accurate full-map probability tensors under a strict 50-query budget. The docs and `AGENTS.md` make the key constraint explicit: hidden simulation parameters are shared across all 5 seeds, so better round-level parameter inference should improve every downstream prediction.

## Why This Avenue Is High-Value

The current solver leans heavily on `benchmark/src/astar_twin/solver/inference/likelihood.py` and `benchmark/src/astar_twin/solver/inference/posterior.py`:

- `compute_particle_loglik()` uses fixed `0.75 * grid + 0.25 * stats` weighting.
- Grid likelihood and settlement-stat likelihood each trigger their own inner Monte Carlo loop.
- Posterior pruning, resampling, and tempering use fixed thresholds.

If this layer is miscalibrated, the allocator and final predictor are optimizing on the wrong posterior.

## Core Hypothesis

Better hidden-parameter inference will raise benchmark mean score and reduce run-to-run variance more than most local heuristics. The most promising directions are:

1. Reusing simulated trajectories across grid and stats likelihood terms.
2. Reweighting or reformulating the grid-vs-stats likelihood split.
3. Improving posterior collapse handling with adaptive ESS / tempering / pruning rules.
4. Exploiting the fact that the same hidden parameters govern all 5 seeds in a round.

## Primary Files To Analyze

- `Astar-Island/docs/overview.md`
- `Astar-Island/docs/RULESET.md`
- `Astar-Island/AGENTS.md`
- `Astar-Island/benchmark/src/astar_twin/solver/inference/likelihood.py`
- `Astar-Island/benchmark/src/astar_twin/solver/inference/posterior.py`
- `Astar-Island/benchmark/src/astar_twin/solver/inference/particles.py`
- `Astar-Island/benchmark/src/astar_twin/solver/observe/features.py`
- `Astar-Island/benchmark/src/astar_twin/solver/pipeline.py`

## Questions You Should Answer In The Plan

1. Which hidden parameters in `docs/RULESET.md` are actually identifiable from the currently extracted observation features?
2. How much duplicated simulation work exists between `_simulate_viewport_classes()` and `_simulate_settlement_stats()`?
3. Where does posterior collapse happen in practice, and can it be detected earlier than the current fixed thresholds?
4. Can the solver update shared round-level beliefs more explicitly before spending more adaptive queries?

## Constraints

- Treat the simulator as a black box.
- Do **not** plan to modify `benchmark/src/astar_twin/engine/`, `phases/`, or `mc/` for this avenue.
- Do **not** change `SimulationParams` default values.
- Preserve determinism for identical `base_seed` inputs.

## Scope Boundary

This avenue owns **posterior / likelihood calibration**, not final tensor blending. That means:

- In scope: likelihood formulation, posterior update behavior, particle reuse, collapse handling, round-level parameter learning.
- Out of scope: hedge tuning, final tensor blending weights, output calibration layers, benchmark reporting redesign, or new benchmark registry strategies.

## Success Criteria

- A concrete plan for improving parameter inference without breaking benchmark comparability.
- A benchmark protocol covering both `run_benchmark_suite.py` and `run_multi_fixture_suite.py`.
- Clear ablations that separate: likelihood changes, posterior changes, and caching/reuse changes.

## Expected Output From The Planning Agent

Produce a plan that explains the current inference path, names the highest-leverage changes, proposes measurable experiments, and defines how to verify score, variance, and runtime impact.
