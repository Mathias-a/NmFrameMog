# Decisions — astar-solver-fix

## Metis-validated decisions
- safe_prediction tolerance: >= 0.01 - 1e-9 (practical floating-point tolerance)
- Reserve batching: 5 + remainder (degrade gracefully)
- current_prediction scope: per-seed dict[int, NDArray]
- Empty-particle fallback: return uniform tensor via uniform_baseline
- dump_prediction_stats: CLI + library function, fixture-derived dimensions
- Disagreement: true cellwise top-2 argmax using lightweight inner MC (2 runs per particle)

