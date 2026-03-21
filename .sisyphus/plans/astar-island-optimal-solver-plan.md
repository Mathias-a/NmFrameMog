# Astar Island Optimal-Performance Solver Plan

## Objective

Build a new Astar Island solver that optimizes the real score function: entropy-weighted KL divergence on a `H×W×6` probability tensor with a strict `0.01` floor. The solver should use the rules that are genuinely hard constraints, infer round-level hidden parameters from limited viewport evidence, and spend the 50-query budget on viewports that maximize expected full-map score improvement rather than local curiosity.

## Verified rule map

The table below is grounded in `docs/*.md` and the local proxy simulator in `round_8_implementation/solver/proxy_simulator.py`. The proxy is not guaranteed to match the live game exactly, but it is the strongest executable source in this repo and should be the benchmark-driving approximation until live analysis data contradicts it.

### Terrain and class mapping

| Internal code | Terrain | Prediction class | Hard constraint | Direct transitions in proxy | Reachable final classes after 50 years | Planning implication |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | Ocean | 0 empty | Static. Ocean stays ocean. | none | empty only | Never waste probability mass on non-empty classes beyond the floor. Ocean also fixes coastline forever. |
| 11 | Plains | 0 empty | Not static. | can stay plains, can become settlement or port via expansion | empty, settlement, port, ruin, forest | Plains near active settlements are dynamic and score-relevant. |
| 0 | Empty | 0 empty | Not static. | can stay empty, can become settlement or port via expansion | empty, settlement, port, ruin, forest | Treat generic empty like buildable land, not as a safe static class. |
| 1 | Settlement | 1 settlement | Dynamic. | can stay settlement, upgrade to port, collapse to ruin, change owner | settlement, port, ruin | Main driver of expansion, conflict, and winter collapse. |
| 2 | Port | 2 port | Dynamic but coast-locked. | can stay port, collapse to ruin | port, ruin, settlement only in malformed non-coastal cases | Coastal trade and longship access make ports high-value observation targets. |
| 3 | Ruin | 3 ruin | Dynamic transitional state. | can stay ruin, rebuild to settlement, rebuild to port, become forest, fade to plains | ruin, settlement, port, forest, empty | Ruins are among the highest-entropy cells and should be modeled explicitly. |
| 4 | Forest | 4 forest | Mostly stable, not static. | can stay forest, can be colonized by expansion | forest, settlement, port, ruin, empty | Forest adjacency strongly affects food, ruin reclamation, and expansion value. |
| 5 | Mountain | 5 mountain | Static. Mountain stays mountain; nothing else becomes mountain. | none | mountain only | Mountains define impassable barriers and must be treated as certainty. |

### Additional hard constraints

- Ocean borders the map and remains ocean for the whole simulation.
- Coastline is static because ocean is static. A cell is coastal iff it starts adjacent to ocean.
- Ports are only valid on coastal settlements. The proxy enforces this with `_normalize_ports()`.
- No rule creates mountains or ocean. No rule converts mountains or ocean into anything else.
- Prediction class 0 merges ocean, plains, and empty. Only ocean is truly static inside that class.
- Static cells contribute little or nothing to score because entropy weighting suppresses near-zero-entropy cells.

### Phase-level mechanics that matter for modeling

1. **Growth**
   - Food rises with adjacent forest and, slightly, adjacent plains.
   - Wealth rises with nearby active settlements, tech, and ports.
   - Defense rises with nearby mountains and wealth.
   - Coastal settlements can upgrade to ports.
   - Rich, advanced ports can gain longships.
   - Strong settlements can expand within Chebyshev radius 2, preferring ruins first, then coastal/buildable land.

2. **Conflict**
   - Raid range depends on base range plus port and longship bonuses.
   - Hungry settlements raid more aggressively.
   - Combat outcomes depend on population, wealth, defense, tech, and randomness.
   - Defeated settlements either flip owner or collapse to ruin.

3. **Trade**
   - Ports trade within Manhattan distance 7.
   - Trade boosts food, wealth, and tech for both ports.

4. **Winter**
   - Every year draws a winter severity.
   - Collapse happens when population, food, or wealth cross failure thresholds.

5. **Environment**
   - Ruins age over time.
   - Nearby strong settlements may reclaim ruins as settlement/port.
   - Old ruins with forest support can become forest.
   - Older ruins can fade to plains.

## What this implies for an optimal solver

The benchmark and scoring rules push the solver toward four priorities:

1. **Exploit exact feasibility constraints first.** Mountains and ocean should be near-deterministic. Ports should only appear on coastal cells. This removes avoidable KL loss.
2. **Focus on dynamic neighborhoods, not individual cells.** Expansion, conflict, trade, and ruin reclamation all depend on local motifs, not isolated tiles.
3. **Infer shared hidden parameters at the round level.** The docs state hidden parameters are shared across all 5 seeds. A query on one seed can improve predictions on all seeds if it reveals expansion, raid, trade, or winter intensity.
4. **Allocate queries by expected score gain, not raw entropy alone.** The current baseline planner is entropy-driven. That is better than random, but it ignores whether a viewport is informative about global hidden parameters or only about one local patch.

## Query strategy: why “all 50 shots at one place” is not optimal

Repeating a viewport is useful because each query samples a fresh stochastic run under the same hidden parameters. That helps estimate the local outcome distribution at that location. The problem is coverage: the score averages over the full map’s dynamic cells, and one location rarely identifies all of expansion rate, raid aggressiveness, trade richness, winter harshness, and ruin reclamation behavior.

The optimal strategy is therefore **controlled replication, not full concentration**:

- Use some repeated queries on high-sensitivity motif windows to estimate local stochastic outcome distributions.
- Spread those repeats across **different structural motifs** so the solver can infer round-level parameters that generalize to the whole map.
- Adapt later queries based on the posterior: once one mechanism is pinned down, shift the budget to the mechanisms still driving uncertainty.

## Proposed solver architecture

### 1. Exact rules layer

Create a deterministic rules module that emits, for every cell:

- allowed final classes
- port-feasibility bit
- static certainty mask
- motif tags such as `coastal`, `mountain_sheltered`, `forest_supported`, `ruin_frontier`, `settlement_cluster`, `conflict_frontier`

This layer should produce a strong feasibility prior before any stochastic modeling.

### 2. Round-level latent parameter posterior

Represent the hidden round settings with a compact parameter vector, for example:

- expansion aggressiveness
- port upgrade propensity
- longship propensity
- conflict aggressiveness
- conquest-vs-collapse bias
- trade richness
- winter harshness
- ruin reclamation propensity
- ruin-to-forest propensity
- ruin-to-plains propensity

Use a particle posterior over these parameters. Each particle defines one simulator variant or surrogate parameterization.

### 3. Seed-specific latent state inference

Per seed, track the current posterior over:

- settlement survival / collapse risk
- port persistence / emergence risk
- likely expansion fronts
- likely conflict pairs and raid corridors
- ruin age / reclamation pressure

The posterior state should combine the exact rules layer, initial map geometry, and observed viewport outcomes.

### 4. Two-sided Monte Carlo viewport valuation

For each candidate viewport, estimate value with a nested Monte Carlo procedure:

1. Sample particles from the current round-level posterior.
2. For each particle, simulate possible final worlds for the candidate seed.
3. Extract the viewport observation distribution that the query would reveal.
4. Update the posterior as if that observation were received.
5. Recompute the predicted full-map tensor under the updated posterior.
6. Score the improvement with a proxy objective aligned to entropy-weighted KL, for example:
   - expected reduction in weighted map entropy
   - expected reduction in cross-particle disagreement
   - expected reduction in surrogate weighted KL against held-out particle rollouts

Choose the viewport with the best **expected full-map gain**, not merely the highest current local entropy.

### 5. Query candidate generation

Do not evaluate all possible windows equally. Generate a compact candidate set from motif-rich anchors:

- coastal settlement clusters for port/trade/longship inference
- rival settlement corridors within plausible raid range
- expansion frontiers: settlement next to empty/plains/forest/ruin
- ruin neighborhoods with and without nearby forests
- mountain-choke conflict fronts
- sparse-food inland settlements for winter-collapse inference

Candidate windows should be deduplicated by motif overlap so compute is spent on structurally different questions.

### 6. Budget policy

Recommended default budget split for 50 total queries:

- **Phase A: 15 queries for parameter discovery**
  - 3 coastal trade windows
  - 3 conflict-front windows
  - 3 expansion-front windows
  - 3 ruin/forest reclamation windows
  - 3 high-risk winter-starvation windows
- **Phase B: 25 adaptive queries**
  - repeatedly choose the highest expected full-map gain viewport under the updated posterior
- **Phase C: 10 reserve queries**
  - use for targeted replication where the posterior still has bimodal behavior

This is a starting policy, not a fixed rule. The actual solver should learn when the first phase can stop early because the posterior has already concentrated.

### 7. Final tensor generation

After the query budget is exhausted:

1. Resample or reweight the posterior particles.
2. Run many full-map rollouts per seed under the retained particles.
3. Aggregate class frequencies into the `H×W×6` tensor.
4. Apply feasibility masks before flooring.
5. Apply the `0.01` floor and renormalize.
6. Run strict validation.

## Concrete implementation plan

### Phase 0 — benchmark and rule harness

- Freeze the current baseline benchmark report as the promotion target.
- Add a rule-audit script that prints allowed class sets and motif tags for each seed.
- Add tests for all hard invariants: ocean static, mountain static, coastal-port only, no mountain creation, no ocean creation.

### Phase 1 — feasibility prior

- Replace the current heuristic prior with a rule-aware prior that uses exact feasibility masks and neighborhood motifs.
- Keep this step cheap and deterministic so it can be used in the query loop.
- Promotion criterion: beats current baseline on benchmark with no live queries.

### Phase 2 — surrogate simulator with explicit parameters

- Refactor the proxy simulator mechanics into a parameterized surrogate.
- Expose the latent parameters as a vector that can be sampled and updated.
- Fit broad priors from the existing proxy and later recalibrate using post-round analysis artifacts.

### Phase 3 — particle posterior and observation likelihood

- Maintain particle weights based on how well each particle explains observed viewport terrain and settlement stats.
- Use settlement stats from the API response (`population`, `food`, `wealth`, `defense`, `owner_id`) because they reveal more than the final grid alone.
- Promotion criterion: posterior predictive log-likelihood improves on cached/emulated observation traces.

### Phase 4 — expected-value query planner

- Replace the current entropy planner with the two-sided Monte Carlo planner.
- Start with a small candidate set and shallow nested rollouts so planning is fast enough for a live round.
- Add a fallback heuristic planner if compute budget becomes tight.

### Phase 5 — calibrated rollout engine

- Use the posterior particles to generate final seed tensors.
- Calibrate confidence with post-round analysis: broaden distributions in motifs with systematic misspecification.
- Promotion criterion: improved benchmark score plus better calibration on high-entropy cells.

### Phase 6 — live-round policy

- Integrate online caching and replay.
- Save every query, posterior snapshot, and chosen viewport score decomposition.
- After each completed round, compare query decisions with analysis-derived regret and update the candidate generator.

## Suggested module layout

- `round_8_implementation/solver/rules.py` — exact terrain feasibility, coastal logic, motif extraction
- `round_8_implementation/solver/latent_params.py` — latent parameter dataclass and priors
- `round_8_implementation/solver/surrogate.py` — parameterized simulator/surrogate dynamics
- `round_8_implementation/solver/posterior.py` — particle weights, resampling, observation likelihood
- `round_8_implementation/solver/query_value.py` — two-sided Monte Carlo viewport valuation
- `round_8_implementation/solver/candidates.py` — motif-aware viewport candidate generation
- `round_8_implementation/solver/final_predictor.py` — benchmarkable `predict(grid)` entrypoint
- `round_8_implementation/solver/calibration.py` — post-round calibration and confidence shaping

## Benchmark baseline comparison

Benchmark command used:

```bash
uv run python -m round_8_implementation benchmark --preset full \
  --model uniform=benchmark_models:uniform_floor_model \
  --model initial=benchmark_models:initial_state_projection_model \
  --model idk1=benchmark_models:idk1_wrapped_model \
  --model masked=benchmark_models:masked_baseline_model \
  --model phase1=benchmark_models:phase1_rules_model \
  --output benchmark/current-model-comparison-full.json
```

Results from `benchmark/current-model-comparison-full.json`:

| Model | Mean | Median | StDev | Min | Max | Time(ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| phase1 | 21.29 | 26.54 | 8.96 | 9.23 | 29.58 | 189 |
| masked | 21.07 | 25.74 | 9.46 | 7.87 | 30.35 | 65 |
| baseline | 17.78 | 21.49 | 8.38 | 6.17 | 26.06 | 33 |
| initial | 6.56 | 6.30 | 4.98 | 0.38 | 14.16 | 11 |
| idk1 | 2.64 | 2.74 | 0.63 | 1.37 | 3.46 | 25 |
| uniform | 2.52 | 2.62 | 0.72 | 1.16 | 3.53 | 1 |

Interpretation:

- Exact feasibility masking alone (`masked`) beats the current baseline by roughly **+3.29 mean score**.
- A lightweight rules-and-motif prior (`phase1`) reaches **21.29 mean**, beating baseline by roughly **+3.51**.
- This is strong evidence that the next big gain should come from better rules, posterior inference, and query valuation rather than simply increasing rollout count.

Current promotion target:

- Champion baseline for future phases: `phase1_rules_model` at mean **21.29** on the full preset.

## Phase-by-phase QA scenarios

Every phase below must ship with executable verification. A phase is not complete until its QA commands pass.

### Phase 0 — benchmark and rule harness QA

```bash
uv run pytest tests/test_proxy_simulator.py tests/test_local_emulator.py
uv run python -m round_8_implementation benchmark --preset full --model phase1=benchmark_models:phase1_rules_model --output benchmark/current-model-comparison-full.json
```

Expected result:

- invariant tests pass
- benchmark JSON is regenerated successfully
- `benchmark/current-model-comparison-full.json` contains `phase1` and a mean score at or above the current frozen champion unless intentionally testing a challenger

### Phase 1 — feasibility prior QA

```bash
uv run python -m round_8_implementation benchmark --preset full --model masked=benchmark_models:masked_baseline_model --model phase1=benchmark_models:phase1_rules_model
```

Expected result:

- no tensor validation failures
- `masked` beats `baseline`
- `phase1` matches or beats `masked`

### Phase 2 — surrogate simulator with explicit parameters QA

```bash
uv run pytest tests/test_proxy_simulator.py
uv run python -m round_8_implementation benchmark --preset quick --model surrogate=round_8_implementation.solver.final_predictor:predict
```

Expected result:

- surrogate tests preserve hard invariants
- benchmark runs successfully with the parameterized predictor
- no regression below the frozen phase-1 champion without an explicit reason documented in the run artifact

### Phase 3 — particle posterior and observation likelihood QA

```bash
uv run pytest tests/test_local_emulator.py
uv run python -m round_8_implementation serve-emulator --port 8010
```

Then replay cached queries or scripted emulator queries against the posterior updater.

Expected result:

- posterior updates complete without shape/type errors
- likelihood weights shift in response to observed terrain and settlement stats
- posterior predictive log-likelihood on replay traces improves relative to the pre-posterior surrogate

### Phase 4 — expected-value query planner QA

```bash
uv run python -m round_8_implementation solve-round --round-detail-file .artifacts/astar-island-live/rounds/c5cdf100-a876-4fb7-b5d8-757162c97989.json --viewport-width 15 --viewport-height 15 --planned-queries-per-seed 10 --rollouts 64
```

Expected result:

- query planning completes without exhausting compute budgets
- debug artifacts show structurally diverse motif windows rather than repeated arbitrary placements
- planner score decomposition for each chosen viewport is saved in artifacts and can be inspected offline

### Phase 5 — calibrated rollout engine QA

```bash
uv run python -m round_8_implementation benchmark --preset full --model champion=round_8_implementation.solver.final_predictor:predict
```

Expected result:

- full benchmark passes
- champion mean score beats the frozen `phase1` champion score of **21.29**
- no floor violations and no collapse in the worst-seed minimum score without a compensating mean gain that is explicitly accepted

### Phase 6 — live-round policy QA

```bash
uv run pytest tests/test_local_emulator.py tests/test_local_scoring.py
uv run ruff check benchmark_models.py round_8_implementation
uv run mypy
```

Expected result:

- offline/live-like integration path passes
- query caching, prediction generation, and scoring artifacts are reproducible
- lint and type checks pass for the shipped implementation

## Decision summary

The best next solver is not “more rollouts on the current entropy planner.” It is a **rules-constrained, posterior-driven solver** that uses repeated viewports sparingly and only where they change the full-map posterior.

Oracle review updated the immediate priority order:

1. freeze `phase1_rules_model` as the current champion
2. build the parameterized surrogate
3. prove posterior updates improve replay/emulator predictive quality
4. only then invest in two-sided Monte Carlo query valuation

So the core strategic upgrade remains posterior-driven inference, but the **immediate next implementation move** is Phase 2/3, not Phase 4. Two-sided Monte Carlo query selection becomes the right next step only after the posterior is strong enough that different candidate viewports produce meaningfully different full-map outcomes.
