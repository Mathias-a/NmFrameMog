# Astar Island — Simulation Ruleset

> Purpose: Precise rule specification for building a digital twin simulator.
> Confidence levels: [CONFIRMED] = stated in docs, [INFERRED] = logical from docs, [UNKNOWN] = needs reverse-engineering.

---

## 1. World Grid

- **Dimensions**: W x H (default 40x40) [CONFIRMED]
- **Cell types**: 8 internal codes mapping to 6 prediction classes [CONFIRMED]

| Code | Terrain | Class | Static? |
|------|---------|-------|---------|
| 10 | Ocean | 0 (Empty) | Yes — never changes |
| 11 | Plains | 0 (Empty) | Mostly — can become Settlement/Port |
| 0 | Empty | 0 (Empty) | Mostly — can become Settlement/Port |
| 1 | Settlement | 1 | No — dynamic |
| 2 | Port | 2 | No — dynamic (settlement + coastal) |
| 3 | Ruin | 3 | No — can revert to Forest/Plains or be reclaimed |
| 4 | Forest | 4 | Mostly — can reclaim ruins; provides food |
| 5 | Mountain | 5 | Yes — never changes |

## 2. Map Generation (from map seed)

[CONFIRMED] Deterministic from map seed. Steps:

1. **Ocean border**: surrounds entire map
2. **Fjords**: cut inland from random edges
3. **Mountain chains**: formed via random walks
4. **Forest patches**: clustered groves on land
5. **Initial settlements**: placed on land cells, spaced apart

[UNKNOWN] Exact generation algorithm (spacing rules, fjord length distribution, mountain walk params, forest cluster sizes).

> The map seed is visible to us. Initial grid is provided via API. We do NOT need to regenerate maps — we receive them.

## 3. Simulation Lifecycle

**Duration**: 50 years (time steps) [CONFIRMED]

Each year executes 5 phases in order: [CONFIRMED]

### Phase 1: Growth

- Settlements produce food based on adjacent terrain [CONFIRMED]
  - [INFERRED] Forest adjacency increases food production
  - [UNKNOWN] Exact food formula (base + per-adjacent-forest? diminishing returns?)
- Population grows when food is sufficient [CONFIRMED]
  - [UNKNOWN] Growth rate formula, carrying capacity
- Coastal settlements develop ports [CONFIRMED]
  - [UNKNOWN] Trigger condition (population threshold? wealth threshold? automatic?)
- Ports build longships [CONFIRMED]
  - [UNKNOWN] Longship construction cost/time
- Prosperous settlements **expand**: found new settlements on nearby land [CONFIRMED]
  - [UNKNOWN] Expansion radius, prosperity threshold, placement rules

### Phase 2: Conflict (Raiding)

- Settlements raid each other [CONFIRMED]
- Longships extend raiding range "significantly" [CONFIRMED]
  - [UNKNOWN] Base raid range vs. longship-extended range
- Low food settlements raid more aggressively [CONFIRMED]
  - [UNKNOWN] Desperation threshold, aggression scaling
- Successful raids: loot resources, damage defender [CONFIRMED]
  - [UNKNOWN] Combat resolution formula (attack vs. defense)
  - [UNKNOWN] Loot calculation
- Conquered settlements can change faction (owner_id) [CONFIRMED]
  - [UNKNOWN] Allegiance change probability/conditions

### Phase 3: Trade

- Ports within range trade if not at war [CONFIRMED]
  - [UNKNOWN] Trade range, "at war" definition (same faction? recent raids?)
- Trade generates wealth and food for both parties [CONFIRMED]
  - [UNKNOWN] Trade value formula
- Technology diffuses between trading partners [CONFIRMED]
  - [UNKNOWN] Tech diffusion rate, tech's effect on other mechanics

### Phase 4: Winter

- Severity varies per year [CONFIRMED]
  - [UNKNOWN] Severity distribution (uniform? trending? hidden parameter?)
- All settlements lose food [CONFIRMED]
  - [UNKNOWN] Food loss formula (flat? proportional to severity?)
- Settlements can collapse from: starvation, sustained raids, harsh winters [CONFIRMED]
  - Collapsed settlement becomes **Ruin** [CONFIRMED]
  - Population disperses to nearby friendly settlements [CONFIRMED]
  - [UNKNOWN] Collapse threshold (food < 0? multi-factor?)

### Phase 5: Environment

- Nearby thriving settlements can **reclaim ruins** [CONFIRMED]
  - New outpost inherits portion of patron's resources/knowledge [CONFIRMED]
  - Coastal ruins can be restored as ports [CONFIRMED]
  - [UNKNOWN] Reclaim probability, "thriving" threshold, inheritance fraction
- Unreclaimed ruins: eventually overtaken by forest or revert to plains [CONFIRMED]
  - [UNKNOWN] Forest reclamation rate vs. plains reversion rate, time delay

## 4. Settlement Properties

Each settlement tracks: [CONFIRMED]

| Property | Visible Initially? | Visible in Sim Query? |
|----------|-------------------|-----------------------|
| position (x, y) | Yes | Yes |
| has_port | Yes | Yes |
| alive | Yes | Yes |
| population | No | Yes (float) |
| food | No | Yes (float) |
| wealth | No | Yes (float) |
| defense | No | Yes (float) |
| tech_level | No | No (implied by docs, never shown in API) |
| owner_id (faction) | No | Yes |
| has_longship | No | No (implied by docs, never shown in API) |

> Note: population, food, wealth, defense are **floats** in API responses (e.g., population=2.8).

## 5. Hidden Parameters

[CONFIRMED] Each round has hidden parameters that are:
- Same for all 5 seeds within a round
- Different between rounds
- Control world behavior

[UNKNOWN] What the hidden parameters are. Likely candidates:
- Winter severity distribution/mean
- Raid aggression scaling
- Expansion probability/threshold
- Trade range/value
- Food production rates
- Growth rates
- Combat resolution weights
- Forest reclamation speed

## 6. Stochasticity

- Same map seed + same hidden params = different outcomes per sim run [CONFIRMED]
- Each API query uses a different random sim_seed [CONFIRMED]
- Ground truth is computed from **hundreds** of Monte Carlo runs [CONFIRMED]

Sources of randomness (per simulation run):
- [INFERRED] Raid target selection
- [INFERRED] Combat outcomes
- [INFERRED] Expansion placement
- [INFERRED] Winter severity per year
- [INFERRED] Trade partner selection
- [INFERRED] Ruin reclamation/forest growth

## 7. Scoring Rules

[ALL CONFIRMED]

```
KL(p || q) = sum( p_i * log(p_i / q_i) )     # per cell
entropy(cell) = -sum( p_i * log(p_i) )         # per cell

weighted_kl = sum( entropy(cell) * KL(truth[cell], pred[cell]) ) / sum( entropy(cell) )

score = max(0, min(100, 100 * exp(-3 * weighted_kl)))
```

- Only dynamic cells (entropy > 0) contribute to scoring
- Higher-entropy cells are weighted more
- Round score = average of 5 seed scores
- Leaderboard = best(round_score * round_weight) across all rounds
- Round weight = 1.05^round_number (later rounds worth more)

## 8. Digital Twin Implementation Priorities

### Must-have (affects scoring directly):
1. Settlement growth/expansion logic — determines where new settlements appear
2. Settlement collapse — determines where ruins form
3. Port formation — Settlement vs. Port distinction matters for class probabilities
4. Ruin reclamation vs. forest takeover — determines Ruin vs. Forest vs. Empty transitions
5. Correct stochasticity — twin must produce a realistic distribution, not a single trajectory

### Nice-to-have (indirect scoring impact):
6. Faction/raiding dynamics — affects which settlements survive
7. Trade mechanics — affects settlement prosperity → survival
8. Technology — affects growth rates

### Not needed:
9. Exact resource values — we care about terrain class outcomes, not internal stats
10. Exact combat formulas — approximate raid success probability is sufficient

## 9. Key Unknowns to Reverse-Engineer

Priority unknowns for building a useful twin:

| Unknown | Impact | How to Discover |
|---------|--------|-----------------|
| Expansion rules | HIGH — new settlements are main terrain change | Compare initial vs. sim grids across queries |
| Collapse conditions | HIGH — creates ruins | Track which settlements disappear |
| Port formation trigger | MEDIUM — Settlement vs. Port class | Observe coastal settlements across sims |
| Forest reclamation rate | MEDIUM — Ruin→Forest transition | Observe ruins over time |
| Winter severity range | MEDIUM — drives collapses | Correlate food levels with survival |
| Raid range / aggression | LOW-MEDIUM — indirect via collapses | Compare faction-adjacent settlements |
| Hidden parameter ranges | HIGH — needed to calibrate twin per round | Cross-round analysis + within-round observations |
