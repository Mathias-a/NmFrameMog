# Astar Island — Reference

This is reference material. Look things up here as needed. Do not try to memorise it.

## The prediction problem

A black-box Norse settlement simulator runs a procedurally generated 40×40 map for
50 years. You cannot see inside it. Your goal is to submit a H×W×6 probability tensor
predicting what each cell is likely to be after 50 years — not a single outcome, but a
full probability distribution across 6 terrain classes.

The organizers run the simulator hundreds of times with fixed hidden parameters and
aggregate the results into the ground truth tensor. Your prediction is scored against
that ground truth using entropy-weighted KL divergence.

## Terrain classes and codes

| API code | Terrain    | Prediction class | Dynamic?                         |
| -------- | ---------- | ---------------- | -------------------------------- |
| 10       | Ocean      | 0 (Empty)        | No — static border               |
| 5        | Mountain   | 5                | No — static                      |
| 11       | Plains     | 0 (Empty)        | Yes                              |
| 0        | Empty      | 0 (Empty)        | Yes                              |
| 4        | Forest     | 4                | Mostly static; ruins can become forest |
| 1        | Settlement | 1                | Yes                              |
| 2        | Port       | 2                | Yes — coastal settlement only    |
| 3        | Ruin       | 3                | Yes — collapsed settlement       |

Ocean (10), Plains (11), Empty (0) all map to **prediction class 0**.
Cells with near-zero entropy (including static types 5 and 10) are excluded from scoring.

## Map generation

Each map is procedurally generated from a visible map seed. The `initial_grid` returned
by the `/analysis` endpoint gives the full starting layout — no queries are needed to
discover the map. All viewport positions can be pre-planned before spending any budget.

Generation follows this structure:

- Ocean borders form the outer edge of the 40×40 grid
- **The ocean border is static** — these cells are always ocean (class 0) with zero entropy.
  Viewports placed on the border waste coverage on cells excluded from scoring.
- Fjords cut inland from random edges (ocean cells extending into land — increases coastal-adjacent cell count)
- Mountain chains form via random walks (mountains cluster, not scatter randomly)
- Forest patches cover land as clustered groves (forest neighbours cluster spatially)
- Initial settlements are placed on land cells, spaced apart from each other

**Implications for modelling:**
Fjords extend the coastal mask inland — more cells are ocean-adjacent than a naive border
check would suggest. Forest clustering means forest-neighbour counts are spatially
correlated: cells near initial forest patches are more likely to have multiple forest
neighbours, boosting food gain. Mountain clustering means defense bonuses concentrate in
specific map regions. Settlement spacing reduces early inter-faction proximity, so early
conflict is lower than if settlements were randomly placed.

## Simulation lifecycle

50 years. Each year runs these 5 phases **in strict order**:

```
GROWTH      → food produced from terrain; ports acquired; expansion triggered
CONFLICT    → settlements raid enemies; food looted; sometimes faction is conquered
TRADE       → ports within range exchange wealth, food, tech (only if not at war)
WINTER      → food drain; starvation causes population loss and collapse → ruin
ENVIRONMENT → ruins reclaimed by neighbours; or eventually become forest/plains
```

Phase order is causal. Growth boosts stats before Conflict uses them.
Winter kills on post-Trade stats. Environment restores after Winter.
A change to any single phase's parameters propagates through subsequent phases.

## Settlement mechanics (what drives simulation outcomes)

**Settlement tracks internally:** position, population, food, wealth, defense,
tech_level, port status, longship ownership, owner_id (faction).

**Growth phase (per settlement per year):**

```
food_gain  = forest_neighbors × forest_mult + plains_neighbors × plains_mult
population grows proportional to food_gain
```

Food production is the primary growth driver. Adjacent forest cells contribute more than plains.
Wealth and defense are not produced in the growth phase — wealth accumulates via trade, defense is set at founding.

Port acquired if coastal (probabilistic — threshold parameters hidden).
Longship acquired if has_port (probabilistic — threshold parameters hidden).
Expansion triggered if population ≥ threshold → new settlement on adjacent plains cell.

**Conflict phase:**

```
raid_chance = base + starvation_bonus (if food < threshold) + longship_bonus
attack      = pop_weight×pop + wealth_weight×wealth + defense_weight×defense + tech_weight×tech + longship_bonus + random
defense     = pop_weight×pop + defense_weight×defense + wealth_weight×wealth + tech_weight×tech + random
```

Longships extend raiding **range** significantly, not just attack strength. A settlement
with longships can raid targets far beyond its immediate neighbours.

If attacker wins: defender loses food (looted by attacker). Sometimes the conquered
settlement changes allegiance to the raiding faction (owner_id changes). A single raid
does not directly produce a ruin — collapse to ruin occurs via sustained food drain
leading to starvation in the Winter phase.

**Trade phase:**

Trade only occurs between ports that are **not at war** (no active raiding between their
factions) and within a configurable range. In conflict-heavy worlds, warring factions
block trade even when ports are in range — coastal port clusters do not automatically
gain trade benefits if factions are hostile.

Both ports gain food and wealth. Tech diffuses between trading partners: the gap between
their tech levels narrows each trade cycle (the lower-tech port gains more).

**Winter phase:**

```
severity  = drawn from a distribution each year   ← varies every year
food_loss = settlement.food × severity
pop_loss  = settlement.population × severity × fraction
```

Collapse if: food ≤ collapse_threshold OR population ≤ 0.
On collapse: half of remaining population disperses to the nearest **friendly** settlement
(same owner_id). This boosts that neighbour, meaning friendly clusters survive longer than
isolated settlements would.

**Environment phase:**

- Nearby thriving settlements (pop > threshold) can reclaim ruins: new settlement inherits partial stats
- **Coastal ruins** reclaimed by a nearby settlement are restored as **ports** (class 2), not plain settlements (class 1). Inland ruins reclaim as class 1 only.
- Unclaimed ruins eventually become forest or plains (exact probabilities are hidden parameters)

## What queries return

**Initial state** (`GET /astar-island/rounds/{round_id}`): settlement objects contain only
`x, y, has_port, alive`. No internal stats are available before querying.

`POST /astar-island/simulate` runs one independent 50-year simulation from t=0 and returns:

**Grid:** terrain codes for the viewport region only (5–15 cells per side).

**Always use 15×15 viewports** to maximise observed entropy per query. The API allows
5–15 cells per side; anything less than 15×15 wastes budget on fewer cells.

**Settlements in viewport** — only terrain codes 1 (Settlement) and 2 (Port) produce stat
objects. Ruins, forests, plains, ocean, and mountains appear only as grid codes.

```
x, y, population, food, wealth, defense, has_port, alive, owner_id
```

Collapsed settlements appear with `alive: false` and retain their final stats. `food` and
`wealth` reflect values at collapse time. `population` is zeroed after dispersal (always 0
on dead settlements). The useful inference signal is `food` ≤ 0 indicating starvation collapse.

`tech_level` and `has_longship` are **never returned** by any endpoint.

**Each query is a fresh independent simulation.** You are not stepping through one run.
You get a different stochastic outcome every time for the same viewport. Each outcome
is a single sample from the same underlying distribution that the ground truth
represents — the ground truth is the aggregate of hundreds of such samples. Repeated
queries over the same viewport converge to the ground truth probabilities.

**Budget: 50 queries total per round, shared across all 5 seeds.**

**Round timing: each round lasts approximately 160 minutes.** The entire window is
available for prediction — there is no separate "observation" vs "submission" phase.
You can query, compute, and submit at any point within the round.

## Hidden parameters

Every numeric coefficient in every formula is a hidden parameter. The admin sets these
at round creation. They are **never exposed by the API**, even after a round closes.

They are **fixed for all 5 seeds within a round** but **vary between rounds**.

## Historical calibration data

The `/analysis` endpoint returns ground truth for **all** completed rounds — including
rounds your team never participated in. The `prediction` field will be null and `score`
will be null, but `ground_truth` and `initial_grid` are always present.

```
GET https://api.ainm.no/astar-island/analysis/{round_id}/{seed_index}
Authorization: Bearer <ASTAR_API_TOKEN from .env>
→ { ground_truth: float[][][6], initial_grid: int[][], score: float|null, ... }
```

`ground_truth[y][x]` is a 6-float probability vector computed from hundreds of
server rollouts with the true hidden parameters. `initial_grid` is the starting map.

This is **supervised signal**: you know the input (initial_grid) and the output
(ground_truth) for every completed round. Use all of them for calibration.

**All completed rounds — every one is usable regardless of participation:**

## Scoring formula

```
score = max(0, min(100,  100 × exp(-3 × weighted_KL)))

weighted_KL = Σ_cells  entropy(cell) × KL(ground_truth[cell], prediction[cell])
              ──────────────────────────────────────────────────────────────────
                              Σ_cells  entropy(cell)

entropy(cell)  = −Σᵢ pᵢ × log(pᵢ)
KL(p || q)     =  Σᵢ pᵢ × log(pᵢ / qᵢ)
```

Only cells where entropy > 0 contribute (static cells are excluded).
Cells with higher entropy count more heavily — the scoring focuses on uncertain,
interesting regions of the map.

**KL divergence blows up only when pᵢ > 0 AND qᵢ = 0.**
The term pᵢ × log(pᵢ / qᵢ) is 0 whenever pᵢ = 0, regardless of qᵢ.
So assigning near-zero is safe — and more accurate — whenever the ground truth
is guaranteed to also assign 0.

## Probability floor — class-specific rules

Confirmed from ground truth data across 14 completed rounds:

| Class      | Cell condition                | Ground truth max observed | Assignment             |
| ---------- | ----------------------------- | ------------------------- | ---------------------- |
| 5 Mountain | Non-mountain dynamic cell     | 0.000000 (exact)          | near-zero (e.g. 0.001) |
| 2 Port     | Non-coastal dynamic cell      | 0.000000 (exact)          | near-zero (e.g. 0.001) |
| 2 Port     | Coastal dynamic cell          | up to 0.59                | standard floor 0.01    |
| 0–4        | Static cells (5, 10)          | excluded from score       | one-hot, irrelevant    |
| all others | Dynamic cell, plausible class | varies                    | standard floor 0.01    |

**Rules:**

1. Static cells (code 5, 10) are excluded from scoring. Assign one-hot on their class.
2. Mountain class (5) on non-mountain dynamic cells: assign near-zero (0.001).
   Ground truth is exactly 0 here — mountains are never created by the simulation.
   Using 0.001 instead of 0.0 is defensive against any floating-point edge case
   in the ground truth while wasting negligible probability mass.
3. Port class (2) on non-coastal dynamic cells: assign near-zero (0.001).
   Ground truth is exactly 0 for inland cells — ports require ocean adjacency.
4. All other uncertain class-cell combinations: apply standard floor of 0.01.
5. After applying floors, renormalise so all 6 values sum to 1.0.

This approach concentrates probability mass on plausible outcomes rather than
spreading it uniformly across impossible ones. For a coastal plains cell, the
saved mass from near-zeroing mountain goes into the classes that actually matter.

Round score = average of 5 seed scores. Missing seed = 0. Always submit all 5.

## Phase lifecycle → parameter estimation strategy

Because phases run in fixed order, each phase leaves a specific footprint in the
final observed state. Queries can be designed to isolate these footprints:

| Phase       | What to look for                                                           | Best query target                                 |
| ----------- | -------------------------------------------------------------------------- | ------------------------------------------------- |
| Growth      | Population and wealth levels; port presence; expansion density             | Isolated settlement with only forest neighbours   |
| Conflict    | Ruin density near initial settlements; faction spread (owner_id diversity) | Multi-faction border regions                      |
| Trade       | Wealth elevation in coastal port clusters vs isolated ports                | Two initial ports close enough to trade           |
| Winter      | Settlement survival rate; low population on well-resourced land            | Full coverage of initial settlement positions     |
| Environment | Ruins converted to forest; new settlements on former ruins                 | Areas with initial ruins or collapsed settlements |
