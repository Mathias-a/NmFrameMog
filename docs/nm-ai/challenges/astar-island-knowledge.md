# Astar Island — Solver Knowledge Reference

> Source: `docs/nm-ai/challenges/astar-island.md` (MCP excerpts).
> All `[UNCLEAR: ...]` tags mark gaps not resolvable from the source alone.

---

## Task Definition

**Problem statement:** Observe a black-box Norse civilisation simulator through a limited viewport and predict the final world state after 50 simulated years.

**Goal:** Predict the probability distribution of terrain types across the entire map.

**Task type:** Observation + probabilistic prediction (calibration challenge, not plain classification).

**Time horizon:** 50 simulated years per round.

**World dimensions:** Rectangular grid, default **40 × 40** cells.

**Input/output:**

- Input: limited viewport observations from the simulator + initial world state
- Output per seed: a `W × H × 6` probability tensor, one 6-vector per cell
- Each 6-vector: probability of each terrain class (must sum to 1.0 per cell)
- dtype: float (Python list of lists of lists sent as JSON) `[UNCLEAR: exact float precision — use float64 / Python native float]`

---

## Terrain & Classes

The world has **8 terrain types** that collapse into **6 prediction classes** (0–5):

| Internal Code | Terrain    | Class Index | Description                           |
| ------------- | ---------- | ----------- | ------------------------------------- |
| 0             | Empty      | 0           | Generic empty cell                    |
| 10            | Ocean      | 0 (Empty)   | Impassable water, borders the map     |
| 11            | Plains     | 0 (Empty)   | Flat land, buildable                  |
| 1             | Settlement | 1           | Active Norse settlement               |
| 2             | Port       | 2           | Coastal settlement with harbour       |
| 3             | Ruin       | 3           | Collapsed settlement                  |
| 4             | Forest     | 4           | Provides food to adjacent settlements |
| 5             | Mountain   | 5           | Impassable terrain                    |

**Many-to-one mapping:** Internal codes 0 (Empty), 10 (Ocean), and 11 (Plains) all map to **class 0**. Predictions never need to distinguish between them.

**Terrain transition rules (from mechanics source):**

- Mountains are **static** — never change.
- Forests are **mostly static** but can reclaim ruined land.
- Settlements can collapse into Ruins (starvation, raids, harsh winters).
- Ruins can be rebuilt as Settlements for Ports by nearby thriving settlements.
- Ruins with no sponsor eventually become Forest or revert to open plains (class 0).
- Coastal ruins can be restored as Ports.
- The most dynamic cells: those that become Settlements (1), Ports (2), or Ruins (3).

**Simulation phases per year (50 total):**

1. **Growth** — settlements produce food, grow population, build ports/longships, found new settlements on nearby land.
2. **Conflict** — settlements raid each other; longships extend range; desperate (low food) settlements raid more aggressively; conquered settlements can flip allegiance.
3. **Trade** — ports in range and not at war trade, generating wealth/food and diffusing technology.
4. **Winter** — all settlements lose food; starvation/raids/harsh winters can collapse settlements into Ruins.
5. **Environment** — nature reclaims ruins (forest growth) or settlements rebuild them.

---

## Observation / Viewport

**What is visible:**

- Initial terrain grid (full map, height × width terrain codes)
- Initial settlement positions and port status (`{x, y, has_port, alive}`)
- [UNCLEAR: exact initial_states structure — assumed from quickstart reference]

**What is hidden:**

- Internal settlement stats: population, food, wealth, defense, tech level, longship count, faction allegiance (`owner_id`) — only visible through simulator queries

**Simulator queries:**

- You have **50 queries per round**, shared across all seeds.
- Each query returns a **5–15 cell wide viewport** (you specify position and size).
- Returns: partial terrain grid after simulation + full settlement stats for settlements in the viewport.
- `[UNCLEAR: exact query budget enforcement — whether exceeding 50 returns an error or is silently ignored]`

**Seeds:**

- Each round has multiple seeds (default **5** per round, from quickstart reference).
- Each seed is a different procedurally generated map + simulation run.
- The 50 query budget is shared across all seeds — allocate carefully.
- The map seed is visible — you can reconstruct the initial terrain layout locally.

---

## Submission Format

**Tensor shape:** `height × width × 6`

> **Axis ordering is height-first (row-major / NumPy default):**
> `prediction[y][x] = [p_empty, p_settlement, p_port, p_ruin, p_forest, p_mountain]`
> i.e., shape is `(H, W, 6)` = `(40, 40, 6)`.

**Class order in the 6-vector (indices 0–5):**

| Index | Class                             |
| ----- | --------------------------------- |
| 0     | Empty (includes Ocean and Plains) |
| 1     | Settlement                        |
| 2     | Port                              |
| 3     | Ruin                              |
| 4     | Forest                            |
| 5     | Mountain                          |

**Constraints:**

- Each 6-vector must sum to **1.0**.
- No value may be **0.0** (see Critical Rule below).
- Minimum floor: **0.01** per class, then renormalize.

**POST endpoint payload:**

```python
{
    "round_id": round_id,       # str/int — active round identifier
    "seed_index": seed_idx,     # int — 0-indexed seed (0 to seeds_count-1)
    "prediction": prediction.tolist(),  # list[list[list[float]]] — shape H×W×6
}
```

**Submit URL:** `POST https://api.ainm.no/astar-island/submit`

**Baseline:** uniform prediction (`np.full((H, W, 6), 1/6)`) scores approximately **1–5** (used for calibration reference).

---

## Scoring

**Metric:** Entropy-weighted KL divergence (lower is better).

**KL divergence per cell:**

```
KL(p || q) = Σ_{c=0}^{5} p_c * log(p_c / q_c)
```

Where:

- `p` = ground-truth distribution for the cell (one-hot in practice, but expressed as a distribution)
- `q` = your predicted distribution for the cell
- Sum is over all 6 classes

**Entropy weighting:**

- Each cell's KL divergence is weighted by the **entropy of the ground-truth distribution** for that cell.
- `[UNCLEAR: exact entropy-weight formula — likely H(p) = -Σ p_c * log(p_c), or a normalized variant]`
- Cells where the ground truth is highly uncertain (high entropy) contribute more to the score.
- Static cells (Mountains: always class 5) have zero entropy and therefore zero weight — getting them right/wrong does not affect score.

**Aggregation:**

- `[UNCLEAR: exact aggregation formula — likely sum or mean of (weight_i * KL_i) over all cells and seeds]`

**Score direction:** **Lower is better.** Entropy-weighted KL divergence of 0 is perfect.

### Critical Rule: Never assign 0.0 probability

If ground truth has non-zero mass on class `c` and your prediction has `q_c = 0.0`:

```
p_c * log(p_c / 0.0) → +∞
```

This makes your score for that cell **infinite**, destroying the entire round score.

**Mandatory mitigation:**

```python
import numpy as np

MIN_PROB = 0.01
prediction = np.clip(prediction, MIN_PROB, 1.0)
# Renormalize so each cell sums to 1.0
prediction /= prediction.sum(axis=-1, keepdims=True)
```

Apply this to **every** prediction tensor before submission, without exception.

---

## API Endpoints

Base URL: `https://api.ainm.no`

| Method | Path                                             | Description                                              |
| ------ | ------------------------------------------------ | -------------------------------------------------------- |
| `GET`  | `/astar-island/rounds`                           | List all rounds; filter for `status == "active"`         |
| `GET`  | `/astar-island/rounds/{round_id}`                | Full round detail: map dims, seeds_count, initial_states |
| `POST` | `/astar-island/simulate`                         | Query the simulator (viewport observation)               |
| `POST` | `/astar-island/submit`                           | Submit prediction tensor for one seed                    |
| `GET`  | `/astar-island/my-predictions/{round_id}`        | Retrieve your submitted predictions for a round          |
| `GET`  | `/astar-island/analysis/{round_id}/{seed_index}` | Ground truth vs. your prediction (post-round only)       |
| `GET`  | `/astar-island/leaderboard`                      | Current leaderboard                                      |

### GET `/astar-island/rounds`

- Response: list of round objects with at least `{id, round_number, status}`

### GET `/astar-island/rounds/{round_id}`

- Response fields: `map_width`, `map_height`, `seeds_count`, `initial_states`
- `initial_states[i].grid` — `height × width` array of terrain codes
- `initial_states[i].settlements` — list of `{x, y, has_port, alive}`

### POST `/astar-island/simulate`

- Request body:

```json
{
  "round_id": "<round_id>",
  "seed_index": 0,
  "viewport_x": 10,
  "viewport_y": 5,
  "viewport_w": 15,
  "viewport_h": 15
}
```

- Viewport dimensions: 5–15 cells wide/tall `[UNCLEAR: can w and h differ? assumed yes]`
- Response: `{grid: [[...]], settlements: [{x, y, ...full_stats}], viewport: {x, y, w, h}}`
- Budget: **50 queries per round, shared across all seeds**

### POST `/astar-island/submit`

- Request body: `{round_id, seed_index, prediction: H×W×6 nested list}`
- Response: HTTP status code; `[UNCLEAR: response body schema]`
- Can resubmit (overwrites previous) `[UNCLEAR: whether resubmission is allowed]`

### GET `/astar-island/my-predictions/{round_id}`

- Returns previously submitted predictions for the round

### GET `/astar-island/analysis/{round_id}/{seed_index}`

- Available only **after the round completes**
- Returns your prediction alongside the ground truth for the specified seed
- Use this to measure per-cell error and recalibrate your model

### GET `/astar-island/leaderboard`

- `[UNCLEAR: response schema]`

**Rate limits:** `[UNCLEAR: not specified in source]`

---

## Authentication

**Method:** JWT access token, obtained by logging in at `app.ainm.no` and inspecting cookies.

**Two options (equivalent):**

```python
BASE = "https://api.ainm.no"
session = requests.Session()

# Option A: Cookie
session.cookies.set("access_token", "YOUR_JWT_TOKEN")

# Option B: Bearer header
session.headers["Authorization"] = "Bearer YOUR_JWT_TOKEN"
```

Use `session.get(...)` / `session.post(...)` for all requests — the token is sent automatically.

---

## Strategy Hints & Implementation Loop

### Recommended iteration loop

```
1. Fetch active round → get round_id, W, H, seeds_count, initial_states
2. For each seed: reconstruct initial terrain from grid codes
3. Use simulate queries to sample viewport observations (budget: 50 total across seeds)
4. Build prediction tensor from observations (start uniform, refine per observation)
5. Apply probability floor (clip to 0.01, renormalize)
6. Submit for each seed
7. After round ends: fetch analysis for each seed → compare prediction vs. ground truth
8. Identify systematic errors → improve model → repeat next round
```

### Baseline strategy

- Uniform prediction: `np.full((H, W, 6), 1/6)` → scores ~1–5
- Obvious improvements: Mountains are static → set class 5 to ~0.95 for mountain cells; class 0 for ocean/border cells

### High-value cells to focus on

- Settlements (class 1), Ports (class 2), Ruins (class 3) are the most dynamic and uncertain
- Mountains (class 5) are static — always predict high confidence for mountain cells
- Ocean border cells — static, always class 0

### Simulation-informed priors

- Settlements near forests and coasts are more likely to survive and grow
- Settlements far from others are safer from raids
- Low-food settlements in dense areas are raid targets → likely to collapse to Ruins
- Coastal settlements have port potential → non-zero Port probability
- Ruins near thriving settlements may recover → non-zero Settlement/Port probability
- Isolated ruins → high Forest or Empty probability

### Pitfalls

1. **Zero probability** — destroys score for that cell. Always floor at 0.01.
2. **Ignoring the initial state** — the initial terrain grid is fully visible; use it to set strong priors for static terrain (Mountains, Ocean).
3. **Wasting query budget** — 50 queries across 5 seeds = 10 per seed; prioritize high-uncertainty regions (settlement clusters, contested zones).
4. **Not recalibrating** — use the analysis endpoint after each round to measure and correct systematic biases.
5. **Not accounting for many-to-one terrain mapping** — Ocean, Plains, and Empty all map to class 0; treat the border as known-class-0.

---

## Quick-Reference: Implementation Skeleton

```python
import numpy as np
import requests

BASE = "https://api.ainm.no"
TOKEN = "YOUR_JWT_TOKEN"

session = requests.Session()
session.headers["Authorization"] = f"Bearer {TOKEN}"

# 1. Get active round
rounds = session.get(f"{BASE}/astar-island/rounds").json()
active = next(r for r in rounds if r["status"] == "active")
round_id = active["id"]

# 2. Get round details
detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()
W = detail["map_width"]   # 40
H = detail["map_height"]  # 40
seeds = detail["seeds_count"]  # 5

# 3. Submit predictions
MIN_PROB = 0.01

for seed_idx in range(seeds):
    prediction = np.full((H, W, 6), 1.0 / 6)  # uniform baseline

    # --- your model here ---
    # prediction[y][x] = [p_empty, p_settlement, p_port, p_ruin, p_forest, p_mountain]

    # Mandatory floor + renormalize
    prediction = np.clip(prediction, MIN_PROB, 1.0)
    prediction /= prediction.sum(axis=-1, keepdims=True)

    session.post(f"{BASE}/astar-island/submit", json={
        "round_id": round_id,
        "seed_index": seed_idx,
        "prediction": prediction.tolist(),
    })

# 4. (After round) Analyze errors
for seed_idx in range(seeds):
    analysis = session.get(
        f"{BASE}/astar-island/analysis/{round_id}/{seed_idx}"
    ).json()
    # Compare prediction vs ground truth, update model
```
