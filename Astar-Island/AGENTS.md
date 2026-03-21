# Astar Island — Agent Briefing

## Challenge in One Sentence

Observe a stochastic Norse civilisation simulator through limited viewports (50 queries, 15x15 max) and predict the probability distribution of 6 terrain classes across the full 40x40 map for each of 5 seeds.

## What We're Predicting

A **H x W x 6 probability tensor** per seed. Each cell gets probabilities for: Empty(0), Settlement(1), Port(2), Ruin(3), Forest(4), Mountain(5). Scored via entropy-weighted KL divergence. Score = 100 * exp(-3 * weighted_kl), range [0, 100].

## Core Constraints

- **50 queries total** per round, shared across all 5 seeds
- **Viewport**: 5-15 cells wide/tall per query
- **Stochastic**: same map + params produce different outcomes each run
- **Hidden parameters**: control world behavior, same for all seeds in a round
- **Map seed visible**: initial terrain layout is reconstructable
- **Time window**: ~2h45m per round

## Critical Rules

- **NEVER output probability 0.0** — floor at 0.01, renormalize. One zero = infinite KL = destroyed score.
- **Always submit all 5 seeds** — missing seed scores 0.
- Uniform baseline scores ~1-5. Any model beats that.

## Strategy Levers

| Lever | Why It Matters |
|-------|---------------|
| Query allocation | 50 queries across 5 seeds — which areas/seeds to observe? |
| Viewport placement | Cover dynamic regions (settlements), skip static (ocean, mountains) |
| Monte Carlo aggregation | Multiple queries on same seed = empirical distribution |
| Cross-seed transfer | Same hidden params across seeds — learn rules once, apply to all |
| Digital twin | Simulate locally to generate unlimited samples without burning queries |

## The Winning Path

1. **Reverse-engineer the simulation rules** from observations
2. **Build a local digital twin** that approximates the real simulator
3. **Run thousands of local Monte Carlo sims** to generate probability tensors
4. **Use API queries strategically** to calibrate/validate the twin, not as primary data source

## Agent Roles

| Agent | Task |
|-------|------|
| **researcher** | Analyze observations, classify simulation mechanics, identify hidden parameter effects |
| **architect** | Design digital twin data model, Monte Carlo pipeline, query strategy |
| **implementer** | Build twin simulator, API client, prediction pipeline (type-safe, quality-gated) |
| **tester** | Validate twin against real API observations, edge cases, scoring math |
| **reviewer** | Verify output format (H x W x 6), probability constraints, KL safety |
| **optimizer** | Tune twin parameters, query allocation strategy, runtime performance |
| **debugger** | Root-cause when twin diverges from observations |

## Key Data Structures

```
Initial state (visible):  grid[H][W] of terrain codes, settlements[{x, y, has_port, alive}]
Sim response (viewport):  grid[vh][vw], settlements[{x, y, pop, food, wealth, defense, has_port, alive, owner_id}]
Prediction (submit):       prediction[H][W][6] — probabilities summing to 1.0 per cell
```

## API Quick Reference

| Endpoint | Purpose | Auth |
|----------|---------|------|
| `GET /astar-island/rounds` | List rounds, find active | Public |
| `GET /astar-island/rounds/{id}` | Initial states for all seeds | Public |
| `GET /astar-island/budget` | Remaining queries | Team |
| `POST /astar-island/simulate` | Observe one viewport (costs 1 query) | Team |
| `POST /astar-island/submit` | Submit H x W x 6 tensor for one seed | Team |
| `GET /astar-island/analysis/{round_id}/{seed}` | Post-round ground truth (after completion) | Team |
