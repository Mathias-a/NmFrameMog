# Draft: Digital Twin Simulator for Astar Island

## Requirements (confirmed)
- Local digital twin simulator that EXACTLY matches the real API behavior
- API structure mirrors the real API (same request/response shapes)
- Automatic tests verifying API behavior match
- Hundreds of Monte Carlo simulations for probability distributions
- Map generation algorithm pluggable (not needed now, but architecture supports it)
- Simulations follow EVERY rule in the rulebook exactly
- Import all previous round initial states as JSON fixtures
- Verified stochastic output for each round state
- Invariant tests: mountains never appear on non-mountain cells, ocean always stays ocean, ports only adjacent to ocean, etc.

## User Decisions (from interview)
- **Unknown Rules**: Parameterize ALL unknowns into a SimulationParams dataclass (~20-30+ tunable values). Two sub-agents analyzing for missing parameters.
- **Project Location**: `Astar-Island/benchmark/` — code lives here
- **API Server**: FastAPI server with identical endpoint paths, request/response schemas, and error codes
- **Round Data**: JSON fixtures in a `data/` directory inside benchmark/
- **Test Strategy**: TDD (RED-GREEN-REFACTOR) — write invariant tests first, implement to pass them

## Technical Decisions
- Language: Python 3.13 (per CLAUDE.md)
- Quality gates: ruff, basedmypy strict
- Package manager: uv
- Test framework: pytest
- Greenfield implementation: zero existing code in this worktree
- FastAPI for local API server
- numpy for Monte Carlo aggregation and tensor operations
- Code location: Astar-Island/benchmark/

## Research Findings
- RULESET.md exists with detailed rules + confidence levels (CONFIRMED/INFERRED/UNKNOWN)
- 50 years of simulation, 5 phases per year: Growth → Conflict → Trade → Winter → Environment
- 8 terrain codes mapping to 6 prediction classes
- Extensive list of UNKNOWN mechanics (exact formulas for food, growth, combat, trade, etc.)
- worktree-4 has round_9_implementation with belief-based rollout model — NOT physics simulation
- worktree-4 rollout_model.py has many magic numbers that can inform default parameter ranges
- API endpoint spec fully documented in docs/endpoint.md
- No existing Python code in this worktree — everything from scratch
- Previous plan scored 2.52 (uniform) to 21.29 (phase1_rules) on benchmark

## Confirmed Simulation Rules (from RULESET.md)
### Static terrain invariants
- Ocean (10): NEVER changes. Borders map.
- Mountain (5): NEVER changes. Nothing becomes mountain.
- Ports: ONLY valid on coastal cells (adjacent to ocean)
- Plains (11) / Empty (0): Can become settlement/port via expansion
- Forest (4): Mostly stable, can reclaim ruins, can be colonized

### 5-Phase Year Cycle (50 years)
1. Growth: food production, population growth, port formation, longship building, expansion
2. Conflict: raiding, combat, loot, allegiance changes
3. Trade: port-to-port trade, wealth/food generation, tech diffusion
4. Winter: severity varies, food loss, collapse → ruin, population dispersal
5. Environment: ruin reclamation, forest takeover, plains reversion

### Settlement Properties
- position (x,y), has_port, alive, population (float), food (float), wealth (float), defense (float), tech_level (hidden), owner_id, has_longship (hidden)

## Open Questions
- [RESOLVED] Project location → Astar-Island/benchmark/
- [RESOLVED] Unknown rules → Parameterize all
- [RESOLVED] Round data → JSON fixtures
- [RESOLVED] API server → FastAPI
- [RESOLVED] Testing → TDD
- [PENDING] Complete parameter list — awaiting two sub-agent analyses
- [PENDING] Dependencies: need numpy, fastapi, uvicorn — anything else?

## Scope Boundaries
- INCLUDE: Full simulation engine (5-phase yearly cycle, 50 years)
- INCLUDE: FastAPI local server matching real API behavior
- INCLUDE: Monte Carlo aggregation (hundreds of runs → H×W×6 tensor)
- INCLUDE: Invariant tests, simulation correctness tests
- INCLUDE: JSON fixtures for round data
- INCLUDE: Pluggable map generation interface (protocol/ABC)
- INCLUDE: SimulationParams dataclass with all tunable parameters
- INCLUDE: Scoring function implementation for local validation
- EXCLUDE: Query strategy/budget optimization
- EXCLUDE: Real API interaction client (separate concern)
- EXCLUDE: Frontend/visualization
- EXCLUDE: Posterior inference / particle filters / belief systems
- EXCLUDE: Calibration against real API observations (future work)
