# NmFrameMog — NM i AI 2026

## Competition
NM i AI (Norwegian Championships in AI), March 19-22 2026. Four challenges:
- **Grocery Bot** — WebSocket real-time pathfinding game (bot navigation in a grocery store)
- **NorgesGruppen Data** — Object detection on grocery shelves (mAP@0.5, Docker submission)
- **Tripletex** — AI accounting agent (HTTPS `/solve` endpoint, Tripletex API integration)
- **Astar Island** — Norse world prediction (W×H×6 probability tensor, KL divergence scoring)

Challenge docs: `docs/nm-ai/challenges/`. Confidence register: `docs/nm-ai/spec-confidence-register.md`.

## Tech Stack
- Python 3.13, uv, ruff, basedmypy (strict)
- Source: `src/nmframemog/`
- Deploy target: Google Cloud Run (see `docs/nm-ai/shared/google-cloud.md`)

## Quality Gates — ALWAYS run before considering work complete
```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
```
A PostToolUse hook auto-runs ruff fix+format after every edit. Mypy must pass manually.

## Code Style
- Complete type annotations on ALL functions — no `Any`
- Python 3.13 features: `X | Y` unions, match statements, modern syntax
- Dataclasses for data structures, Enum for finite sets
- Use `collections`, `itertools`, `functools` from stdlib
- Descriptive names (single letters OK in algorithms: i, j, n, dp)

## Agent Workflow
Use `/solve-challenge <name>` for the full pipeline, or invoke agents individually:
1. **researcher** → problem classification, optimal algorithms, competition docs
2. **architect** → solution design, data models, pseudocode
3. **implementer** → code (type-safe, passes all quality gates)
4. **tester** → comprehensive tests (edge cases, stress tests)
5. **reviewer** → correctness, performance, output format verification
6. **optimizer** → performance tuning if needed
7. **debugger** → systematic root cause analysis if bugs found

For parallel work, use agent teams: "Create an agent team with 3 teammates for X."

## Critical Competition Rules
- NEVER assign probability 0.0 in Astar Island (KL divergence → infinity). Floor at 0.01.
- Tripletex auth: username `0`, password = session_token
- Grocery Bot: 300 max rounds, 60s cooldown, handle `game_over` message properly
- NorgesGruppen: `run.py` must be at zip root, runs in sandboxed Docker
