# NmFrameMog Monorepo

Monorepo containing three task-focused Python projects:

- Object Detection
- AI Accounting Agent
- A\* Island

## Structure

```text
apps/
	object-detection/
	ai-accounting-agent/
	astar-island/
src/
	nmframemog/  # legacy root package
```

Each app has its own `pyproject.toml`, `src/` package, and `tests/` directory.

## Development setup

This repository uses [uv](https://github.com/astral-sh/uv) with Python 3.13.

```bash
uv python install 3.13
uv sync
```

## Quality checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

## Run a specific app

```bash
uv run -p apps/object-detection python -m object_detection.pipeline
uv run -p apps/ai-accounting-agent python -m ai_accounting_agent.agent
uv run -p apps/astar-island python -m astar_island.solver
```
