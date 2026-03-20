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

## Task packages

- `src/task_norgesgruppen_data`: offline COCO-style prediction runner with root `run.py`
- `src/task_tripletex`: FastAPI `POST /solve` service for structured Tripletex API execution
- `src/task_astar_island`: authenticated client and CLI for fetch / predict / submit workflows

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
uv run pytest
```

## Task-specific entrypoints

NorgesGruppen Data:

```bash
python run.py --input /data/images --output /output/predictions.json
```

Tripletex service:

```bash
PYTHONPATH=src uv run uvicorn task_tripletex.service:app --reload
```

Astar Island CLI:

```bash
PYTHONPATH=src uv run python -m task_astar_island.cli --help
```

To fetch third-party dependencies before running the new task packages:

```bash
uv sync
```

## Run a specific app

```bash
uv run --package object-detection python -m object_detection.pipeline
uv run --package ai-accounting-agent python -m ai_accounting_agent.agent
uv run --package astar-island python -m astar_island.solver
```
