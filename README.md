# NmFrameMog
Faen som vi frame mogger

## Development setup

This project uses [uv](https://github.com/astral-sh/uv) with Python 3.13.

```bash
uv python install 3.13
uv sync
```

### Quality checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
```
