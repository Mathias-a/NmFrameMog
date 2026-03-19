---
name: quality-check
description: Run all quality gates (ruff lint, ruff format, basedmypy strict) and report results. Use proactively after any code changes.
allowed-tools: Bash
context: fork
agent: reviewer
---

Run all quality checks and report a pass/fail summary:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

If any check fails:
1. List every error with file:line references
2. Categorize as: lint error, format error, or type error
3. Suggest specific fixes for each
4. If there are auto-fixable issues, run `uv run ruff check --fix .` and `uv run ruff format .`

Report a final summary: PASS (all green) or FAIL (with count of remaining issues).
