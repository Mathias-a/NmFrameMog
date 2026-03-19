---
name: solve-challenge
description: Full pipeline for solving an NM i AI competition challenge. Orchestrates research, architecture, implementation, testing, and review.
disable-model-invocation: true
argument-hint: [challenge-name]
---

Solve the NM i AI challenge: $ARGUMENTS

## Pipeline

Execute this workflow step by step. Use subagents for each phase to preserve context.

### Phase 1: Research
Use the **researcher** agent to:
- Read challenge docs from `docs/nm-ai/challenges/`
- Check the confidence register at `docs/nm-ai/spec-confidence-register.md`
- Use the nmiai MCP server for additional details if available
- Classify the problem and identify optimal approaches
- Report: problem type, recommended algorithm/approach, key constraints, scoring formula

### Phase 2: Architecture
Use the **architect** agent to:
- Design the solution based on research findings
- Define file structure, data models, types
- Write pseudocode for core algorithms
- Identify edge cases and risks
- Output a detailed implementation plan

### Phase 3: Implementation
Use the **implementer** agent to:
- Implement the solution following the architect's plan
- Ensure all type annotations are complete (basedmypy strict)
- Run quality checks: `uv run ruff check . && uv run ruff format --check . && uv run mypy`

### Phase 4: Testing
Use the **tester** agent to:
- Write comprehensive tests including edge cases
- Run tests: `uv run pytest -v`
- Verify against sample inputs from challenge docs

### Phase 5: Review
Use the **reviewer** agent to:
- Review for correctness, edge cases, performance
- Check scoring formula alignment
- Verify output format matches exactly
- Report any issues found

### Phase 6: Optimize (if needed)
Use the **optimizer** agent if:
- Performance doesn't meet time limits
- There's a clear path to better complexity

Report final status and any remaining concerns.
