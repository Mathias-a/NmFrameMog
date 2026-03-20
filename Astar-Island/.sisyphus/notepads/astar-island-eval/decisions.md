# Astar Island Eval System - Architectural Decisions

## Decision 1: Manifest Format
**Chosen**: Custom YAML/JSON manifest with SHA256 checksums
**Rationale**: 
- Croissant is too complex (JSON-LD overhead)
- KitOps is good but focused on model packaging
- Need: lightweight, Python-native, frozen dataset focus

## Decision 2: Versioning Scheme  
**Chosen**: Semantic Versioning 2.0.0 (MAJOR.MINOR.PATCH)
**Rationale**:
- Industry standard, well understood
- Matches Croissant recommendation
- Clear upgrade rules for users

## Decision 3: Report Schema
**Chosen**: lm-evaluation-harness-inspired JSON structure
**Rationale**:
- Proven in production use
- Task-level granularity
- Supports aggregation and stderr

## Decision 4: Immutability Strategy
**Chosen**: Content-addressed storage + version bumps
**Rationale**:
- SHA256 ensures bit-perfect reproducibility
- Version numbers prevent silent overwrites
- Append-only logs for audit trail
