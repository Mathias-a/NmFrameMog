# Astar Island Eval System - Research Learnings

## Immutable Dataset Manifests

### Key Patterns from Croissant 1.1 (MLCommons, Jan 2026)
- **Checksum requirement**: Use `sha256` for every file - mandatory for reproducibility
- **Semantic versioning**: MAJOR.MINOR.PATCH following semver 2.0.0
  - PATCH: same data, possible re-serialization
  - MINOR: same data + additions, old records retrievable
  - MAJOR: existing data changed/removed/shuffled
- **Live datasets**: Use `isLiveDataset: true` flag, skip checksums for mutable files
- **Distribution types**: FileObject (individual) vs FileSet (glob patterns)

### KitOps Manifest Format
- manifestVersion field (string, semver)
- package: name, version, description, authors
- datasets array: name, path, description, license
- model section with framework, version, parameters

### Essential Manifest Fields for Reproducibility
1. `manifest_version` - schema version identifier
2. `dataset_name` - human-readable name
3. `dataset_version` - semver
4. `files[]` - list with:
   - `path` or `glob` pattern
   - `sha256` checksum (REQUIRED)
   - `content_size` bytes
   - `encoding_format` (MIME type)
5. `created_at` - ISO timestamp
6. `is_frozen` - boolean lock flag

## Benchmark Report Schema

### Key Standards from lm-evaluation-harness (EleutherAI)
- Task-level results with version, filter, n-shot, metric, value, stderr
- Binary metrics for structured outputs (json_validity, schema_compliance)
- Aggregation types: mean, etc.

### JSONSchemaBench Report Structure
```json
{
  "task": "task_name",
  "version": "0.1",
  "metrics": {
    "metric_name": {
      "value": 0.95,
      "stderr": 0.02,
      "aggregation": "mean"
    }
  },
  "config": {
    "num_samples": 1000,
    "seed": 42
  }
}
```

### AgentHub/AgentClash EvaluationSpec Pattern
- Versioned EvaluationSpec schema
- Challenge pack format with fixtures/workspace inputs
- Validation declarations
- Immutability/versioning rules documented

### Essential Benchmark Report Fields
1. `report_version` - schema version
2. `benchmark_name` - identifier
3. `benchmark_version` - semver
4. `run_timestamp` - ISO datetime
5. `environment` - Python version, platform, seed
6. `results[]` - per-task results:
   - `task_name`
   - `task_version`
   - `metrics{}` with value, stderr, n
   - `config{}` (n_shot, etc.)
7. `artifact_refs[]` - links to frozen datasets/code
8. `git_commit` - code version if applicable

## Evaluation Artifact Versioning

### DVC Pipeline Pattern (production standard)
- dvc.yaml stages with cmd, deps, params, outs, metrics
- dvc.lock tracks md5/size of all dependencies
- Metrics JSON files compared across experiments

### AgentHub Artifact Versioning Pattern
- Version-bound evidence records
- Linked to auditable artifacts
- Append-only lifecycle event log
- Publish-time validation

### Key Versioning Patterns
1. **Semantic versions** for datasets, benchmarks, reports
2. **Content-addressed** storage via SHA256 hashes
3. **Git commit refs** for code versions
4. **Timestamps** for run provenance
5. **Parent refs** for lineage tracking

## Cross-Cutting Patterns

### For Reproducibility - Essential Fields
1. **Checksum (SHA256)** - every artifact file
2. **Version (semver)** - manifest, dataset, benchmark, report
3. **Timestamp** - creation/evaluation time
4. **Environment fingerprint** - Python version, platform, random seed
5. **Parent references** - lineage chain
6. **Validation schema** - JSON Schema for outputs

### For Immutability
1. **Read-only manifests** - once published, never modified
2. **Version bumps only** - PATCH for fixes, MINOR for additions
3. **Content-addressed** storage - hash-based file references
4. **Append-only logs** - lifecycle events

### Practical Implementation for Python CLI
- Pydantic models for all schemas (type validation)
- JSON Schema export for external validation
- SHA256 hashing via hashlib
- Semver parsing via packaging library
- Timestamps in ISO 8601 UTC
