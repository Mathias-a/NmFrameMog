# Benchmarking Astar Island Models

Benchmark any prediction model against local Monte Carlo ground truth using the built-in benchmarking system.

## Quick Start (CLI)

Run the built-in baseline model against all available fixture rounds:

```bash
uv run python -m round_8_implementation benchmark
```

This uses the default configuration (128 rollouts) and the two bundled fixture rounds.

### Fast iteration

Use the `--preset quick` flag for faster feedback (16 rollouts, less accurate ground truth):

```bash
uv run python -m round_8_implementation benchmark --preset quick
```

### Full precision

Use `--preset full` for the most accurate scoring (256 rollouts, slower):

```bash
uv run python -m round_8_implementation benchmark --preset full
```

## Benchmarking a Custom Model

A model is any callable with the signature `(list[list[int]]) -> list[list[list[float]]]`. It receives the initial terrain grid and returns a W×H×6 probability tensor.

Pass your model using `--model NAME=module.path:callable`:

```bash
uv run python -m round_8_implementation benchmark \
  --model mymodel=my_package.predictor:predict
```

This imports `predict` from `my_package.predictor` and benchmarks it alongside the baseline.

### Multiple models

Repeat `--model` to compare several models side-by-side:

```bash
uv run python -m round_8_implementation benchmark \
  --model v1=my_models.v1:predict \
  --model v2=my_models.v2:predict
```

### Exclude the baseline

If you only want your own models without the built-in baseline:

```bash
uv run python -m round_8_implementation benchmark \
  --no-baseline \
  --model mymodel=my_package.predictor:predict
```

## CLI Options

| Flag | Description | Default |
|---|---|---|
| `--model NAME=mod:fn` | Add a model (repeatable) | none |
| `--no-baseline` | Exclude built-in baseline | included |
| `--preset quick\|full` | Configuration preset | 128 rollouts |
| `--rollouts N` | Override rollout count | from preset |
| `--seed-index N` | Benchmark only one seed | all seeds |
| `--fixture PATH` | Custom fixture JSON (repeatable) | bundled fixtures |
| `--output PATH` | Write JSON report to file | stdout only |

## Saving Results

Export a JSON report for programmatic analysis:

```bash
uv run python -m round_8_implementation benchmark \
  --output results/benchmark-report.json
```

## Python API

For scripting or notebooks:

```python
from round_8_implementation.solver.benchmark import (
    BenchmarkConfig,
    BenchmarkRunner,
    ModelSpec,
)
from round_8_implementation.solver.baseline import build_baseline_tensor

# Load fixture data (defaults to the two bundled rounds)
runner = BenchmarkRunner.from_fixture_paths()

# Define models to compare
models = [
    ModelSpec(name="baseline", predict=build_baseline_tensor),
    ModelSpec(name="my_model", predict=my_predict_function),
]

# Run benchmark
config = BenchmarkConfig.quick()  # or .full(), or BenchmarkConfig(rollout_count=64)
report = runner.compare(models, config)

# Display results
print(report.format_table())

# Export as dict
data = report.to_dict()
```

## How Scoring Works

1. **Ground truth generation**: For each (round, seed) pair, the proxy simulator runs N Monte Carlo rollouts and aggregates terrain distributions into a probability tensor (with 0.0 floor — no artificial smoothing on ground truth).
2. **Model prediction**: Your model receives the initial grid and returns a W×H×6 probability tensor. Probabilities are validated (must sum to ~1.0 per cell, minimum 0.01 floor per class).
3. **Scoring**: Each cell is scored using entropy-weighted KL divergence. The final score is `max(0, min(100, 100 × exp(-3 × weighted_kl)))`.

Higher rollout counts produce more stable ground truth at the cost of speed. The `quick` preset (16 rollouts) is good for rapid iteration; `full` (256 rollouts) gives the most reliable comparison.

## Model Contract

Your predict function must:

- Accept `list[list[int]]` — the initial terrain grid (H rows × W columns)
- Return `list[list[list[float]]]` — probability tensor (H × W × 6 classes)
- Each cell's 6 probabilities must sum to approximately 1.0
- No probability may be below 0.01 (the competition's probability floor)

The 6 terrain classes (in order): `empty`, `settlement`, `port_settlement`, `ruin`, `port_ruin`, `water`.
