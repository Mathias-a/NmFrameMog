import time
import numpy as np
from pathlib import Path
from astar_twin.data.loaders import list_fixtures, load_fixture
from astar_twin.solver.adapters.benchmark import BenchmarkAdapter
from astar_twin.solver.pipeline import solve
from astar_twin.solver.eval.run_benchmark_suite import load_or_compute_ground_truths
from astar_twin.scoring import compute_score

data_dir = Path("data")
fixtures = [f for f in list_fixtures(data_dir) if not f.id.startswith("test-")]
fixtures.sort(key=lambda f: f.round_number)

# Run just 1 representative round
selected = [f for f in fixtures if f.round_number == 14]

print("Running fast evaluation on 1 round...")
for fixture_info in selected:
    fixture = load_fixture(Path(f"data/rounds/{fixture_info.id}/round_detail.json"))
    gts = load_or_compute_ground_truths(fixture, n_mc_runs=200, base_seed=0)

    adapter = BenchmarkAdapter(fixture, n_mc_runs=5, sim_seed_offset=0)
    t0 = time.monotonic()

    # 8 particles, 2 inner runs, 16 sims per seed
    result = solve(
        adapter,
        fixture.id,
        n_particles=8,
        n_inner_runs=2,
        sims_per_seed=16,
        base_seed=42,
    )

    per_seed_scores = [float(compute_score(gt, t)) for gt, t in zip(gts, result.tensors)]
    mean_score = float(np.mean(per_seed_scores))
    elapsed = time.monotonic() - t0

    print(f"Round {fixture.round_number:>2}: {mean_score:.2f} (Time: {elapsed:.1f}s)")
    if result.observation_ledger:
        print(f"  Ledger size: {len(result.observation_ledger.observations)}")
    print(f"  Queries used: {result.total_queries_used}")
