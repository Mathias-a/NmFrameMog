import sys
import json
from pathlib import Path
import numpy as np
import time
sys.path.insert(0, str(Path('benchmark/src').resolve()))

from astar_twin.data.loaders import load_fixture
from astar_twin.solver.adapters.benchmark import BenchmarkAdapter
from astar_twin.solver.inference.posterior import create_posterior, update_posterior, resample_if_needed, temper_if_collapsed
from astar_twin.solver.predict.posterior_mc import predict_all_seeds
from astar_twin.scoring.kl import compute_score
from astar_twin.scoring.safe_prediction import safe_prediction

start_time = time.time()
with open("../benchmark/previous_run_data.json") as f:
    run_data = json.load(f)

rid = run_data["round_id"]
rnum = run_data["round_number"]
queries = run_data["queries"]
print(f"Benchmarking round {rnum} ({rid}) using {len(queries)} previous viewport queries.")

# 1. Set digital twin to initial state
fixture_path = Path(f"data/rounds/{rid}")
if not fixture_path.exists():
    print(f"Fixture {fixture_path} missing.")
    sys.exit(1)

fixture = load_fixture(fixture_path)
adapter = BenchmarkAdapter(fixture)

# 2. Use queries to estimate hidden parameters
# Fast parameters for a reliable run
n_particles = 12
n_inner_runs = 2
print(f"Estimating parameters: {n_particles} particles, {n_inner_runs} inner runs per query.")
posterior = create_posterior(n_particles=n_particles, seed=42)

for i, q in enumerate(queries):
    seed_idx = q["seed_index"]
    x = q["viewport_x"]
    y = q["viewport_y"]
    w = q["viewport_w"]
    h = q["viewport_h"]
    
    obs = adapter.simulate(rid, seed_idx, x, y, w, h)
    
    # Update estimation
    posterior = update_posterior(posterior, obs, fixture.initial_states[seed_idx], n_inner_runs=n_inner_runs, base_seed=42+i)
    posterior = resample_if_needed(posterior, seed=42+i)
    posterior = temper_if_collapsed(posterior)
    if (i+1) % 10 == 0:
        elapsed = time.time() - start_time
        print(f"  Processed query {i+1}/50... (Elapsed: {elapsed:.1f}s)")

if posterior.ess_history:
    print("Estimated final ESS:", float(posterior.ess_history[-1]))
else:
    print("Estimated ESS history empty.")

# 3. Generate a solution map for the digital twin
top_k = 4
sims_per_seed = 16
print(f"Generating solution map from estimated parameters: top_k={top_k}, sims_per_seed={sims_per_seed}")
tensors, metrics = predict_all_seeds(
    posterior=posterior,
    initial_states=fixture.initial_states,
    map_height=fixture.map_height,
    map_width=fixture.map_width,
    top_k=top_k,
    sims_per_seed=sims_per_seed,
    base_seed=123
)

# 4. Compare to final distribution from prod api
print("\nEvaluating against prod API ground truth...")
scores = []
api_analysis = run_data["analysis"]

for seed_idx, tensor in enumerate(tensors):
    safe_tensor = safe_prediction(tensor)
    
    gt_list = api_analysis[seed_idx]["ground_truth"]
    gt_tensor = np.array(gt_list, dtype=np.float64)
    
    score = compute_score(safe_tensor, gt_tensor)
    scores.append(score)
    print(f"Seed {seed_idx} Score: {score:.2f}")

mean_score = sum(scores) / len(scores)
total_time = time.time() - start_time
print(f"\nFinal Strategy Score: {mean_score:.2f} (Computed in {total_time:.1f}s)")

