import argparse
import sys
import warnings
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
from typing import Any

# Suppress runtime warnings from exploratory parameter combinations
warnings.filterwarnings("ignore", category=RuntimeWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR / "src"))

from astar_twin.data.loaders import list_fixtures, load_fixture
from astar_twin.fixture_prep.writer import write_fixture
from astar_twin.data.models import ParamsSource
from astar_twin.engine import Simulator
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.scoring.kl import compute_score
from astar_twin.scoring.safe_prediction import safe_prediction
from astar_twin.solver.inference.particles import INFERRED_PARAMS, _CONTINUOUS_RANGES, _INTEGER_PARAMS
from dataclasses import replace

def objective(x, param_keys, fixture, base_seed, n_runs=10):
    params = {}
    for i, key in enumerate(param_keys):
        val = x[i]
        if key in _INTEGER_PARAMS or isinstance(getattr(fixture.simulation_params, key), int):
            val = int(round(val))
        params[key] = val
        
    sim_params = replace(fixture.simulation_params, **params)
    
    total_score = 0.0
    for seed_idx in range(fixture.seeds_count):
        initial_state = fixture.initial_states[seed_idx]
        gt = np.array(fixture.ground_truths[seed_idx])
        
        sim = Simulator(params=sim_params)
        runner = MCRunner(sim)
        runs = runner.run_batch(initial_state, n_runs, base_seed=base_seed + seed_idx * 1000)
        
        pred = aggregate_runs(runs, fixture.map_height, fixture.map_width)
        safe_pred = safe_prediction(pred)
        
        score = compute_score(gt, safe_pred)
        total_score += score
        
    return -(total_score / fixture.seeds_count)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(BENCHMARK_DIR / "data" / "rounds"))
    parser.add_argument("--fixture", type=str, help="Specific fixture directory name")
    parser.add_argument("--fast", action="store_true", help="Run a quick 3-iteration optimization")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    parent_dir = data_dir.parent if data_dir.name == "rounds" else data_dir
    
    if args.fixture:
        fixtures = [load_fixture(parent_dir / "rounds" / args.fixture)]
    else:
        fixtures = [f for f in list_fixtures(parent_dir) if not f.id.startswith("test-") and f.ground_truths is not None]
    
    optim_keys = [k for k in INFERRED_PARAMS if k in _CONTINUOUS_RANGES or k in _INTEGER_PARAMS]
    
    bounds = []
    for k in optim_keys:
        if k in _CONTINUOUS_RANGES:
            bounds.append(_CONTINUOUS_RANGES[k][:2])
        elif k in _INTEGER_PARAMS:
            bounds.append((0, 20))
            
    bounds_dict = {
        "expansion_radius": (1, 10),
        "raid_range_base": (1, 15),
        "raid_range_longship_bonus": (0, 10),
        "trade_range": (1, 20),
    }
    for i, k in enumerate(optim_keys):
        if k in bounds_dict:
            bounds[i] = bounds_dict[k]
    
    print(f"Optimizing {len(optim_keys)} parameters for {len(fixtures)} fixtures...")
    
    maxiter = 3 if args.fast else 10
    popsize = 2 if args.fast else 5
    
    for fixture in fixtures:
        print(f"Processing round {fixture.round_number} ({fixture.id})...")
        
        result = differential_evolution(
            objective, 
            bounds=bounds, 
            args=(optim_keys, fixture, 42, 10),
            maxiter=maxiter, 
            popsize=popsize,
            disp=True,
            workers=-1,
            updating='deferred'
        )
        
        print(f"  Best score during optim: {-result.fun:.2f}")
        
        best_params = {}
        for i, key in enumerate(optim_keys):
            val = result.x[i]
            if key in _INTEGER_PARAMS or isinstance(getattr(fixture.simulation_params, key), int):
                val = int(round(val))
            best_params[key] = val
            
        new_sim_params = replace(fixture.simulation_params, **best_params)
        
        eval_runs = 50 if args.fast else 100
        final_score = -objective(result.x, optim_keys, fixture, 123, n_runs=eval_runs)
        print(f"  Final evaluated score ({eval_runs} runs): {final_score:.2f}")
        
        updated_fixture = fixture.model_copy(update={
            "simulation_params": new_sim_params,
            "params_source": ParamsSource.INFERRED
        })
        
        fixture_path = parent_dir / "rounds" / fixture.id / "round_detail.json"
        write_fixture(updated_fixture, fixture_path)
        print(f"  Saved to {fixture_path}\n")

if __name__ == "__main__":
    main()
