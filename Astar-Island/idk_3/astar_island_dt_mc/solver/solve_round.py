# astar_island_dt_mc/solver/solve_round.py
import json
import logging
import argparse
from pathlib import Path
from idk_2.astar_island_dr_plan_1.solver.api import AstarIslandClient
from .config import *
from .particle_filter import ParticleFilter
from .query_acquisition import select_best_viewport
import os

logger = logging.getLogger(__name__)

def solve(round_detail_file: str, cache_dir: str):
    """
    Main entrypoint invoked by the 'astar-solve-round' skill in the Solver Lane.
    """
    with open(round_detail_file, 'r') as f:
        round_data = json.load(f)
        
    client = AstarIslandClient(base_url=os.getenv("AINM_BASE_URL", "http://127.0.0.1:8000"), token="dummy")
    round_id = round_data['id']
    seeds_count = round_data['seeds_count']
    
    logger.info(f"Starting Digital Twin run for round {round_id} with {seeds_count} seeds.")
    
    predictions_dir = Path(cache_dir) / 'predictions' / round_id
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each seed sequentially
    for seed_idx in range(seeds_count):
        initial_state = round_data['initial_states'][seed_idx]
        grid = initial_state['grid']
        settlements = initial_state['settlements']
        
        # Initialize Digital Twin MC framework
        logger.info(f"Initializing 10,000 Particle Filters for seed {seed_idx}...")
        pf = ParticleFilter(initial_map_grid=grid, initial_settlements=settlements)
        
        # 50 simulated years, interleaving 50 API queries
        for year in range(YEARS_TO_SIMULATE):
            # 1. Forward simulation step
            pf.step_forward()
            
            # 2. Spend 1 viewport API query per year
            if year < QUERIES_BUDGET:
                qx, qy, qw, qh = select_best_viewport(pf)
                from idk_2.astar_island_dr_plan_1.solver.api import query_response_to_payload
                from idk_2.astar_island_dr_plan_1.solver.models import Viewport

                # Live API assimilation call
                try:
                    viewport_obj = client.simulate(
                        round_id=round_id, 
                        seed_index=seed_idx,
                        viewport=Viewport(x=qx, y=qy, width=qw, height=qh)
                    )
                    viewport_response = query_response_to_payload(viewport_obj)
                    pf.assimilate_observation(viewport_response)
                except Exception as e:
                    logger.warning(f"Failed to fetch viewport query at year {year}: {e}")
                    
            logger.info(f"Seed {seed_idx}: Year {year}/50 simulated. ESS: {1.0 / np.sum(pf.weights**2):.1f}")
            
        # Extract the finalized probability tensor
        final_tensor = pf.generate_final_tensor().tolist()
        
        # We save it but also formally submit it via the required pipeline tool if possible
        prediction_file = predictions_dir / f"seed-{seed_idx:02d}.json"
        
        payload = {
            "round_id": round_id,
            "seed_index": seed_idx,
            "prediction": final_tensor
        }
        
        with open(prediction_file, 'w') as f:
            json.dump(payload, f)
            
        logger.info(f"Seed {seed_idx} complete. Wrote {prediction_file}")

    # Write solver run summary
    runs_dir = Path(cache_dir) / 'runs' / round_id
    runs_dir.mkdir(parents=True, exist_ok=True)
    with open(runs_dir / 'summary.json', 'w') as f:
        json.dump({
            "run_id": round_id,
            "solver": "digital_twin_v1",
            "status": "completed"
        }, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-detail-file", required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    solve(args.round_detail_file, args.cache_dir)
