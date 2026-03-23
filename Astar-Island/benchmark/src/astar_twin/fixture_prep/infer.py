from __future__ import annotations

import numpy as np
from dataclasses import replace
from astar_twin.data.models import RoundFixture, ParamsSource
from astar_twin.engine import Simulator
from astar_twin.mc import MCRunner, aggregate_runs
from astar_twin.scoring.kl import compute_score
from astar_twin.scoring.safe_prediction import safe_prediction
from astar_twin.solver.inference.particles import (
    INFERRED_PARAMS,
    _CONTINUOUS_RANGES,
    _INTEGER_PARAMS,
    create_posterior,
    Particle
)
from numpy.random import default_rng

def evaluate_particle(particle: Particle, fixture: RoundFixture, base_seed: int, n_runs: int = 20) -> float:
    score = 0.0
    for seed_idx in range(fixture.seeds_count):
        gt = np.array(fixture.ground_truths[seed_idx])
        initial_state = fixture.initial_states[seed_idx]
        
        sim_params = replace(fixture.simulation_params, **particle.params)
        sim = Simulator(params=sim_params)
        mc_runner = MCRunner(sim)
        runs = mc_runner.run_batch(initial_state, n_runs, base_seed=base_seed + seed_idx * 1000)
        
        pred = aggregate_runs(runs, fixture.map_height, fixture.map_width)
        safe_pred = safe_prediction(pred)
        
        score += compute_score(gt, safe_pred)
    
    return score / fixture.seeds_count

def infer_params_for_fixture(fixture: RoundFixture, pop_size: int = 50, generations: int = 3) -> RoundFixture:
    rng = default_rng(42)
    # Generate initial population
    particles = create_posterior(pop_size, rng)
    
    best_particle = None
    best_score = -1.0
    
    for gen in range(generations):
        print(f"Generation {gen+1}/{generations}")
        scores = []
        for p in particles:
            score = evaluate_particle(p, fixture, base_seed=123, n_runs=15)
            scores.append((score, p))
            if score > best_score:
                best_score = score
                best_particle = p
                print(f"  New best score: {best_score:.2f}")
                
        # Select top 20%
        scores.sort(key=lambda x: x[0], reverse=True)
        top_particles = [p for s, p in scores[:pop_size // 5]]
        
        # Create new population by mutating top particles
        new_particles = []
        new_particles.append(Particle(params=best_particle.params.copy(), log_weight=0.0))
        
        while len(new_particles) < pop_size:
            parent = rng.choice(top_particles)
            child_params = parent.params.copy()
            
            # Mutate 1-3 parameters
            num_mutations = rng.integers(1, 4)
            keys = rng.choice(INFERRED_PARAMS, num_mutations, replace=False)
            
            for key in keys:
                if key in _CONTINUOUS_RANGES:
                    lo, hi, default = _CONTINUOUS_RANGES[key]
                    if key in _INTEGER_PARAMS:
                        sigma = _INTEGER_PARAMS[key]
                        val = int(np.clip(rng.normal(child_params[key], sigma), lo, hi))
                    else:
                        sigma = (hi - lo) * 0.1  # 10% mutation
                        val = np.clip(rng.normal(child_params[key], sigma), lo, hi)
                    child_params[key] = val
                    
            new_particles.append(Particle(params=child_params, log_weight=0.0))
            
        particles = new_particles

    # Final evaluation of the best particle with more runs
    final_score = evaluate_particle(best_particle, fixture, base_seed=123, n_runs=50)
    print(f"Final best score: {final_score:.2f}")
    
    inferred_sim_params = replace(fixture.simulation_params, **best_particle.params)
    updated_fixture = fixture.model_copy(update={
        "simulation_params": inferred_sim_params,
        "params_source": ParamsSource.INFERRED
    })
    
    return updated_fixture
