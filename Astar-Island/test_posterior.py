import numpy as np
from astar_twin.solver.inference.posterior import PosteriorState, Particle, temper_if_collapsed

particles = [Particle(params={}, log_weight=-np.inf) for _ in range(5)]
state = PosteriorState(particles=particles)

print("ESS:", state.ess)
print("Top mass:", state.top_particle_mass)
print("Normalized:", state.normalized_weights())

state = temper_if_collapsed(state)
print("After temper:", [p.log_weight for p in state.particles])

particles2 = [Particle(params={}, log_weight=np.nan) for _ in range(5)]
state2 = PosteriorState(particles=particles2)
print("ESS2:", state2.ess)
print("Top mass2:", state2.top_particle_mass)
print("Normalized2:", state2.normalized_weights())

