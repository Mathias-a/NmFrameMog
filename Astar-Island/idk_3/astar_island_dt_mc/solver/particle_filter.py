# astar_island_dt_mc/solver/particle_filter.py
import numpy as np
from .config import *
from .digital_twin import VectorizedDigitalTwin

class ParticleFilter:
    def __init__(self, initial_map_grid, initial_settlements):
        self.N = NUM_PARTICLES
        self.weights = np.ones(self.N) / self.N
        self.twin = VectorizedDigitalTwin(self.N, initial_map_grid, initial_settlements)
        
    def step_forward(self):
        """Advances all particles by 1 year natively in the twin."""
        self.twin.advance_one_year()
        
    def assimilate_observation(self, viewport):
        """Updates particle weights based on the true viewport returned by the API."""
        vx, vy = viewport['viewport']['x'], viewport['viewport']['y']
        vw, vh = viewport['viewport']['w'], viewport['viewport']['h']
        
        ground_truth_grid = np.array(viewport['grid'])
        
        # 1. Evaluate Terrain match likelihood
        # particle_viewport shape: (self.N, vh, vw)
        particle_viewport = self.twin.terrain[:, vy:vy+vh, vx:vx+vw]
        
        # Mismatches (N, vh, vw) -> sum across spatial dims -> (N,)
        mismatches = np.sum(particle_viewport != ground_truth_grid, axis=(1, 2))
        
        # Exponential decay of weight based on terrain disagreements
        likelihood_terrain = np.exp(mismatches * PENALTY_TERRAIN_MISMATCH)
        
        # 2. Evaluate Continuous Settlement matching
        likelihood_stats = np.ones(self.N)
        for s in viewport['settlements']:
            sy, sx = s['y'], s['x']
            
            # L2 Gaussian distance over hidden stats
            pop_dist = ((self.twin.population[:, sy, sx] - s['population']) ** 2) / (2 * SIGMA_POPULATION**2)
            food_dist = ((self.twin.food[:, sy, sx] - s['food']) ** 2) / (2 * SIGMA_FOOD**2)
            wealth_dist = ((self.twin.wealth[:, sy, sx] - s['wealth']) ** 2) / (2 * SIGMA_WEALTH**2)
            defense_dist = ((self.twin.defense[:, sy, sx] - s['defense']) ** 2) / (2 * SIGMA_DEFENSE**2)
            
            likelihood_stats *= np.exp(-(pop_dist + food_dist + wealth_dist + defense_dist))
            
        # Update weights
        self.weights *= (likelihood_terrain * likelihood_stats)
        
        # Normalize
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # Complete collapse (edge case fallback: uniform reset)
            self.weights = np.ones(self.N) / self.N
            
        self._resample_if_needed()

    def _resample_if_needed(self):
        """Sequential Importance Resampling using Systematic method."""
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < ESS_THRESHOLD:
            # Systematic resampling
            cumulative_sum = np.cumsum(self.weights)
            cumulative_sum[-1] = 1.0  # Precision safety
            
            indices = np.zeros(self.N, dtype=np.int32)
            step = 1.0 / self.N
            r = np.random.uniform(0, step)
            
            i, j = 0, 0
            while i < self.N:
                if r < cumulative_sum[j]:
                    indices[i] = j
                    i += 1
                    r += step
                else:
                    j += 1
                    
            # Clone states
            self.twin.terrain = self.twin.terrain[indices]
            self.twin.population = self.twin.population[indices]
            self.twin.food = self.twin.food[indices]
            self.twin.wealth = self.twin.wealth[indices]
            self.twin.defense = self.twin.defense[indices]
            
            # Reset weights uniformly
            self.weights = np.ones(self.N) / self.N
            
            # Inject jitter to prevent exact continuous overlap (roughening)
            self.twin.population += np.random.normal(0, RESAMPLE_JITTER_STD, size=self.twin.population.shape)
            self.twin.population = np.clip(self.twin.population, 0, None)
            # Repeat for other continuous fields as needed

    def generate_final_tensor(self):
        """Collapses particle distribution into HxWx6 probability tensor."""
        tensor = np.zeros((MAP_HEIGHT, MAP_WIDTH, 6), dtype=np.float32)
        
        for k in range(6):
            # Mask of particles whose terrain holds class k
            mask = (self.twin.terrain == k)
            
            # Dot product weights over the particle axis
            # shape (N, H, W). We want to sum weights where mask is true
            # weights[:, None, None] broadcasts (N,) to (N, 1, 1)
            weighted_mask = mask * self.weights[:, None, None]
            tensor[:, :, k] = np.sum(weighted_mask, axis=0)
            
        # Mathematical safeguard to prevent infinity KL constraints
        tensor = np.maximum(tensor, 0.01)
        tensor = tensor / np.sum(tensor, axis=-1, keepdims=True)
        return tensor
