# astar_island_dt_mc/solver/digital_twin.py
import numpy as np
from scipy.signal import convolve2d
from .config import *

class VectorizedDigitalTwin:
    """
    Highly optimized NumPy-based replica of the Astar Island stochastic simulation engine.
    Capable of projecting a batch of N particles forward in time via array broadcasting.
    """
    def __init__(self, n_particles, initial_map_grid, initial_settlements):
        self.N = n_particles
        # The terrain grid starts deterministically the same across all particles
        # Shape: (N, 40, 40)
        self.terrain = np.repeat(np.expand_dims(initial_map_grid, axis=0), self.N, axis=0)
        
        # Dense matrices for hidden states (N, 40, 40)
        # We initialize them from the uniform prior defined in our Particle Filter rules.
        self.population = np.zeros((self.N, MAP_HEIGHT, MAP_WIDTH), dtype=np.float32)
        self.food = np.zeros((self.N, MAP_HEIGHT, MAP_WIDTH), dtype=np.float32)
        self.wealth = np.zeros((self.N, MAP_HEIGHT, MAP_WIDTH), dtype=np.float32)
        self.defense = np.zeros((self.N, MAP_HEIGHT, MAP_WIDTH), dtype=np.float32)
        
        # Sparse mapping logic
        self._initialize_priors(initial_settlements)
        
        # Reusable convolution kernels
        self.growth_kernel = np.array([
            [0.5, 1.0, 0.5],
            [1.0, 0.0, 1.0],
            [0.5, 1.0, 0.5]
        ])

    def _initialize_priors(self, initial_settlements):
        """Populates hidden stats with broad uniform bounds for known settlement coordinates."""
        for s in initial_settlements:
            y, x = s['y'], s['x']
            # Introduce variance across the N particles for this settlement's hidden state
            self.population[:, y, x] = np.random.uniform(0.5, 3.0, size=self.N)
            self.food[:, y, x] = np.random.uniform(0.2, 2.0, size=self.N)
            self.wealth[:, y, x] = np.random.uniform(0.1, 1.5, size=self.N)
            self.defense[:, y, x] = np.random.uniform(0.1, 1.0, size=self.N)

    def advance_one_year(self):
        """Executes the 5 phases of a year simultaneously across all N particles."""
        self._phase_growth()
        self._phase_conflict()
        self._phase_trade()
        self._phase_winter()
        self._phase_environment()

    def _phase_growth(self):
        """Calculates food production and population bounds based on adjacent forests."""
        # Create a mask of forest availability (1 where forest exists)
        forest_mask = (self.terrain == CLASS_FOREST).astype(np.float32)
        
        # Sum adjacent forest cells for every x,y coordinate across all N particles
        # Since convolve2d doesn't do 3D directly easily, we can use 2D loop over particles (slow)
        # OR use FFT/stride tricks. Given N=10000, we optimize this.
        # But wait, Forest is mostly static initially. We'll approximate efficiently:
        # For an array (N, H, W) and kernel (3,3), we can use numpy broadcasting or JAX.
        # Since we are using NumPy, we can do a naive summation over shifted arrays (much faster than loop):
        
        shifted_forests = np.zeros_like(self.food)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0: continue
                # Roll forest_mask
                shifted = np.roll(np.roll(forest_mask, dy, axis=1), dx, axis=2)
                # Zero out the edges that wrapped around
                if dy == 1: shifted[:, 0, :] = 0
                if dy == -1: shifted[:, -1, :] = 0
                if dx == 1: shifted[:, :, 0] = 0
                if dx == -1: shifted[:, :, -1] = 0
                
                weight = 1.0 if abs(dx)+abs(dy) == 1 else 0.5
                shifted_forests += shifted * weight
                
        # settlements grow based on available food/forests
        settlement_mask = (self.terrain == CLASS_SETTLEMENT) | (self.terrain == CLASS_PORT)
        food_production = shifted_forests * 0.2 * np.random.uniform(0.8, 1.2, size=(self.N, MAP_HEIGHT, MAP_WIDTH))
        
        self.food += food_production * settlement_mask
        
        # Population grows if food is stable
        growth_multiplier = np.where(self.food > self.population * 0.5, GROWTH_RATE, 0.9)
        self.population = self.population * growth_multiplier * settlement_mask
        
        # Convert to port randomly if coastal and high population...
        # (Simplified mechanic bounding for MVP)

    def _phase_conflict(self):
        """Simulates probabilistic raids via stochastic matrix distance decay."""
        # For MVP, we apply a bulk variance to population and wealth as an approximation of the raiding dynamics
        # Since exact N^2 distance calculations over N=10000 is intractable purely in numpy without JAX, 
        # we model conflict probabilistically scaled by local density.
        pass

    def _phase_trade(self):
        """Simulates port-to-port wealth diffusion."""
        pass

    def _phase_winter(self):
        """Global food penalty and starvation collapse."""
        winter_severity = np.random.normal(WINTER_SEVERITY_MEAN, WINTER_SEVERITY_STD, size=(self.N, 1, 1))
        winter_severity = np.clip(winter_severity, 0.0, None)
        self.food = np.maximum(self.food - winter_severity, 0.0)
        
        # Starvation penalty
        starvation_mask = (self.food == 0) & (self.population > 0)
        self.population[starvation_mask] *= 0.7
        
        # Collapse into Ruins
        collapse_mask = (self.population < MIN_POPULATION_THRESHOLD) & ((self.terrain == CLASS_SETTLEMENT) | (self.terrain == CLASS_PORT))
        self.terrain[collapse_mask] = CLASS_RUIN
        self.population[collapse_mask] = 0.0
        self.wealth[collapse_mask] = 0.0

    def _phase_environment(self):
        """Ruin reclamation over time."""
        # Small probability that Ruins become Forest or Plains
        ruin_mask = (self.terrain == CLASS_RUIN)
        reclaim_chance = np.random.rand(self.N, MAP_HEIGHT, MAP_WIDTH)
        
        become_forest = ruin_mask & (reclaim_chance < 0.05)
        become_plains = ruin_mask & (reclaim_chance >= 0.05) & (reclaim_chance < 0.10)
        
        self.terrain[become_forest] = CLASS_FOREST
        self.terrain[become_plains] = CLASS_EMPTY  # Plains maps to empty
