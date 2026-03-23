import numpy as np
from astar_twin.harness.runner import BenchmarkRunner
from astar_twin.strategies import REGISTRY
from astar_twin.data.loaders import load_fixture
from pathlib import Path
from astar_twin.data.models import InitialState
from astar_twin.contracts.types import TERRAIN_TO_CLASS

class InitialStateStrategy:
    @property
    def name(self): return "initial_state"
    def predict(self, initial_state, budget, base_seed):
        H = len(initial_state.grid)
        W = len(initial_state.grid[0])
        pred = np.zeros((H, W, 6))
        for y in range(H):
            for x in range(W):
                terrain = initial_state.grid[y][x]
                cls_idx = TERRAIN_TO_CLASS[terrain]
                pred[y, x, cls_idx] = 1.0
        return pred

fixture = load_fixture(Path('data/rounds/b0f9d1bf-4b71-4e6e-816c-19c718d29056'))
strategies = [InitialStateStrategy()]
report = BenchmarkRunner(fixture=fixture, base_seed=42).run(strategies)
print(f"Initial State Strategy: {report.strategy_reports[0].mean_score:.2f}")

