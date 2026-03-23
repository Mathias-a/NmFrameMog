import numpy as np
from astar_twin.harness.runner import BenchmarkRunner
from astar_twin.strategies import REGISTRY
from astar_twin.data.loaders import load_fixture
from pathlib import Path
from astar_twin.data.models import InitialState

class CustomStrategy:
    def __init__(self, pred_func):
        self.pred_func = pred_func
    @property
    def name(self): return "custom"
    def predict(self, initial_state, budget, base_seed):
        H = len(initial_state.grid)
        W = len(initial_state.grid[0])
        return self.pred_func(H, W)

fixture = load_fixture(Path('data/rounds/b0f9d1bf-4b71-4e6e-816c-19c718d29056'))

def run_test(name, pred_func):
    strategies = [CustomStrategy(pred_func)]
    report = BenchmarkRunner(fixture=fixture, base_seed=42).run(strategies)
    print(f"{name}: {report.strategy_reports[0].mean_score:.2f}")

# 1. Uniform
run_test("Uniform", lambda H, W: np.ones((H, W, 6)) / 6.0)

# 2. Only Empty (class 0)
def only_empty(H, W):
    p = np.zeros((H, W, 6))
    p[:, :, 0] = 1.0
    return p
run_test("Only Empty", only_empty)

# 3. Only Forest (class 4)
def only_forest(H, W):
    p = np.zeros((H, W, 6))
    p[:, :, 4] = 1.0
    return p
run_test("Only Forest", only_forest)

# 4. Empty + Forest (50/50)
def empty_forest(H, W):
    p = np.zeros((H, W, 6))
    p[:, :, 0] = 0.5
    p[:, :, 4] = 0.5
    return p
run_test("Empty + Forest", empty_forest)

# 5. Empty + Forest + Settlement (33/33/33)
def empty_forest_settlement(H, W):
    p = np.zeros((H, W, 6))
    p[:, :, 0] = 1/3
    p[:, :, 4] = 1/3
    p[:, :, 1] = 1/3
    return p
run_test("Empty + Forest + Settlement", empty_forest_settlement)


# 6. Empty + Forest + Settlement + Port (25/25/25/25)
def empty_forest_settlement_port(H, W):
    p = np.zeros((H, W, 6))
    p[:, :, 0] = 0.25
    p[:, :, 4] = 0.25
    p[:, :, 1] = 0.25
    p[:, :, 2] = 0.25
    return p
run_test("Empty + Forest + Settlement + Port", empty_forest_settlement_port)

# 7. Empty + Forest + Settlement + Port + Ruin (20/20/20/20/20)
def empty_forest_settlement_port_ruin(H, W):
    p = np.zeros((H, W, 6))
    p[:, :, 0] = 0.2
    p[:, :, 4] = 0.2
    p[:, :, 1] = 0.2
    p[:, :, 2] = 0.2
    p[:, :, 3] = 0.2
    return p
run_test("Empty + Forest + Settlement + Port + Ruin", empty_forest_settlement_port_ruin)

