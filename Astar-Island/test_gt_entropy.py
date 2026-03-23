import numpy as np
from astar_twin.data.loaders import load_fixture
from pathlib import Path

fixture = load_fixture(Path('data/rounds/b0f9d1bf-4b71-4e6e-816c-19c718d29056'))
gt = np.array(fixture.ground_truths[0])

eps = 1e-15
gt_safe = np.clip(gt, eps, None)
entropy = -np.sum(np.where(gt > 0, gt * np.log(gt_safe), 0.0), axis=2)
mask = entropy >= 1e-10

print(f"Total cells: {gt.shape[0] * gt.shape[1]}")
print(f"Cells with entropy >= 1e-10: {np.sum(mask)}")
print(f"Mean entropy of uncertain cells: {np.mean(entropy[mask]):.4f}")
print(f"Max entropy: {np.max(entropy):.4f}")
