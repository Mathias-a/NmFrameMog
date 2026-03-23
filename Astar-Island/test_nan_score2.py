import numpy as np
from astar_twin.scoring.kl import compute_score

gt = np.zeros((2, 2, 6))
gt[:, :, :2] = 0.5
pred = np.full((2, 2, 6), np.nan)

score = compute_score(gt, pred)
print(f"Score with NaN on uncertain GT: {score}")

