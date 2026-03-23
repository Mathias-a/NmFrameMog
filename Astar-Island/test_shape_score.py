import numpy as np
from astar_twin.scoring.kl import compute_score

gt = np.zeros((10, 10, 6))
gt[:, :, 0] = 1.0
pred = np.zeros((6, 10, 10))

try:
    score = compute_score(gt, pred)
    print(f"Score with wrong shape: {score}")
except Exception as e:
    print("Error:", e)

