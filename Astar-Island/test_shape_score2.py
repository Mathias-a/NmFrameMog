import numpy as np
from astar_twin.scoring.kl import compute_score

gt = np.zeros((10, 10, 6))
gt[:, :, :2] = 0.5
pred = np.zeros((6, 10, 10))

try:
    score = compute_score(gt, pred)
    print(f"Score with wrong shape on uncertain GT: {score}")
except Exception as e:
    print("Error:", e)

