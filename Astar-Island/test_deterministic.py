import numpy as np
from astar_twin.scoring.kl import compute_score
from astar_twin.scoring.safe_prediction import safe_prediction

# GT is 100% deterministic
gt = np.zeros((10, 10, 6))
gt[:, :, 0] = 1.0

# Pred is completely wrong
pred = np.zeros((10, 10, 6))
pred[:, :, 1] = 1.0
pred = safe_prediction(pred)

score = compute_score(gt, pred)
print(f"Score for completely wrong pred on deterministic GT: {score}")
