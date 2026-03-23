import numpy as np
from astar_twin.scoring.kl import compute_score
from astar_twin.scoring.safe_prediction import safe_prediction

# Create a dummy ground truth
# Let's say 50% of cells are deterministic (entropy=0, ignored)
# 50% of cells are uniform over 3 classes (e.g. Empty, Forest, Settlement)
gt = np.zeros((10, 10, 6))
gt[:5, :, 0] = 1.0 # Deterministic Empty
gt[5:, :, :3] = 1.0 / 3.0 # Uniform over 3 classes

# Create a uniform prediction
pred = np.ones((10, 10, 6)) / 6.0
pred = safe_prediction(pred)

score = compute_score(gt, pred)
print(f"Score for uniform prediction vs 3-class GT: {score:.2f}")

# What if GT is uniform over 2 classes?
gt2 = np.zeros((10, 10, 6))
gt2[:, :, :2] = 0.5
score2 = compute_score(gt2, pred)
print(f"Score for uniform prediction vs 2-class GT: {score2:.2f}")

# What if GT is uniform over 4 classes?
gt4 = np.zeros((10, 10, 6))
gt4[:, :, :4] = 0.25
score4 = compute_score(gt4, pred)
print(f"Score for uniform prediction vs 4-class GT: {score4:.2f}")

