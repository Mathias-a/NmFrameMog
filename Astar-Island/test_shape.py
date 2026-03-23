import numpy as np
from astar_twin.scoring.safe_prediction import safe_prediction

tensor = np.zeros((6, 10, 10))
try:
    res = safe_prediction(tensor)
    print("Shape:", res.shape)
except Exception as e:
    print("Error:", e)

