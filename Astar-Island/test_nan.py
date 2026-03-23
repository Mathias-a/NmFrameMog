import numpy as np
from astar_twin.scoring.safe_prediction import safe_prediction

tensor = np.full((2, 2, 6), np.nan)
res = safe_prediction(tensor)
print("NaN input:")
print(res[0, 0])

