import numpy as np
from astar_twin.scoring.safe_prediction import safe_prediction

# 1. All zeros
tensor = np.zeros((2, 2, 6))
res = safe_prediction(tensor)
print("All zeros:")
print(res[0, 0])

# 2. Unnormalized
tensor = np.ones((2, 2, 6)) * 10.0
res = safe_prediction(tensor)
print("\nUnnormalized:")
print(res[0, 0])

# 3. One class 1.0, others 0.0
tensor = np.zeros((2, 2, 6))
tensor[:, :, 0] = 1.0
res = safe_prediction(tensor)
print("\nOne class 1.0:")
print(res[0, 0])

# 4. Negative values
tensor = np.ones((2, 2, 6))
tensor[:, :, 0] = -1.0
res = safe_prediction(tensor)
print("\nNegative values:")
print(res[0, 0])

