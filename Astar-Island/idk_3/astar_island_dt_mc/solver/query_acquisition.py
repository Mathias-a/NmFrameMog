# astar_island_dt_mc/solver/query_acquisition.py
import numpy as np

def select_best_viewport(particle_filter, w=15, h=15):
    """
    Active Learning Query Function: 
    Finds the 15x15 window that exhibits maximum variance across the particle distribution's terrain states. 
    By targeting uncertainty, we prune the worst particles earliest and maximize information gain.
    """
    terrain = particle_filter.twin.terrain  # Shape: (N, 40, 40)
    
    # Calculate cell-wise variance of the terrain class across all particles
    # Using simple variance over integer classes serves as a rapid proxy for entropy
    cell_variance = np.var(terrain, axis=0)  # Shape: (40, 40)
    
    max_var = -1.0
    best_x, best_y = 0, 0
    
    height, width = cell_variance.shape
    
    # Sliding window to find max total variance area
    for y in range(height - h + 1):
        for x in range(width - w + 1):
            window_var_sum = np.sum(cell_variance[y:y+h, x:x+w])
            if window_var_sum > max_var:
                max_var = window_var_sum
                best_x = x
                best_y = y
                
    return best_x, best_y, w, h
