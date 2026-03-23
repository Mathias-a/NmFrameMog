import numpy as np
from astar_twin.solver.predict.mixture import apply_soft_mixture

def test_apply_soft_mixture():
    height, width = 2, 2
    tensor1 = np.full((height, width, 6), 1.0 / 6.0)
    tensor2 = np.zeros((height, width, 6))
    tensor2[:, :, 0] = 1.0
    
    weights = [0.2, 0.8]
    
    blended = apply_soft_mixture([tensor1, tensor2], weights, height, width)
    
    # Check shape
    assert blended.shape == (height, width, 6)
    
    # Check normalization
    sums = np.sum(blended, axis=2)
    np.testing.assert_allclose(sums, 1.0)
    
    # Check blending logic
    assert blended[0, 0, 0] > 0.8
    assert blended[0, 0, 1] > 0.03
