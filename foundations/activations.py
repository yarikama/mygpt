import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        z = np.clip(z, -500, 500)
        return np.round(1 / (1 + np.exp(-z)),5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        # return np.array([i if i > 0 else 0.0 for i in z])
        return np.maximum(0, z)
