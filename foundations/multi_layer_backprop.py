import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]], b1: List[float],
        W2: List[List[float]], b2: List[float],
        y_true: List[float]
    ) -> dict:

        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        x = np.array(x)
        W1 = np.array(W1)
        W2 = np.array(W2)
        b1 = np.array(b1)
        b2 = np.array(b2)
        y = np.array(y_true)
        
        # Forward
        # Layer 1
        z1 = x @ W1.T + b1
        h = np.maximum(0, z1) # relu

        # Layer 2
        y_hat = h @ W2.T + b2
        error = y_hat - y # z2 = prediction
        loss = np.mean(np.square(error))

        
        dL_dy_hat = 2 * (y_hat - y) / y.size 
        dL_dW2 = np.outer(dL_dy_hat, h)
        dL_db2 = dL_dy_hat * 1

        dL_dh = W2.T @ dL_dy_hat
        dL_dz1 = dL_dh * (z1 > 0)
        dL_dW1 = np.outer(dL_dz1, x)
        dL_db1 = dL_dz1 * 1

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dL_dW1, 4).tolist(),
            'db1': np.round(dL_db1, 4).tolist(),
            'dW2': np.round(dL_dW2, 4).tolist(),
            'db2': np.round(dL_db2, 4).tolist(),
        }
