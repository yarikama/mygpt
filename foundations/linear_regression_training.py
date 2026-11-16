import numpy as np
from numpy.typing import NDArray


class Solution:
    learning_rate = 0.01

    # def get_derivative(
    #     self, 
    #     X: NDArray[np.float64], 
    #     Y: NDArray[np.float64], 
    #     model_prediction: NDArray[np.float64], 
    #     desired_weight: int
    # ) -> float:
    #     return 
    def get_model_prediction(
        self, 
        X: NDArray[np.float64], 
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return X @ weights

    def get_gradient(
        self, 
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        predictions:  NDArray[np.float64],
    ) ->  NDArray[np.float64]:
        error = predictions - Y
        return 2 * (X.T @ error) / len(X)

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        weights = np.array(initial_weights, dtype=np.float64)

        for _ in range(num_iterations):
            predictions = self.get_model_prediction(X, weights)
            weights -= self.learning_rate * self.get_gradient(X, Y, predictions) 
        
        return np.round(weights, 5)








