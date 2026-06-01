import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(
        self, 
        model_prediction: NDArray[np.float64], 
        ground_truth: NDArray[np.float64], 
        N: int, 
        X: NDArray[np.float64], 
        desired_weight: int
    ) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(
        self, 
        X: NDArray[np.float64], 
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.squeeze(X @ weights)

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        # W = initial_weights.copy()
    
        # # For each iteration:
        # for _ in range(num_iterations):
    
        # #   1. Compute predictions with get_model_prediction(X, weights)
        #     preds = self.get_model_prediction(X, W)
    
        # #   2. For each weight index j, compute gradient with get_derivative()
        #     for j in range(len(W)):
        #         grads = self.get_derivative(preds, Y, len(X), X, j)
        #     #   3. Update: weights[j] -= learning_rate * gradient
        #         W[j] -= self.learning_rate * grads

        # return np.round(W, 5)

        W = initial_weights

        for _ in range(num_iterations):
            grads = 2 * X.T @ (X @ W - Y) / len(X)
            W -= self.learning_rate * grads

        return np.round(W, 5)
            


