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

    # =====================================================================
    # 利用 Jacobian 矩陣與微積分連鎖律 (Matrix Chain Rule) 進行全向量化梯度計算
    #
    # 數學推導過程 (Jacobian Derivation):
    # 1. 損失函數 (MSE): L = (1/N) * ||E||² = (1/N) * EᵀE  , 其中 E = XW - Y
    # 2. 依據連鎖律: ∂L/∂W = ∂L/∂E * ∂E/∂W
    #    - 外層微分: ∂L/∂E = (2/N) * Eᵀ           [維度: 1 × N]
    #    - 內層微分 (Jacobian): ∂E/∂W = X         [維度: N × D]
    # 3. 合體得到列梯度: ∂L/∂W = (2/N) * EᵀX       [維度: 1 × D]
    # 4. 轉置成立向量 (配合 W 的維度): ∇_w L = (2/N) * XᵀE  [維度: D × 1]
    # =====================================================================

        for _ in range(num_iterations):
            grads = 2 * X.T @ (X @ W - Y) / len(X)
            W -= self.learning_rate * grads

        return np.round(W, 5)
            


