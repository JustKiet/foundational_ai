import numpy as np
from typing import Annotated

class RidgeRegressionModel:
    def __init__(self,
                 lambda_: Annotated[float, 'Regularization parameter'] = 1.0,
                 epochs: Annotated[int, 'Number of epochs'] = 1000,
                 learning_rate: Annotated[float, 'Learning rate'] = 0.01,
                 ):
        self.lambda_ = lambda_
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray):
        _ , n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for epoch in range(self.epochs):
            self.update_weights(X, y)
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Loss: {self.loss}')

    def update_weights(self, X: np.ndarray, y: np.ndarray):
        n_rows , _ = X.shape
        y_pred = X @ self.w
        res = y - y_pred
        dW = ((-2 * (X.T @ (res)) + 2 * self.lambda_ * self.w) / n_rows)
        dB = (-2 * np.sum(res)) / n_rows
        self.w -= self.learning_rate * dW
        self.b -= self.learning_rate * dB
        self.loss = np.mean((y - y_pred) ** 2)
        
    def predict(self, X: np.ndarray):
        return X @ self.w + self.b