import numpy as np
from typing import Annotated

class LassoRegression:
    def __init__(self,
                 lambda_: Annotated[float, "Regularization strength"] = 0.1,
                 epochs: Annotated[int, "Number of training epochs"] = 1000,
                 learning_rate: Annotated[float, "Learning rate for optimization"] = 0.001,
                 ):
        self.lambda_ = lambda_
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.beta = np.zeros(X.shape[1])
        self.bias = 0
        for epoch in range(self.epochs):
            self.update_weights(X, y)

            if epoch % 100 == 0:
                self.cost += self.lambda_ * np.linalg.norm(self.beta, 1)
                print(f"Epoch: {epoch}, Cost: {self.cost}")

    def predict(self, X: np.ndarray):
        return np.dot(X, self.beta)
    
    def update_weights(self, X: np.ndarray, y: np.ndarray):
        y_pred = np.dot(X, self.beta)
        res = y - y_pred

        gradient = ((-2 / X.shape[0]) * np.dot(X.T, res)) + (self.lambda_ / X.shape[0]) * np.sign(self.beta)

        self.beta -= self.learning_rate * gradient
        self.bias -= self.learning_rate * np.mean(res)
        self.cost = np.mean(np.square(res)) + self.lambda_ * np.linalg.norm(self.beta, 1)

        return self.beta, self.bias, self.cost