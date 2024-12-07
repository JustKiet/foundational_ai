import numpy as np

class OLSRegressionModel:
    def __init__(self,
                 X: np.array,
                 y: np.array,
                ):
        self.X = X
        self.y = y
        self.model = None

    def add_intercept(self, X: np.array):
        X = np.c_[np.ones(X.shape[0]), X] if X.shape[1] == 3 else X
        return X
    
    def fit(self, X: np.array, y: np.array):
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        return beta_hat
    
    def predict(self, X: np.array, beta_hat: np.array):
        y_hat = X @ beta_hat
        return y_hat
    
    def residuals(self, y: np.array, y_hat: np.array):
        residuals = y - y_hat
        return residuals
    
    def MSE(self, y: np.array, y_hat: np.array):
        MSE = np.mean((y - y_hat) ** 2)
        return MSE
    
    def forward(self):
        # Add intercept
        self.add_intercept(X=self.X)

        # Fit model
        beta_hat = self.fit(X=self.X, y=self.y)

        # Predict values
        y_hat = self.predict(X=self.X, beta_hat=beta_hat)

        # Compute residuals
        residuals = self.residuals(y=self.y, y_hat=y_hat)

        # Calculate MSE
        MSE = self.MSE(y=self.y, y_hat=y_hat)

        return beta_hat, y_hat, residuals, MSE