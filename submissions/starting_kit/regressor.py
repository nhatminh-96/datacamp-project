import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, Y):
        self.X_scaling_ = np.max(X, axis=0, keepdims=True)
        self.Y_scaling_ = np.max(Y, axis=0, keepdims=True)
        self.model.fit(X / self.X_scaling_, Y / self.Y_scaling_)

    def predict(self, X):
        res = self.model.predict(X / self.X_scaling_) * self.Y_scaling_
        return res.reshape(-1,1)

