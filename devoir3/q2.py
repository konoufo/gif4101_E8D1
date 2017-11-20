import numpy as np
from sklearn.utils import check_X_y


class GradientDescent:
    def __init__(self, taux=0.2):
        self.taux = taux
        self.weights = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        y = y[:, np.newaxis]
        norms_sq = np.sum(np.power(X, 2), axis=1)
        weights = np.random.uniform(size=X.shape[1] + 1)[:, np.newaxis]
        it = 0
        itermax = 150
        delta = 1.0
        epsilon = 1e-6
        while it < itermax and delta > epsilon:
            weights_old = np.copy(weights)
            eX = (y - np.dot(X, weights[:-1]) - weights[-1]) / norms_sq
            dErr = np.dot(eX.T, X[:, ]).T
            # mise à jour des w_i avec i=1..D
            weights[:-1] += self.taux * dErr
            # mise à jour de w_0
            weights[-1] += self.taux * eX
            delta = np.min(np.abs(weights_old - weights))
            it += 1
        self.weights = weights

    def _hfunc(self, X):
        return np.dot(X, self.weights[:-1]) + self.weights[-1]

    def predict(self, X):
        return self._hfunc(X)

    def score(self, X, y):
        X, y = check_X_y(X, y)
        check = self._hfunc(X) * y
        return 1 - (check[check < 0] / y.shape[0]) 
