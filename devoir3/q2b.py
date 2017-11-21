# -*-coding:utf8-*-
import numpy as np
from sklearn.utils import check_X_y


class GradientDescent:
    """Algorithme de descente de gradient
    taux (float): taux d'apprentissage
    weights (ndarray): poids pour le dioscriminant linéaire
    """
    def __init__(self, taux=0.04):
        self.taux = taux
        self.weights = None

    def fit(self, X, r):
        X, r = check_X_y(X, r)
        r = r[:, np.newaxis]
        weights = np.random.uniform(size=X.shape[1] + 1)[:, np.newaxis]
        it = 0
        itermax = 150
        delta = 1.0
        epsilon = 1e-6
        while it < itermax and delta >= epsilon:
            # valeur courante des poids
            weights_old = np.copy(weights)
            mask = (r * self._hfunc(X, weights=weights))[:, 0] <= 0
            Y = X[mask, :]
            if Y.shape[0] > 0:
                norms_sq = np.sum(np.power(Y, 2), axis=1)[:, np.newaxis]
                h = self._hfunc(Y, weights=weights)
                eX = (r[mask, :] - h) / norms_sq
                dErr = np.dot(eX.T, Y).T
                # mise à jour des w_i avec i=1..D
                weights[:-1] += self.taux * dErr
                # mise à jour de w_0
                weights[-1] += self.taux * np.sum(eX, axis=0)

                print('iter: #{}; normsq: {}; error_sum: {}'.format(it, norms_sq, np.sum(dErr)))
            delta = np.mean(np.abs(weights_old - weights))
            it += 1
        print('\nFinished after {} iterations. err_moy: {}.'.format(it, delta))
        self.weights = weights

    def _is_fitted(self):
        return self.weights is not None

    def _hfunc(self, X, weights=None):
        if weights is None and not self._is_fitted():
            raise AssertionError('Il faut executer la methode fit() avant tout.')
        weights = weights if weights is not None else self.weights
        return np.dot(X, weights[:-1]) + weights[-1]

    def predict(self, X):
        return self._hfunc(X)

    def score(self, X, r):
        X, r = check_X_y(X, r)
        check = self._hfunc(X) * r
        return 1 - (X[check[:, 0] <= 0, :].shape[0] / r.shape[0])
