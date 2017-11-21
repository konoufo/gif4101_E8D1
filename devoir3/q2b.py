# -*-coding:utf8-*-
import numpy as np
from sklearn.utils import check_X_y


class GradientDescent:
    def __init__(self, taux=0.04):
        self.taux = taux
        self.weights = None

    def hx(self, X, r, w):
        h = []
        for t in range(X.shape[0]):
            h.append(w[:-1].T.dot(X[t, :]) + w[-1])
        return np.array(h)

    def get_mal_classees(self, X, r, w):
        bad_cls = []
        h = self.hx(X, r, w)
        for t in range(X.shape[0]):
            e = h[t] * r[t]
            if e <= 0:
                bad_cls.append(t)
        return X[bad_cls, :], r[bad_cls, :]

    def compute_err(self, Y, ry, w):
        h = self.hx(Y, ry, w)
        err = np.zeros_like(w)
        errx = np.zeros_like(Y[:, 0])
        for t in range(Y.shape[0]):
            errx[t] = (ry[t] -  h[t]) / np.sum(Y[t, :] ** 2)
        for i in range(Y.shape[1]):
            dwi = 0
            for t in range(Y.shape[0]):
                dwi += errx[t] * Y[t, i]
            err[i] = self.taux * dwi
        err[-1] = self.taux * np.sum(errx)
        return err

    def fit2(self, X, r):
        X, r = check_X_y(X, r)
        r = r[:, np.newaxis]
        weights = np.random.uniform(size=X.shape[1] + 1)[:, np.newaxis]
        for it in range(150):
            Y, ry = self.get_mal_classees(X, r, weights)
            dw = self.compute_err(Y, ry, weights)
            for i in range(Y.shape[1] + 1):
                weights[i] += dw[i]
        self.weights = weights

    def fit(self, X, r):
        X, r = check_X_y(X, r)
        r = r[:, np.newaxis]
        weights = np.random.uniform(size=X.shape[1] + 1)[:, np.newaxis]
        it = 0
        itermax = 150
        delta = 1.0
        epsilon = 1e-6
        while it < itermax:
            assert X.shape[1] > 1
            mask = (r * self._hfunc(X, weights=weights))[:, 0] <= 0
            Y = X[mask, :]
            norms_sq = np.sum(np.power(Y, 2), axis=1)[:, np.newaxis]
            h = self._hfunc(Y, weights=weights)
            eX = (r[mask, :] - h) / norms_sq
            # print('eX: {}; Y: {}; r: {}; h: {}; nsq: {}'.format(eX.shape, Y.shape, r[mask, :].shape, h.shape,
            #                                                     norms_sq.shape))
            dErr = np.dot(eX.T, Y).T
            # print('dErr: {} ; w: {}'.format(dErr.shape, weights[:-1].shape))
            weights_old = np.copy(weights)
            # mise à jour des w_i avec i=1..D
            weights[:-1] += self.taux * dErr
            # mise à jour de w_0
            weights[-1] += self.taux * np.sum(eX, axis=0)
            delta = np.mean(np.abs(weights_old - weights))
            it += 1
            print('iter: #{}; normsq: {}; error: {}'.format(it, norms_sq, np.sum(dErr)))
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

