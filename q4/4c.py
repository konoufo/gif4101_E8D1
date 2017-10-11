import numpy as np
from scipy import linalg

from sklearn import datasets, svm, metrics

# Load the iris data
iris = datasets.load_iris()

# Create a variable for the feature data
X = iris.data

# Create a variable for the target data
y = iris.target


class Num4():
    def _class_means(X, y):

        means = []
        classes = np.unique(y)
        for group in classes:
            Xg = X[y == group, :]
            means.append(Xg.mean(0))
        me = np.asarray(means)
        return me

    def _variance(X, y):
        variance = []
        classes = np.unique(y)
        for group in classes:
            Xv = X[y == group, :]
            Xv = np.concatenate(Xv)
            variance.append(Xv.var(0))
        va = np.asarray(variance)
        return va

    def _fit(X, y):

        means = []
        variance = []
        means_ = Num4._class_means(X, y)
        means = means_
        variance_ = Num4._variance(X, y)
        variance = variance_
        return means, variance

    def _predictproba(X, y):

        means, variance = Num4._fit(X, y)
        l = len(X[:, 0])
        d = len(X[0, :])
        c = 3
        pxc = np.zeros((l, c))
        for j in range(0, c):
            for i in range(0, l):
                var = variance[j]
                xu = X[i, :] - means[j, :]
                xu2 = np.dot(xu, xu)
                ex = np.exp(-0.5 * xu2 / var)
                coef = ((2 * np.pi) ** (0.5 * d)) * np.sqrt(var)
                pxc[i, j] = ex * (1 / coef)

        priors = np.array([1 / 3, 1 / 3, 1 / 3])
        deno = np.zeros(l)
        for i in range(0, l):
            den = (priors[0] * pxc[i, 0]) + (priors[1] * pxc[i, 1]) + (priors[2] * pxc[i, 2])
            deno[i] = den

        pcx = np.zeros((l, c))
        for i in range(0, l):
            for j in range(0, c):
                p = priors[j] * pxc[i, j]
                pcx[i, j] = p / deno[i]
        print(pcx)
        return pcx

    def _predict(X, y, lamb):

        l = len(X[:, 0])
        pcx = Num4._predictproba(X, y)
        predict = np.zeros((l, 1))

        for j in range(0, l):
            pcxt = pcx[j, :]
            if all(i < (1 - lamb) for i in pcxt):
                predict[j] = 3
            else:
                predict[j] = pcx[j, :].argmax()

        return predict


## Erreur totale, en considérant le rejet ayant le même cout qu'un mauvais classement####
erreur = 1 - metrics.accuracy_score(y, Num4._predict(X, y, 0.9))
print(erreur)