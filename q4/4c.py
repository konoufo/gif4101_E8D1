import numpy as np
from scipy import linalg

from sklearn import datasets

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
        c = len(classes)
        pxc = np.zeros((l, c))
        for j in range(0, c):
            for i in range(0, l):
                var = variance[j]
                xu = X[i, :] - means[j, :]
                xu2 = np.dot(xu, xu)
                ex = np.exp(-0.5 * xu2 / var)
                coef = ((2 * np.pi) ** (0.5 * d)) * np.sqrt(var)
                pxc[i, j] = ex * (1 / coef)

                # classe = np.unique(y)
                # for group in classe:
                #   Xg = X[y == group, :]

        priors = np.array([1 / 3, 1 / 3, 1 / 3])

        for j in range(0, c):
            for i in range(0, l):
                pcx[i, j] = priors(j) * pxc[i, j]

        return pcx, priors