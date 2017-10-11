import numpy as np
from sklearn.utils import check_X_y


class CustomBayes:

    def __init__(self, cout_lambda=0.4):
        self.cout_lambda = abs(cout_lambda)
        self.check_lambda()
        self.classes = None
        self.variances = None
        self.means = None
        self.priors = None
        self.rejet_label = None
        self.is_fitted = False

    def check_lambda(self):
        if self.cout_lambda < 0:
            raise ValueError('Le coût de rejet est compris entre 0 et 1.')

    def _means(self, X, y):
        """Retourne la matrice des moyennes intra-classe
        :return: (array) matrice de dimension (n_classes, n_features)
        """
        means = []
        for r in self.classes:
            means.append(X[y == r, :].mean(0))
        self.means = np.asarray(means)
        return self.means

    def _variances(self, X, y):
        """Calcule la variance par classe
        :return: (array) vecteur de dimension (n_classes,1) des variances intra-classes sigma_i
        """
        variances = []
        N = self.classes.shape[0]
        for r in self.classes:
            variances.append(np.concatenate(X[y == r, :]).var(0))
        self.variances = np.asarray(variances)
        return self.variances

    def _priors(self, X, y):
        self.priors = np.asarray([X[y == r, :].shape[0] for r in self.classes]) / X.shape[0]
        return self.priors

    def fit(self, X, y):
        """Calculer les paramètres du maximum de vraisemblance et retourner le classifieur résultant.
        :param X: la matrice des échantillons d'entrainement
        :param y: le vecteur des etiquettes correspondant
        :return: self
        """
        X, y = check_X_y(X, y)
        self.classes = np.unique(y)
        self._means(X, y)
        self._variances(X, y)
        self._priors(X, y)
        self.is_fitted = True
        self.rejet_label = np.max(self.classes) + 1
        return self

    def predict(self, X):
        """Retourner le classement à partir des probabilites a posteriori
        :param X:
        :return:
        """
        posteriori = self.predict_proba(X)
        print('Probabilites a posteriori:\n{} \n'.format(posteriori))
        classement = self.classes[np.argmax(posteriori, axis=1)]
        classement[np.max(posteriori, axis=1) < (1 - self.cout_lambda)] = self.rejet_label
        print('Prediction:\n{}\n'.format(classement))
        return classement

    def predict_proba(self, X):
        """Retourner les probabilités a posteriori pour chaque entrée et chaque classe.
        :param X: matrice des entrees
        :return: (array) matrice des probabilités de dimension (n_entrees, n_classes)
        """
        if not self.is_fitted:
            raise AssertionError('Entrainer avec la methode fit(X, y) d\'abord.')
        means, variance = self.means, self.variances
        l = len(X[:, 0])
        d = len(X[0, :])
        c = self.classes.shape[0]
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
        return pcx


    def score(self, X, y):
        """Retourne le cout total du classement (>=0). Plus le score est bas, plus le classement est bon."""
        classement = self.predict(X)
        non_rejet = (classement[:, :] != self.rejet_label)
        cout = np.count_nonzero(y[non_rejet, :] - classement[non_rejet, :])
        cout += (classement.shape[0] - classement[non_rejet].shape[0]) * self.cout_lambda
        return cout


if __name__ == '__main__':
    # essayer python3 -m q4.q4c
    from q3.utils import Testeur, Traceur
    for clf in [CustomBayes(cout_lambda=1.2), CustomBayes(cout_lambda=0.4)]:
        error = Testeur(clf).compute_error()
        notice = 'sans rejet' if clf.cout_lambda > 1 else 'avec rejet'
        print('L\'erreur empirique pour le classifieur {notice} est {e:0.3f} [lambda={l}].'.format(e=error,
                                                                                                   l=clf.cout_lambda,
                                                                                                   notice=notice))
