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
        self.is_fitted = False

    def check_lambda(self):
        if self.cout_lambda > 1:
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
            variances.append(2/N * np.sum(X[y == r, :].var(0)))
        self.variances = np.c_[variances]
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
        return self

    def predict(self, X):
        """Retourner le classement à partir des probabilites a posteriori
        :param X:
        :return:
        """
        posteriori = self.predict_proba(X)
        print(posteriori)
        classement = self.classes[np.argmax(posteriori, axis=1)]
        print(np.max(posteriori, axis=1))
        classement[np.max(posteriori, axis=1) < (1 - self.cout_lambda)] = -1
        print(classement)
        return classement

    def predict_proba(self, X):
        """Retourner les probabilités a posteriori pour chaque entrée et chaque classe.
        :param X: matrice des entrees
        :return: (array) matrice des probabilités de dimension (n_entrees, n_classes)
        """
        if not self.is_fitted:
            raise AssertionError('Train the classifier first.')
        h = []
        N = self.classes.shape[0]
        for i in range(self.classes.shape[0]):
            m = self.means[i, :]
            v = self.variances[i]
            h.append(np.c_[1/np.sqrt(v**N) * np.exp(-1/(2*v) * np.sum((X - m)**2, axis=1))])
        self.h = np.hstack(tuple(h))
        evidence = np.c_[np.dot(self.h, self.priors)]
        print('h: {}\n'.format(self.h))
        print('Prioris: {}\n'.format(self.priors))
        print('L\'evidence est: {}'.format(evidence))
        return self.h / evidence


    def score(self, X, y):
        """Retourne le cout total du classement (>=0). Plus le score est bas, plus le classement est bon."""
        classement = self.predict(X)
        non_rejet = (classement[:, :] >= 0)
        cout = np.count_nonzero(y[non_rejet, :] - classement[non_rejet, :])
        cout += (classement.shape[0] - classement[non_rejet].shape[0]) * self.cout_lambda
        return cout


if __name__ == '__main__':
    from q3.utils import Testeur
    error = Testeur(CustomBayes).compute_error()
    print('L\'erreur empirique pour le classifieur avec rejet est {:0.3f}.'.format(error))
