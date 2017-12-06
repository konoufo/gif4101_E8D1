import numpy as np
import time
from sklearn.utils import check_X_y
from sklearn import metrics, model_selection


class Testeur:
    """Permet de traiter differents classifieurs sur des données.
    Attributs:
        clfactory (class/type): type du classifieur
        X (array): donnees
        y (array): etiquettes
    """

    def __init__(self, X, y, clfactory=None):
        self.clfactory = clfactory
        self.X, self.y = check_X_y(X, y)
        self.scores = None

    def compute_error(self, clfactory=None, test_split=None, fitted=False):
        try:
            clf = (clfactory or self.clfactory)()
        except TypeError:
            clf = clfactory or self.clfactory
        X_train = X_test = self.X
        y_train = y_test = self.y
        if test_split is not None:
            X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=test_split)
        if not fitted:
            clf.fit(X_train, y_train)
        return 1 - metrics.accuracy_score(clf.predict(X_test), y_test)

    def compute_cross_val(self, k=3, clfactory=None, params=None):
        try:
            params = params or {}
            clf = (clfactory or self.clfactory)(**params)
        except TypeError:
            clf = clfactory or self.clfactory
        kf = model_selection.KFold(k, shuffle=False)
        scores = np.empty([3, k])
        ifold = 0
        for train_index, test_index in kf.split(self.X):
            debut = time.clock()
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            clf.fit(X_train, y_train)
            scores[0, ifold] = 1 - metrics.accuracy_score(clf.predict(X_train), y_train)
            scores[1, ifold] = 1 - metrics.accuracy_score(clf.predict(X_test), y_test)
            duree = time.clock() - debut
            scores[2, ifold] = duree
            ifold += 1
        self.scores = scores
        return scores
