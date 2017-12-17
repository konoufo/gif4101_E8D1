import itertools
import numpy as np
from sklearn.utils import check_X_y
from sklearn import model_selection, metrics


class RechercheGrille:
    """Algorithme de recherche en grille pour trouver la meilleur paire
    d'hyperparamètres en se basant sur leur influence conjointe.
        Attributs:
            hyparams_combi: toutes les paires d'hyperparamètres à tester
            best_hyparams (1d-array: int): position de la meilleur paire de paramètres dans `hyparams_combi`
            errors: erreur de validation pour chaque paire d'hyperparamètres testée.
    """
    def __init__(self, X, y, hyparams_combi, use_score_fun=False):
        X, y = check_X_y(X, y)
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, y, train_size=0.5)
        self.hyparams_combi = hyparams_combi
        self.errors = []
        self.best_hyparams = None
        self.use_score_fun = use_score_fun

    @staticmethod
    def generate_pairs(setA, setB):
        return itertools.product(setA, setB)

    def fit(self, clfactory):
        self.errors = [self._validate(clfactory(**params_dict)) for params_dict in self.hyparams_combi]
        self.best_hyparams = np.argmin(self.errors)
        return self

    def _validate(self, clf):
        clf.fit(self.X_train, self.y_train)
        if self.use_score_fun:
            return 1 - clf.score(self.X_test, self.y_test)
        return 1 - metrics.accuracy_score(clf.predict(self.X_test), self.y_test)

