import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.utils import check_X_y
from sklearn import metrics
from sklearn.base import is_classifier, clone
from sklearn.utils import indexable
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _fit_and_score


def cross_val_score(estimator, X, y=None, groups=None, scoring=None, cv=None,
                    n_jobs=1, verbose=0, fit_params=None,
                    pre_dispatch='2*n_jobs'):
    """Une copie quasi-verbatim de la méthode de scikit-learn pour cross-validation.
    Aux exception près, que celle-ci retourne les scores pour l'ensemble d'entrainement également
    ainsi que le temps de calcul pour les évaluations de score."""
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    scores = parallel(delayed(_fit_and_score)(clone(estimator), X, y, scorer,
                                              train, test, verbose, None,
                                              fit_params, return_train_score=True)
                      for train, test in cv.split(X, y, groups))
    return np.array(scores)


class Traceur:
    """Contient les méthodes pour représenter graphiquement les régions de décision
    de'un quelconque classifieur sur des données."""

    # selection de couleurs
    cmap_light = ListedColormap(['#FFFFAA', '#AAAAFF', '#FFAAAA', '#008000'])
    cmap_bold = ListedColormap(['#FFFF00', '#0000FF', '#FF0000', ])

    def __init__(self, X, y, target_names=None, fig_index=1, avec_rejet=False):
        self.index = fig_index
        self.avec_rejet = avec_rejet
        self.X, self.y = check_X_y(X, y)
        self.target_names = ['C{}'.format(tr + 1) for tr in np.unique(y)] if target_names is None else target_names
        self.feature_names = ['feature {}'.format(n + 1) for n in range(X.shape[1])]

    def mesh(self, x1=0, x2=1, h=.01):
        """Crée un quadrillage sur les 2 dimensions fournies
        x1 (int): index de la première feature/dimension
        x2 (int): index de la seconde feature/dimension
        h (float): espacement du quadrillage
        """
        X = self.X
        x_min, x_max = X[:, x1].min() - 1, X[:, x1].max() + 1
        y_min, y_max = X[:, x2].min() - 1, X[:, x2].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy

    def fit(self, clf, x1, x2):
        """Entraine avec le classifieur fourni
        clf (type): objet `class` du type de classifieur
        """
        X = self.X
        data = np.matrix([X[:, x1], X[:, x2]])
        data = np.matrix.transpose(data)
        try:
            clf = clf()
        except TypeError:
            clf = clf
        return clf.fit(data, self.y)

    def generate_scatter(self, x1, x2):
        X = self.X
        target_names = self.target_names
        for target in np.unique(self.y):
            plt.scatter(X[self.y == target, x1], X[self.y == target, x2], c=self.cmap_bold.colors[target], edgecolor='k',
                        label=target_names[target])
        if self.avec_rejet:
            nans = np.full_like(X[:,0], np.nan, dtype=np.double)
            plt.scatter(nans, nans, c='#008000', edgecolor='k', label='rejet')

    def tracer_hyperplan(self, discrims, xx):
        for i in range(len(discrims)):
            discrim = discrims[i]
            w1, w2, w0 = discrim.weights[0, 0], discrim.weights[1, 0], discrim.weights[2, 0]
            plt.plot(xx[0, :], (-w1 * xx[0, :] - w0) / w2, linestyle='--', linewidth=2, c=self.cmap_bold.colors[i])

    def tracer(self, clf, x1=0, x2=1, layout=(1, 1), subplot=0, loc=4, fitted=False, discrim_lin=False):
        if not fitted:
            clf = self.fit(clf, x1, x2)
        xx, yy = self.mesh(x1, x2)
        prediction_map = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        print(np.unique(prediction_map))
        plt.figure(self.index, figsize=(8, 6))
        if max(layout) > 1:
            plt.subplot(220 + subplot)
        cmap = np.array(self.cmap_light.colors)[np.unique(prediction_map)]
        plt.pcolormesh(xx, yy, prediction_map, cmap=ListedColormap(list(cmap)))
        # Mise en graphique des points d'entrainement
        if discrim_lin:
           self.tracer_hyperplan(clf.discrims, xx)
        self.generate_scatter(x1, x2)

        plt.xlabel(self.feature_names[x1])
        plt.ylabel(self.feature_names[x2])
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.legend(loc=loc,
                   fontsize=8)


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

    def compute_error(self, clfactory=None):
        try:
            clf = (clfactory or self.clfactory)()
        except TypeError:
            clf = clfactory or self.clfactory
        clf.fit(self.X, self.y)
        return 1 - metrics.accuracy_score(clf.predict(self.X), self.y)

    def compute_cross_val(self, k=3, clfactory=None):
        try:
            clf = (clfactory or self.clfactory)()
        except TypeError:
            clf = clfactory or self.clfactory
        self.scores = cross_val_score(clf, self.X, self.y, cv=k)
        return self.scores[:, 1]
