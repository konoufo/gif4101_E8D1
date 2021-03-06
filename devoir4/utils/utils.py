import numpy as np
import time
from sklearn.utils import check_X_y
from sklearn import metrics, model_selection
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



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
