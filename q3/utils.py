import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn import datasets


iris = datasets.load_iris()


class Traceur:
    X = iris.data
    y = iris.target
    feature_names = ('Longueur des sépales', 'Largeur des sépales', 'Longueur des pétales', 'Largeur des pétales')
    # selection de couleurs
    cmap_light = ListedColormap(['#FFFFAA', '#AAAAFF', '#FFAAAA'])
    cmap_bold = ListedColormap(['#FFFF00', '#0000FF', '#FF0000'])

    def __init__(self, fig_index=1):
        self.index = fig_index

    def mesh(self, x1, x2, h=.01):
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
        clf (type): classe de classifieur
        """
        X = self.X
        data = np.matrix([X[:, x1], X[:, x2]])
        data = np.matrix.transpose(data)
        return clf().fit(data, self.y)

    def generate_scatter(self, x1, x2):
        X = self.X
        target_names = iris.target_names
        for target in np.unique(self.y):
            plt.scatter(X[self.y == target, x1], X[self.y == target, x2], c=self.cmap_bold.colors[target], edgecolor='k',
                        label=target_names[target])
            # plt.scatter(X[50:99, x1], X[50:99, x2], c='#0000FF', edgecolor='k',
            #             label=target_names[1])
            # plt.scatter(X[100:149, x1], X[100:149, x2], c='#FF0000', edgecolor='k',
            #             label=target_names[2])

    def tracer(self, clf, x1, x2, subplot=0, loc=4):
        clf = self.fit(clf, x1, x2)
        xx, yy = self.mesh(x1, x2)
        prediction_map = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        plt.figure(self.index, figsize=(8, 6))
        plt.subplot(220 + subplot)
        plt.pcolormesh(xx, yy, prediction_map, cmap=self.cmap_light)
        # Mise en graphique des points d'entrainement
        self.generate_scatter(x1, x2)
        plt.xlabel(self.feature_names[x1])
        plt.ylabel(self.feature_names[x2])
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.legend(loc=loc,
                   fontsize=8)

    def tracer_tous_les_duos(self, clf, show=True):
        # paires = itertools.combinations([0, 1, 2, 3], 2)
        paires = ((0, 3), (0, 1), (2, 1), (2, 3))
        subplot = 1
        # position des légendes
        legend_locs = (4, 1, 1, 4)
        for i, paire in enumerate(paires):
            self.tracer(clf, paire[0], paire[1], subplot, legend_locs[i])
            subplot += 1
        if show:
            plt.show()


class Testeur:
    """Permet de traiter differents classifieurs sur le dataset iris
    clfactory (class/type): classe du classifieur
    X (arrau): donnees de iris
    y (array): etiquettes de iris
    """
    X = iris.data
    y = iris.target

    def __init__(self, clfactory=None):
        self.clfactory = clfactory

    def compute_error(self, clfactory=None):
        clf = (clfactory or self.clfactory)()
        clf.fit(self.X, self.y)
        return 1 - metrics.accuracy_score(clf.predict(self.X), self.y)
