import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification

from q2b import GradientDescent
from utils import Traceur


def randomize_test_train(X, y):
    X = np.copy(X)
    y = np.copy(y)
    random_slice = np.arange(X.shape[0])
    np.random.shuffle(random_slice)
    trainX = X[random_slice[:50], :]
    trainR = y[random_slice[:50]]
    testX = X[random_slice[50:], :]
    testR = y[random_slice[50:]]
    return trainX, trainR, testX, testR


def convert_target(y, k=0):
    labels = np.unique(y)
    y[y != labels[k]] = -1
    y[y == labels[k]] = 1
    # print('labels: {}'.format(np.unique(y)))
    return y


class OneVSAll:
    """Classifieur multi-classes par descente de gradient.
    On utilise l'approche 'un contre tous'."""
    def fit(self, X, y):
        orig_y = y
        labels = np.unique(y)
        discrims = []
        for i in range(labels.shape[0]):
            y = np.copy(orig_y)
            y = convert_target(y, i)
            gd = GradientDescent()
            gd.fit(X, y)
            discrims.append(gd)
        self.discrims = discrims
        return self

    def predict(self, X):
        all_h = self.discrims[0].hfunc(X)
        for gd in self.discrims[1:]:
            all_h = np.concatenate((all_h, gd.hfunc(X)), axis=1)
        return np.argmax(all_h, axis=1)

    def score(self, X, y):
        pred = self.predict(X)
        return (pred == y).sum() / pred.shape[0]

    def get_params(self, deep=True):
        # pour rendre ce classifieur compatible avec les modules de sklearn
        return {}

    def set_params(self, **parameters):
        # pour rendre ce classifieur compatible avec les modules de sklearn
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def report(self):
        print('''
            h1: w1={:.3f}, w2={:.3f}; w0={:.3f}
            h2: w1={:.3f}, w2={:.3f}; w0={:.3f}
            h3: w1={:.3f}, w2={:.3f}; w0={:.3f}
            '''.format(self.discrims[0].weights[0, 0], self.discrims[0].weights[1, 0], self.discrims[0].weights[2, 0],
                       self.discrims[1].weights[0, 0], self.discrims[1].weights[1, 0], self.discrims[1].weights[2, 0],
                       self.discrims[2].weights[0, 0], self.discrims[2].weights[1, 0], self.discrims[2].weights[2, 0]))


if __name__ == '__main__':
    gd = GradientDescent()
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1,
                               n_classes=2)
    trainX, trainR, testX, testR = randomize_test_train(X, y)
    trainR = convert_target(trainR)
    print('\n--------------- Probleme avec 2 classes: ---------------')
    gd.fit(trainX, trainR)
    print('w1={:.3f}, w2={:.3f}; w0={:.3f}'.format(gd.weights[0, 0], gd.weights[1, 0], gd.weights[2, 0]))
    print('Score (test): {:.3f}'.format(gd.score(testX, testR)))
    print('Score (tout): {:.3f}'.format(gd.score(X, y)))

    print('\n--------------- Regions de decision avec 2 classes: ---------------')
    trc = Traceur(X, y, fig_index=1)
    trc.tracer(gd, fitted=True)
    print('w1={:.3f}, w2={:.3f}; w0={:.3f}'.format(gd.weights[0, 0], gd.weights[1, 0], gd.weights[2, 0]))
    print('Score (test): {:.3f}'.format(gd.score(testX, testR)))
    print('Score (tout): {:.3f}'.format(gd.score(X, y)))
    plt.title('Régions de décision du classifieur Descente de gradient avec 2 classes')

    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1,
                               n_classes=3)
    trainX, trainR, testX, testR = randomize_test_train(X, y)
    print('\n--------------- Probleme multi-classes avec 3 classes (Un contre tous): ---------------')
    ova = OneVSAll()
    ova.fit(trainX, trainR)
    ova.report()
    print('Score (test): {:.3f}'.format(ova.score(testX, testR)))
    print('Score (tout): {:.3f}'.format(ova.score(X, y)))

    trc = Traceur(X, y, fig_index=2)
    print('\n--------------- Regions de decision avec 3 classes (Un contre tous): ---------------')
    trc.tracer(ova, discrim_lin=True)
    ova.report()
    print('Score (tout): {:.3f}'.format(ova.score(X, y)))
    plt.title('Régions de décision du classifieur Descente de Gradient (K-classes). K = 3')
    plt.show()
