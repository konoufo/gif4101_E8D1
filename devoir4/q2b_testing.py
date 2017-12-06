import pickle

import tabulate

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.neural_network import MLPClassifier
from sklearn import svm

from utils.utils import Testeur


def load_data():
    with open('data.pkl', 'rb') as f:
        return pickle.load(f)

def save_scores(step, values):
    """Sauvegarder les performances et formatter en tableau."""
    with open('testing_{}_scores.pkl'.format(step), 'wb') as f:
        pickle.dump(values, f)
    with open('table_{}.txt'.format(step), 'w') as f:
        f.write(tabulate.tabulate(values, tablefmt="latex", floatfmt=".3f"))

def get_sigma_min(X=None):
    distances = euclidean_distances(X, X)
    return np.min(distances[distances > 0])


if __name__ == '__main__':
    data = load_data()
    X, y = data[:, :-1], data[:, -1]
    # donnees de test non touchees
    X, y = X[5000:, :], y[5000:]
    gamma = 1/ (2 * (16*get_sigma_min(X))**2)
    clfs = [
        {'clf': svm.SVC, 'params': {'C': 10, 'gamma': gamma}},
        {'clf': KNC, 'params': {'n_neighbors': 3}},
        {'clf': MLPClassifier, 'params': {'hidden_layer_sizes': (100,) * 1, 'activation': 'logistic'}}]
    testeur = Testeur(X, y)
    n_folds = 3
    train_scores = np.empty([len(clfs), n_folds + 1])
    test_scores = np.empty([len(clfs), n_folds + 1])
    time_scores = np.empty([len(clfs), n_folds + 1])
    for i, clf in enumerate(clfs):
        scores = testeur.compute_cross_val(k=n_folds, clfactory=clf['clf'], params=clf['params'])
        train_scores[i, :-1] = scores[0, :]
        train_scores[i, -1] = np.mean(scores[0, :])
        test_scores[i, :-1] = scores[1, :]
        test_scores[i, -1] = np.mean(scores[1, :])
        time_scores[i, :-1] = scores[2, :]
        time_scores[i, -1] = np.mean(scores[2, :])
    save_scores('train', train_scores)
    save_scores('test', test_scores)
    save_scores('duree', time_scores)
