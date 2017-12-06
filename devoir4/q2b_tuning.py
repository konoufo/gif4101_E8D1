import pickle

import numpy as np
from sklearn import svm
from sklearn.datasets import fetch_mldata
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from utils.utils import Testeur
from utils.recherche_grille import RechercheGrille


def report(classifier_name, best, params, errors):
    print('\nLa meilleur combinaison d\'hyperparamètres ({}): {}'.format(classifier_name, best))
    with open('{}_params.pkl'.format(classifier_name), 'wb') as f:
        pickle.dump(params, f)
    with open('{}_errors.pkl'.format(classifier_name), 'wb') as f:
        pickle.dump(errors, f)

def save_state(data=None):
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    pendigits = fetch_mldata('uci-20070111 pendigits')
    np.random.shuffle(pendigits.data)
    X, y = pendigits.data[:, :-1], pendigits.data[:, -1]
    X_train, y_train = X[:5000, :], y[:5000]
    X_test, y_test = X[5000:, :], y[5000:]
    save_state(pendigits.data)
    # SVM avec hyperparametres C et sigma
    C_set = [10**p for p in range(-5, 6)]
    distances = euclidean_distances(X, X)
    sigma_min = np.min(distances[distances > 0])
    sigma_set = [(2**k)*sigma_min for k in range(7)]
    paires = list(RechercheGrille.generate_pairs(C_set, sigma_set))
    hyparams_combi = [{'C': pair[0], 'gamma': 1/(2 * pair[1]**2)} for pair in paires]
    recherche = RechercheGrille(X_train, y_train, hyparams_combi)
    recherche.fit(svm.SVC)
    C_best = paires[recherche.best_hyparams][0]
    sigma_best = paires[recherche.best_hyparams][1]
    report('svm', {'C': C_best, 'sigma': sigma_best, 'sigma/sigma_min': sigma_best/sigma_min}, hyparams_combi,
         recherche.errors)

    # k-voisins avec hyperparametre k
    k_set = [1, 3, 5, 7, 11]
    validation_err = [Testeur(X_train, y_train, KNeighborsClassifier(n_neighbors=k)).compute_error(test_split=.5)
                      for k in k_set]
    k_best = k_set[np.argmin(validation_err)]
    report('k_voisins', {'k': k_best}, k_set, validation_err)

    # Perceptron multi-couche avec hyperparametres nbre de couches et fonction d'activation
    n_couches_set = list(range(5))
    activation_set = ['logistic', 'tanh', 'relu']
    paires = list(RechercheGrille.generate_pairs(n_couches_set, activation_set))
    hyparams_combi = [{'hidden_layer_sizes': (100,) * pair[0], 'activation': pair[1]} for pair in paires]
    recherche = RechercheGrille(X_train, y_train, hyparams_combi)
    recherche.fit(MLPClassifier)
    n_couches_best = paires[recherche.best_hyparams][0]
    activation_best = paires[recherche.best_hyparams][1]
    report('perceptron_multi_couche', {'nombre_couches_caches': n_couches_best, 'fonction_activation': activation_best,},
           hyparams_combi, recherche.errors)
