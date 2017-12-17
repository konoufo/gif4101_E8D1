from math import exp, log
import random
import pickle

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.spatial import distance
from sklearn import datasets as ds, metrics

from utils.utils import Traceur
from utils.recherche_grille import RechercheGrille


def report(classifier_name, best, params, errors):
    print('\nLa meilleur combinaison d\'hyperparamètres ({}): {}'.format(classifier_name, best))
    print('errors: {}'.format(errors))
    with open('{}_params.pkl'.format(classifier_name), 'wb') as f:
        pickle.dump(params, f)
    with open('{}_errors.pkl'.format(classifier_name), 'wb') as f:
        pickle.dump(errors, f)


def gaussian_k(X,Y, sd):
    # Retourne le noyau gaussien pour X et Y
    k = np.exp(-(distance.euclidean(X, Y)**2)/(sd**2))
    print(k)
    return k


def discrim(A_t, X_t, R_t, sd, x_t):
    # Retourne la fonction discriminante en fonction du jeu de données X et des paramètres du modèle
    h_s = 0
    for i in range(0, len(X_t)):
        h_s += A_t[i] * R_t[i] * gaussian_k(X_t[i], x_t, sd) + A_t[-1]
    return h_s


def loss(A_t, lam, sd, X, y):
    # Retourne la fonction de perte en fonction des paramètres et de l'ensemble Y
    X_y, R_y = Y_sample(A_t, X, y, sd)
    E = 0
    for i in range(0,len(X_y)):
        E += 1 - R_y[i]*discrim(A_t, X, y, sd, X_y[i]) + lam*sum(A_t[:-1])
    return E


def Y_sample(A_t, X_t, R_t, sd):
    # Retourne l'ensemble Y
    X_y = []
    R_y = []
    for i in range(len(X_t)):
        if discrim(A_t, X_t, R_t, sd, X_t[i])*R_t[i] < 1:
            X_y.append(X_t[i])
            R_y.append(R_t[i])
    return X_y,R_y


def grad(A_t, lam, sd, X, y):
    X_y, R_y = Y_sample(A_t, X, y, sd)
    A_t_new = np.empty_like(A_t)
    for t in range(len(R_t)):
        A_t_new[t] = 0
        for k in range(len(R_y)):
            A_t_new[t] += R_y[k] * gaussian_k(X_t[t], X_y[k], sd=sd)
        A_t_new[t] = eta * (R_t[t] * A_t_new[t] - lam)
        A_t_new[-1] = eta * np.sum(R_t)
    return A_t_new


class DiscriminantDG:
    def __init__(self, lam, sd):
        self.lam = lam
        self.sd = sd

    def fit(self, X, y):
        # renvoi les paramètres du modèle.
        optim = fmin_l_bfgs_b(loss, x0=x, approx_grad=False, maxiter=3, fprime=grad, args=(self.lam, self.sd, X, y))
        self.coeffs = optim[0]
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        # Renvoie les étiquettes prédites en fonction des paramètres
        pred = np.empty([X.shape[0], 1])
        for t in range(X.shape[0]):
            pred[t] = discrim(self.coeffs, self.X, self.y, self.sd, X[t])
        return pred

    def score(self, X, y):
        # renvoie le taux de prédiction correct sur l'ensemble du jeu de données de validation
        pred = self.predict(X)
        score = 0
        for t in range(y.shape[0]):
            score += 1 if pred[t] * y[t] > 0 else 0
        return score
        # bien_classees = (self.predict(X) * y)
        # return bien_classees[bien_classees >= 0].shape[0] / y.shape[0]


def convert_target(y, k=0):
    labels = np.unique(y)
    y[y != labels[k]] = -1
    y[y == labels[k]] = 1
    # print('labels: {}'.format(np.unique(y)))
    return y

# le jeu de donnes utilise
data = ds.make_moons(n_samples=1000, noise=0.3)
X_t = data[0][:500, :] # les variables du jeu d'entrainement
R_t = data[1][:500] # les etiquettes du jeu d'entrainement
convert_target(data[1])
print(R_t)
x = np.random.random((1,len(R_t)+1)) # la valeur initiales des poids et de la constante (initialisé aléatoirement)

# le jeu de validation
X_test, y_test = data[0][500:, :], data[1][500:]

lam_set = [.05 + e * 0.01 for e in range(6)]
sd_set = [0.1 + e * 0.1 for e in range(8)]
lam = 0.05 #lambda = 0.05
sd = 0.3 # l'ecart-type = 0.3
eta = 0.008 # taux d'apprentissage

# Trouver les parametres optimaux
t = 1
SS = []
s = 0

paires = list(RechercheGrille.generate_pairs(lam_set, sd_set))
hyparams_combi = [{'lam': pair[0], 'sd': pair[1]} for pair in paires]
recherche = RechercheGrille(X_t, R_t, hyparams_combi, use_score_fun=True)
recherche.fit(DiscriminantDG)
lam_best = paires[recherche.best_hyparams][0]
sigma_best = paires[recherche.best_hyparams][1]
report('DiscriminantDG', {'lambda': lam_best, 'sigma': sigma_best, }, hyparams_combi, recherche.errors)

dg = DiscriminantDG(lam, sd)

# while(s < 0.9):
#     lam_p = lam + random.random[-1,1]*log(t) # On fait ociller le parametre lambda tranquillement dans le temps
#     sd_p = sd + random.random[-1,1]*log(t) # on fait ociller la variance du bb tranquillement dans le temps.
#     if sd > 0 and lam >= 0: # on s'assure que les deux parametres sont plus grand que 0
#         lam = lam_p # si les conditions sont respectes, ont ajuste les params
#         sd = sd_p
#         dg.fit()
#         s = dg.score(X_test, y_test) # on test le score et on retient les parametres dans SS.
#         SS.append((lam,sd,s))
#         t += 1 # incrémente t

param_finaux = dg.coeffs
print(param_finaux)
trc = Traceur(X_test, y_test, fig_index=1)
trc.tracer(dg, discrim_lin=True, fitted=True)
