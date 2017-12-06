from scipy.optimize import fmin_l_bfgs_b
from scipy.spatial import distance
from sklearn import datasets as ds
import numpy as np
from math import exp, log
import random

def gaussian_k(X,Y):
    # Retourne le noyau gaussien pour X et Y
    return exp(-(distance.euclidean(X,Y)**2)/(sd**2))

def discrim(A_t,x_t):
    # Retourne la fonction discriminante en fonction du jeu de données X et des paramètres du modèle
    h_s = 0
    for i in range(0, len(X_t)-1):
        h_s += A_t[i]*R_t[i]*gaussian_k(X_t[i],x_t) + A_t[-1]
    return h_s

def loss(A_t, **param):
    # Retourne la fonction de perte en fonction des paramètres et de l'ensemble Y
    X_y, R_y = Y_sample(A_t)
    E = 0
    for i in range(0,len(X_y)-1):
        E += 1 - R_y[i]*discrim(A_t, X_y[i]) + lam*sum(A_t[:-1])
    return E

def Y_sample(A_t):
    # Retourne l'ensemble Y
    X_y = []
    R_y = []
    for i in range(0,len(X_t)-1):
        if discrim(A_t,X_t[i])*R_t[i] < 1:
            X_y.append(X_t[i])
            R_y.append(R_t[i])
    return X_y,R_y

# ici le problème
def grad(A_t):
    X_y, R_y = Y_sample(A_t)
    for i in range(1,len(R_y)):


def fit():
    # renvoi les paramètres du modèle.
    y = fmin_l_bfgs_b(loss, x0=x, approx_grad=False,maxiter=3,fprime=grad)
    return y[0]

def predict(a,x_t):
    # Renvoie les étiquettes prédites en fonction des paramètres
    return discrim(a,x_t)

def score(test,a):
    # renvoie le taux de prédiction correct sur l'ensemble du jeu de données de validation
    s =  0 # succes
    for i in range(0,len(test)-1):
        if predict(a, test[0][i]) == test[1][i]:
            s += 1
    return s/len(test)

# le jeu de donnes utilise
train = ds.make_moons(n_samples=500, noise=0.3)
X_t = train[0] # les variables du jeu d'entrainement
R_t = train[1] # les etiquettes du jeu d'entrainement
x = np.random.random((1,len(R_t)+1)) # la valeur initiales des poids et de la constante (initialisé aléatoirement)

# le jeu de validation
test = ds.make_moons(n_samples=500, noise=0.3)

lam = 0.05 #lambda = 0.05
sd = 0.3 # l'ecart-type = 0.3


# Trouver les parametres optimaux
t = 1
SS = []
while(s > 0.1):
    lam_p = lam + random.random[-1,1]*log(t) # On fait ociler le parametre lambda tranquillement dans le temps
    sd_p = sd + random.random[-1,1]*log(t) # on fait ociler la variance du bb tranquillement dans le temps.
    if sd > 0 & lam >= 0: # on s'assure que les deux parametres sont plus grand que 0
        lam = lam_p # si les conditions sont respectes, ont ajuste les params
        sd = sd_p
        s = score(test, fit()) # on test le score et on retient les parametres dans SS.
        SS.append((lam,sd,s))
        t += 1 # incrémente t

