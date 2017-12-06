from scipy.optimize import fmin_l_bfgs_b
from scipy.spatial import distance
from sklearn import datasets as ds
import numpy as np
from math import exp
import random

def gaussian_k(X,Y,sd=0.3):
    return exp(-(distance.euclidean(X,Y)**2)/(sd**2))

def discrim(A_t,x_t):
    h_s = 0
    for i in range(0, len(X_t)-1):
        h_s = h_s + A_t[i]*R_t[i]*gaussian_k(X_t[i],x_t) + A_t[-1]
    return h_s

def loss(A_t, **param):
    X_y, R_y = Y_sample(A_t)
    E = 0
    for i in range(0,len(X_y)-1):
        E = E + 1 - R_y[i]*discrim(A_t, X_y[i]) + lam*sum(A_t[:-1])
    return E

def Y_sample(A_t):
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


k = ds.make_moons(n_samples=1000, noise=0.3)
X_t = k[0]
R_t = k[1]
lam = 0.05
x0 = []
for i in range(0,len(R_t)-1):
    x0.append(random.randint(1,12))
x = np.random.random((1,len(R_t)+1))
y = fmin_l_bfgs_b(loss, x0=x, approx_grad=False,maxiter=3,fprime=grad)
print(y[0])
