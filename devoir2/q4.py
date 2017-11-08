from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NearestCentroid
import matplotlib.pyplot as plt

####### IRIS ##########
# Importer les données
X = datasets.load_iris()

# Standardiser les données
X_S = StandardScaler().fit_transform(X.data)

# Principal component analysis
pca = PCA(n_components=4)
pca_iris = pca.fit(X_S)

# Regarder la proportion de variance expliqués
(pca_iris.explained_variance_/sum(pca_iris.explained_variance_))*100
# La premiere composante explique a elle seule 72.77 % de la variance du jeu de données

# Quelles sont les varianbles qui influence la première composante ?
# Il faut regarder les vecteurs propres
pca_iris.components_
# on voit que la premiere composante est principalement une combinaison linéaire
# des variable sepal_length et petal_length


# Refaire le PCA avec deux composante.
pca = PCA(n_components=2)
pca_iris = pca.fit(X_S)

# Projection dans les deux premières composantes
X_h  = pca_iris.transform(X_S)

pc1 = list(X_h[:,0])
pc2 = list(X_h[:,1])

plt.plot(pc1,pc2, 'ro')
plt.xlabel('Composante 1')
plt.ylabel('Composante 2')
plt.title('Projection du jeu de donnée iris dans ses deux premières composante principales')

plt.show()

######### DIGITS ###########
# Importer les données
X = datasets.load_digits()

# Standardiser les données
X_S = StandardScaler().fit_transform(X.data)


# Principal component analysis
pca = PCA(n_components=64)
pca_digit = pca.fit(X_S)


np.set_printoptions(formatter={'float_kind':'{:f}'.format})
# 1)
# Regarder la proportion de variance expliqués
k = list((pca_digit.explained_variance_/sum(pca_digit.explained_variance_))*100)
for i in range(0,len(k)):
    if sum(k[0:i]) >= 60:
        print(i)
        break
# Les 11+1 = 12 premières composantes capturent au moins 60% de la variance totales du jeu de données

# 2)
# Quelles sont les varianbles qui influence la première composante ?
# Il faut regarder les vecteurs propres
propre = abs(pca_digit.components_[0])
moy = np.mean(propre)
sd = np.sqrt(np.var(propre))
z = []
for i in range(0,len(propre)):
    if propre[i] > (moy + sd):
      z.append(i + 1)

print(z)
### Les valeurs qui ont un plus grand impact ( on choisi plus grand que la moyenne + un ecart type )sont
### les variabe [2, 3, 4, 10, 11, 27, 31, 34, 35, 42, 59, 60]

# 3)
# Refaire le PCA avec deux composante.
pca = PCA(n_components=2)
pca_digit = pca.fit(X_S)

# Projection dans les deux premières composantes
X_h  = pca_digit.transform(X_S)

pc1 = list(X_h[:,0])
pc2 = list(X_h[:,1])

plt.plot(pc1,pc2, 'ro')
plt.xlabel('Composante 1')
plt.ylabel('Composante 2')
plt.title('Projection du jeu de donnée DIGITS dans ses deux premières composante principales')

plt.show()

#### OLIVETTI #####
# Importer les données
X = datasets.fetch_olivetti_faces()


# Standardiser les données
X_S = StandardScaler().fit_transform(X.data)

# PCA
pca = PCA(n_components=4092)
pca_olivetti = pca.fit(X_S)

# Proportion de la variance
k = list((pca_olivetti.explained_variance_/sum(pca_olivetti.explained_variance_))*100)
for i in range(0,len(k)):
    if sum(k[0:i]) >= 60:
        print(i)
        break

# Si on utilise le même critère de sélection que pour DIGITS mais 1.5 fois l'écart-type
propre = abs(pca_olivetti.components_[0])
moy = np.mean(propre)
sd = np.sqrt(np.var(propre))
z = []
for i in range(0,len(propre)):
    if propre[i] > (moy + 1.5*sd):
      z.append(i + 1)

print(z)
# on retient :
#[37, 38, 39, 40, 41, 42, 100, 101, 102, 103, 104, 105, 106, 107, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 226,
# 227, 228, 229, 230, 231, 232, 290, 293, 294, 295, 414, 416, 417, 480, 481, 482, 544, 545, 1651, 1652, 1653, 1715, 1716,
# 1717, 1718, 1719, 1781, 1782, 1783, 1845, 1846, 1847, 1848, 1910, 1911, 1912, 2164, 2165, 2166, 2167, 2168, 2191, 2227,
#  2228, 2229, 2230, 2231, 2232, 2255, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2354, 2355, 2356, 2357, 2358, 2359, 2360,
# 2419, 2420, 2421, 2422, 2423, 2484, 2485, 2486, 2487, 2549, 2550, 2613, 2614, 2740, 2741, 2804, 2805, 2868, 2869, 2932, 2933,
# 3433, 3434, 3497, 3498, 3499, 3500, 3561, 3562, 3563, 3564, 3626, 3627, 3690]


# 3)
# Refaire le PCA avec deux composante.
pca = PCA(n_components=2)
pca_olivetti = pca.fit(X_S)

# Projection dans les deux premières composantes
X_h  = pca_olivetti.transform(X_S)

pc1 = list(X_h[:,0])
pc2 = list(X_h[:,1])

plt.plot(pc1,pc2, 'ro')
plt.xlabel('Composante 1')
plt.ylabel('Composante 2')
plt.title('Projection du jeu de donnée OLIVETTI dans ses deux premières composante principales')

plt.show()


# d) (IRIS)
# Importer les données
X = datasets.load_iris()

# Standardiser les données
X_S = StandardScaler().fit_transform(X.data)

# LDA
lin = LDA(n_components=2)
X_lda = lin.fit(X_S,X.target)

Z = X_lda.transform(X_S)

d1 = list(Z[:,0])
d2 = list(Z[:,1])
#### A inclure dans le rapport
plt.plot(d1,d2, 'ro')
plt.xlabel('Axe 1')
plt.ylabel('Axe 2')
plt.title('Projection du jeu de donnée IRIS dans ses deux premièrs axes LDA')

plt.show()


nciris = NearestCentroid()
V = nciris.fit(Z, X.target)

# e) Taux de classement
X = datasets.load_iris()
# plus proche moyenne
nciris = NearestCentroid()
V = nciris.fit(Z, X.target)
# LDA
erreur = []
prediction_lda = list(V.predict(Z))
label = list(X.target)
for i in range(0,len(prediction_lda)):
    if prediction_lda[i] != label[i]:
        erreur.append(1)
1 - float(sum(erreur))/float(len(label))

# PCA
pca = PCA(n_components=2)
pca_iris = pca.fit(X_S)
X_h  = pca_iris.transform(X_S)
nciris = NearestCentroid()
V = nciris.fit(X_h, X.target)
erreur = []
prediction_pca = list(V.predict(X_h))
label = list(X.target)
for i in range(0,len(prediction_pca)):
    if prediction_pca[i] != label[i]:
        erreur.append(1)
1 - float(sum(erreur))/float(len(label))


# (DIGITS)
# Importer les données
X = datasets.load_digits()

# Standardiser les données
X_S = StandardScaler().fit_transform(X.data)

# LDA
lin = LDA(n_components=2)
X_lda = lin.fit(X_S,X.target)

Z = X_lda.transform(X_S)

d1 = list(Z[:,0])
d2 = list(Z[:,1])
# A inclure dans le rapport
plt.plot(d1,d2, 'ro')
plt.xlabel('Axe 1')
plt.ylabel('Axe 2')
plt.title('Projection du jeu de donnée DIGITS dans ses deux premiers axes LDA')

plt.show()
# e) Taux de classement
X = datasets.load_digits()
# plus proche moyenne
ncdigits = NearestCentroid()
V = ncdigits.fit(Z, X.target)
# LDA
erreur = []
prediction_lda = list(V.predict(Z))
label = list(X.target)
for i in range(0,len(prediction_lda)):
    if prediction_lda[i] != label[i]:
        erreur.append(1)

1 - float(sum(erreur))/float(len(label))
# PCA
pca = PCA(n_components=2)
pca_digit = pca.fit(X_S)
X_h  = pca_digit.transform(X_S)
ncdigits = NearestCentroid()
V = ncdigits.fit(X_h, X.target)
erreur = []
prediction_pca = list(V.predict(X_h))
label = list(X.target)
for i in range(0,len(prediction_pca)):
    if prediction_pca[i] != label[i]:
        erreur.append(1)
1 - float(sum(erreur))/float(len(label))




# (OLIVETTI)
# Importer les données
X = datasets.fetch_olivetti_faces()

# Standardiser les données
X_S = StandardScaler().fit_transform(X.data)

# LDA
lin = LDA(n_components=2)
X_lda = lin.fit(X_S,X.target)

Z = X_lda.transform(X_S)

d1 = list(Z[:,0])
d2 = list(Z[:,1])

# a inclure dans le rapport
plt.plot(d1,d2, 'ro')
plt.xlabel('Axe 1')
plt.ylabel('Axe 2')
plt.title('Projection du jeu de donnée OLIVETTI dans ses deux premièrs axes LDA')

plt.show()

# e) Taux de classement
X = datasets.fetch_olivetti_faces()
# plus proche moyenne
ncolivetti = NearestCentroid()
V = ncolivetti.fit(Z, X.target)
# LDA
erreur = []
prediction_lda = list(V.predict(Z))
label = list(X.target)
for i in range(0,len(prediction_lda)):
    if prediction_lda[i] != label[i]:
        erreur.append(1)

1 - float(sum(erreur))/float(len(label))
# PCA
pca = PCA(n_components=2)
pca_olivetti = pca.fit(X_S)
X_h  = pca_olivetti.transform(X_S)
ncolivetti = NearestCentroid()
V = ncolivetti.fit(X_h, X.target)
erreur = []
prediction_pca = list(V.predict(X_h))
label = list(X.target)
for i in range(0,len(prediction_pca)):
    if prediction_pca[i] != label[i]:
        erreur.append(1)
1 - float(sum(erreur))/float(len(label))
### Recommit