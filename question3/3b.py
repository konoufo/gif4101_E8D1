import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn import datasets
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.neighbors import NearestCentroid as nc
##########Gestion des données ##########################
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_name = iris.target_names[:3]

################### quadratic discriminant #################################

### Erreur
qda = qda()
qda.fit(iris.data, iris.target)
erreur_qda = 1 - metrics.accuracy_score(qda.predict(iris.data),iris.target)
print('L''erreur pour le discriminant quadratique est %.3f.' % erreur_qda)


## Creer un mesh pour faire le trace
Xq = np.matrix([X[:,0],X[:,3]])
Xq = np.matrix.transpose(Xq)
clf = qda.fit(Xq,y)

h = .01       # Espacement du mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 3].min() - 1, X[:, 3].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

Yq = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Mettre le resultat dans un trace couleur
cmap_light = ListedColormap(['#FFFFAA', '#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#FFFF00', '#0000FF', '#FF0000'])

Yq1 = Yq.reshape(xx.shape)
plt.figure(1,figsize = (8,6))
plt.subplot(221)
plt.pcolormesh(xx, yy, Yq1, cmap=cmap_light)
# Mise en graphique des points d<entrainement
plt.scatter(X[0:49, 0], X[0:49, 3], c='#FFFF00', edgecolor='k',
            label = target_name[0])
plt.scatter(X[50:99, 0], X[50:99, 3], c='#0000FF', edgecolor='k',
            label = target_name[1])
plt.scatter(X[100:149, 0], X[100:149, 3], c='#FF0000', edgecolor='k',
            label = target_name[2])
plt.xlabel('Longueur des sépales')
plt.ylabel('Largeur des pétales')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(loc=4,
     fontsize=8)

## Creer un mesh pour faire le trace
Xq = np.matrix([X[:,0],X[:,1]])
Xq = np.matrix.transpose(Xq)
clf = qda.fit(Xq,y)

h = .01       # Espacement du mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

Yq = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Mettre le resultat dans un trace couleur
cmap_light = ListedColormap(['#FFFFAA', '#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#FFFF00', '#0000FF', '#FF0000'])

Yq1 = Yq.reshape(xx.shape)
plt.figure(1,figsize = (8,6))
plt.subplot(223)
plt.pcolormesh(xx, yy, Yq1, cmap=cmap_light)
# Mise en graphique des points d<entrainement
plt.scatter(X[0:49, 0], X[0:49, 1], c='#FFFF00', edgecolor='k',
            label = target_name[0])
plt.scatter(X[50:99, 0], X[50:99, 1], c='#0000FF', edgecolor='k',
            label = target_name[1])
plt.scatter(X[100:149, 0], X[100:149, 1], c='#FF0000', edgecolor='k',
            label = target_name[2])
plt.xlabel('Longueur des sépales')
plt.ylabel('Largeur des sépales')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(loc=1,
     fontsize=8)

## Creer un mesh pour faire le trace
Xq = np.matrix([X[:,2],X[:,1]])
Xq = np.matrix.transpose(Xq)
clf = qda.fit(Xq,y)

h = .01       # Espacement du mesh
x_min, x_max = X[:, 2].min() - 1, X[:, 2].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

Yq = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Mettre le resultat dans un trace couleur
cmap_light = ListedColormap(['#FFFFAA', '#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#FFFF00', '#0000FF', '#FF0000'])

Yq1 = Yq.reshape(xx.shape)
plt.figure(1,figsize = (8,6))
plt.subplot(224)
plt.pcolormesh(xx, yy, Yq1, cmap=cmap_light)
# Mise en graphique des points d<entrainement
plt.scatter(X[0:49, 2], X[0:49, 1], c='#FFFF00', edgecolor='k',
            label = target_name[0])
plt.scatter(X[50:99, 2], X[50:99, 1], c='#0000FF', edgecolor='k',
            label = target_name[1])
plt.scatter(X[100:149, 2], X[100:149, 1], c='#FF0000', edgecolor='k',
            label = target_name[2])
plt.xlabel('Longueur des pétales')
plt.ylabel('Largeur des sépales')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(loc=1,
     fontsize=8)


## Creer un mesh pour faire le trace
Xq = np.matrix([X[:,2],X[:,3]])
Xq = np.matrix.transpose(Xq)
clf = qda.fit(Xq,y)

h = .01       # Espacement du mesh
x_min, x_max = X[:, 2].min() - 1, X[:, 2].max() + 1
y_min, y_max = X[:, 3].min() - 1, X[:, 3].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

Yq = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Mettre le resultat dans un trace couleur
cmap_light = ListedColormap(['#FFFFAA', '#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#FFFF00', '#0000FF', '#FF0000'])

Yq1 = Yq.reshape(xx.shape)
plt.figure(1,figsize = (8,6))
plt.subplot(222)
plt.pcolormesh(xx, yy, Yq1, cmap=cmap_light)
# Mise en graphique des points d<entrainement
plt.scatter(X[0:49, 2], X[0:49, 3], c='#FFFF00', edgecolor='k',
            label = target_name[0])
plt.scatter(X[50:99, 2], X[50:99, 3], c='#0000FF', edgecolor='k',
            label = target_name[1])
plt.scatter(X[100:149, 2], X[100:149, 3], c='#FF0000', edgecolor='k',
            label = target_name[2])
plt.xlabel('Longueur des pétales')
plt.ylabel('Largeur des pétales')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(loc=4,
     fontsize=8)



plt.show()





################ LINEAR DISCRIMINANT ######################################



### Erreur
lda = lda()
lda.fit(iris.data, iris.target)
erreur_lda = 1 - metrics.accuracy_score(lda.predict(iris.data),iris.target)
print('L''erreur pour le discriminant linéaire est %.3f.' % erreur_lda)


## Creer un mesh pour faire le trace
Xq = np.matrix([X[:,0],X[:,3]])
Xq = np.matrix.transpose(Xq)
clf = lda.fit(Xq,y)

h = .01       # Espacement du mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 3].min() - 1, X[:, 3].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

Yq = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Mettre le resultat dans un trace couleur
cmap_light = ListedColormap(['#FFFFAA', '#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#FFFF00', '#0000FF', '#FF0000'])

Yq1 = Yq.reshape(xx.shape)
plt.figure(1,figsize = (8,6))
plt.subplot(221)
plt.pcolormesh(xx, yy, Yq1, cmap=cmap_light)
# Mise en graphique des points d<entrainement
plt.scatter(X[0:49, 0], X[0:49, 3], c='#FFFF00', edgecolor='k',
            label = target_name[0])
plt.scatter(X[50:99, 0], X[50:99, 3], c='#0000FF', edgecolor='k',
            label = target_name[1])
plt.scatter(X[100:149, 0], X[100:149, 3], c='#FF0000', edgecolor='k',
            label = target_name[2])
plt.xlabel('Longueur des sépales')
plt.ylabel('Largeur des pétales')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(loc=4,
     fontsize=8)

## Creer un mesh pour faire le trace
Xq = np.matrix([X[:,0],X[:,1]])
Xq = np.matrix.transpose(Xq)
clf = lda.fit(Xq,y)

h = .01       # Espacement du mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

Yq = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Mettre le resultat dans un trace couleur
cmap_light = ListedColormap(['#FFFFAA', '#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#FFFF00', '#0000FF', '#FF0000'])

Yq1 = Yq.reshape(xx.shape)
plt.figure(1,figsize = (8,6))
plt.subplot(223)
plt.pcolormesh(xx, yy, Yq1, cmap=cmap_light)
# Mise en graphique des points d<entrainement
plt.scatter(X[0:49, 0], X[0:49, 1], c='#FFFF00', edgecolor='k',
            label = target_name[0])
plt.scatter(X[50:99, 0], X[50:99, 1], c='#0000FF', edgecolor='k',
            label = target_name[1])
plt.scatter(X[100:149, 0], X[100:149, 1], c='#FF0000', edgecolor='k',
            label = target_name[2])
plt.xlabel('Longueur des sépales')
plt.ylabel('Largeur des sépales')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(loc=1,
     fontsize=8)

## Creer un mesh pour faire le trace
Xq = np.matrix([X[:,2],X[:,1]])
Xq = np.matrix.transpose(Xq)
clf = lda.fit(Xq,y)

h = .01       # Espacement du mesh
x_min, x_max = X[:, 2].min() - 1, X[:, 2].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

Yq = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Mettre le resultat dans un trace couleur
cmap_light = ListedColormap(['#FFFFAA', '#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#FFFF00', '#0000FF', '#FF0000'])

Yq1 = Yq.reshape(xx.shape)
plt.figure(1,figsize = (8,6))
plt.subplot(224)
plt.pcolormesh(xx, yy, Yq1, cmap=cmap_light)
# Mise en graphique des points d<entrainement
plt.scatter(X[0:49, 2], X[0:49, 1], c='#FFFF00', edgecolor='k',
            label = target_name[0])
plt.scatter(X[50:99, 2], X[50:99, 1], c='#0000FF', edgecolor='k',
            label = target_name[1])
plt.scatter(X[100:149, 2], X[100:149, 1], c='#FF0000', edgecolor='k',
            label = target_name[2])
plt.xlabel('Longueur des pétales')
plt.ylabel('Largeur des sépales')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(loc=1,
     fontsize=8)


## Creer un mesh pour faire le trace
Xq = np.matrix([X[:,2],X[:,3]])
Xq = np.matrix.transpose(Xq)
clf = lda.fit(Xq,y)

h = .01       # Espacement du mesh
x_min, x_max = X[:, 2].min() - 1, X[:, 2].max() + 1
y_min, y_max = X[:, 3].min() - 1, X[:, 3].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

Yq = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Mettre le resultat dans un trace couleur
cmap_light = ListedColormap(['#FFFFAA', '#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#FFFF00', '#0000FF', '#FF0000'])

Yq1 = Yq.reshape(xx.shape)
plt.figure(1,figsize = (8,6))
plt.subplot(222)
plt.pcolormesh(xx, yy, Yq1, cmap=cmap_light)
# Mise en graphique des points d<entrainement
plt.scatter(X[0:49, 2], X[0:49, 3], c='#FFFF00', edgecolor='k',
            label = target_name[0])
plt.scatter(X[50:99, 2], X[50:99, 3], c='#0000FF', edgecolor='k',
            label = target_name[1])
plt.scatter(X[100:149, 2], X[100:149, 3], c='#FF0000', edgecolor='k',
            label = target_name[2])
plt.xlabel('Longueur des pétales')
plt.ylabel('Largeur des pétales')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(loc=4,
     fontsize=8)



plt.show()







################### NAIVE BAYES #################################

### Erreur
gnb = gnb()
gnb.fit(iris.data, iris.target)
erreur_gnb = 1 - metrics.accuracy_score(gnb.predict(iris.data),iris.target)
print('L''erreur pour le discriminant nayésien naif est %.3f.' % erreur_gnb)


## Creer un mesh pour faire le trace
Xq = np.matrix([X[:,0],X[:,3]])
Xq = np.matrix.transpose(Xq)
clf = gnb.fit(Xq,y)

h = .01       # Espacement du mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 3].min() - 1, X[:, 3].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

Yq = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Mettre le resultat dans un trace couleur
cmap_light = ListedColormap(['#FFFFAA', '#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#FFFF00', '#0000FF', '#FF0000'])

Yq1 = Yq.reshape(xx.shape)
plt.figure(1,figsize = (8,6))
plt.subplot(221)
plt.pcolormesh(xx, yy, Yq1, cmap=cmap_light)
# Mise en graphique des points d<entrainement
plt.scatter(X[0:49, 0], X[0:49, 3], c='#FFFF00', edgecolor='k',
            label = target_name[0])
plt.scatter(X[50:99, 0], X[50:99, 3], c='#0000FF', edgecolor='k',
            label = target_name[1])
plt.scatter(X[100:149, 0], X[100:149, 3], c='#FF0000', edgecolor='k',
            label = target_name[2])
plt.xlabel('Longueur des sépales')
plt.ylabel('Largeur des pétales')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(loc=4,
     fontsize=8)

## Creer un mesh pour faire le trace
Xq = np.matrix([X[:,0],X[:,1]])
Xq = np.matrix.transpose(Xq)
clf = gnb.fit(Xq,y)

h = .01       # Espacement du mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

Yq = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Mettre le resultat dans un trace couleur
cmap_light = ListedColormap(['#FFFFAA', '#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#FFFF00', '#0000FF', '#FF0000'])

Yq1 = Yq.reshape(xx.shape)
plt.figure(1,figsize = (8,6))
plt.subplot(223)
plt.pcolormesh(xx, yy, Yq1, cmap=cmap_light)
# Mise en graphique des points d<entrainement
plt.scatter(X[0:49, 0], X[0:49, 1], c='#FFFF00', edgecolor='k',
            label = target_name[0])
plt.scatter(X[50:99, 0], X[50:99, 1], c='#0000FF', edgecolor='k',
            label = target_name[1])
plt.scatter(X[100:149, 0], X[100:149, 1], c='#FF0000', edgecolor='k',
            label = target_name[2])
plt.xlabel('Longueur des sépales')
plt.ylabel('Largeur des sépales')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(loc=1,
     fontsize=8)

## Creer un mesh pour faire le trace
Xq = np.matrix([X[:,2],X[:,1]])
Xq = np.matrix.transpose(Xq)
clf = gnb.fit(Xq,y)

h = .01       # Espacement du mesh
x_min, x_max = X[:, 2].min() - 1, X[:, 2].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

Yq = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Mettre le resultat dans un trace couleur
cmap_light = ListedColormap(['#FFFFAA', '#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#FFFF00', '#0000FF', '#FF0000'])

Yq1 = Yq.reshape(xx.shape)
plt.figure(1,figsize = (8,6))
plt.subplot(224)
plt.pcolormesh(xx, yy, Yq1, cmap=cmap_light)
# Mise en graphique des points d<entrainement
plt.scatter(X[0:49, 2], X[0:49, 1], c='#FFFF00', edgecolor='k',
            label = target_name[0])
plt.scatter(X[50:99, 2], X[50:99, 1], c='#0000FF', edgecolor='k',
            label = target_name[1])
plt.scatter(X[100:149, 2], X[100:149, 1], c='#FF0000', edgecolor='k',
            label = target_name[2])
plt.xlabel('Longueur des pétales')
plt.ylabel('Largeur des sépales')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(loc=1,
     fontsize=8)


## Creer un mesh pour faire le trace
Xq = np.matrix([X[:,2],X[:,3]])
Xq = np.matrix.transpose(Xq)
clf = gnb.fit(Xq,y)

h = .01       # Espacement du mesh
x_min, x_max = X[:, 2].min() - 1, X[:, 2].max() + 1
y_min, y_max = X[:, 3].min() - 1, X[:, 3].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

Yq = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Mettre le resultat dans un trace couleur
cmap_light = ListedColormap(['#FFFFAA', '#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#FFFF00', '#0000FF', '#FF0000'])

Yq1 = Yq.reshape(xx.shape)
plt.figure(1,figsize = (8,6))
plt.subplot(222)
plt.pcolormesh(xx, yy, Yq1, cmap=cmap_light)
# Mise en graphique des points d<entrainement
plt.scatter(X[0:49, 2], X[0:49, 3], c='#FFFF00', edgecolor='k',
            label = target_name[0])
plt.scatter(X[50:99, 2], X[50:99, 3], c='#0000FF', edgecolor='k',
            label = target_name[1])
plt.scatter(X[100:149, 2], X[100:149, 3], c='#FF0000', edgecolor='k',
            label = target_name[2])
plt.xlabel('Longueur des pétales')
plt.ylabel('Largeur des pétales')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(loc=4,
     fontsize=8)



plt.show()










######### nearest centroid###################################

## Erreur
nc = nc()
nc.fit(iris.data, iris.target)
erreur_nc = 1 - metrics.accuracy_score(nc.predict(iris.data),iris.target)
print('L''erreur sur la plus proche moyenne est %.3f.' % erreur_nc)


## Creer un mesh pour faire le trace
Xq = np.matrix([X[:,0],X[:,3]])
Xq = np.matrix.transpose(Xq)
clf = nc.fit(Xq,y)

h = .01       # Espacement du mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 3].min() - 1, X[:, 3].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

Yq = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Mettre le resultat dans un trace couleur
cmap_light = ListedColormap(['#FFFFAA', '#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#FFFF00', '#0000FF', '#FF0000'])

Yq1 = Yq.reshape(xx.shape)
plt.figure(1,figsize = (8,6))
plt.subplot(221)
plt.pcolormesh(xx, yy, Yq1, cmap=cmap_light)
# Mise en graphique des points d<entrainement
plt.scatter(X[0:49, 0], X[0:49, 3], c='#FFFF00', edgecolor='k',
            label = target_name[0])
plt.scatter(X[50:99, 0], X[50:99, 3], c='#0000FF', edgecolor='k',
            label = target_name[1])
plt.scatter(X[100:149, 0], X[100:149, 3], c='#FF0000', edgecolor='k',
            label = target_name[2])
plt.xlabel('Longueur des sépales')
plt.ylabel('Largeur des pétales')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(loc=4,
     fontsize=8)

## Creer un mesh pour faire le trace
Xq = np.matrix([X[:,0],X[:,1]])
Xq = np.matrix.transpose(Xq)
clf = nc.fit(Xq,y)

h = .01       # Espacement du mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

Yq = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Mettre le resultat dans un trace couleur
cmap_light = ListedColormap(['#FFFFAA', '#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#FFFF00', '#0000FF', '#FF0000'])

Yq1 = Yq.reshape(xx.shape)
plt.figure(1,figsize = (8,6))
plt.subplot(223)
plt.pcolormesh(xx, yy, Yq1, cmap=cmap_light)
# Mise en graphique des points d<entrainement
plt.scatter(X[0:49, 0], X[0:49, 1], c='#FFFF00', edgecolor='k',
            label = target_name[0])
plt.scatter(X[50:99, 0], X[50:99, 1], c='#0000FF', edgecolor='k',
            label = target_name[1])
plt.scatter(X[100:149, 0], X[100:149, 1], c='#FF0000', edgecolor='k',
            label = target_name[2])
plt.xlabel('Longueur des sépales')
plt.ylabel('Largeur des sépales')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(loc=1,
     fontsize=8)

## Creer un mesh pour faire le trace
Xq = np.matrix([X[:,2],X[:,1]])
Xq = np.matrix.transpose(Xq)
clf = nc.fit(Xq,y)

h = .01       # Espacement du mesh
x_min, x_max = X[:, 2].min() - 1, X[:, 2].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

Yq = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Mettre le resultat dans un trace couleur
cmap_light = ListedColormap(['#FFFFAA', '#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#FFFF00', '#0000FF', '#FF0000'])

Yq1 = Yq.reshape(xx.shape)
plt.figure(1,figsize = (8,6))
plt.subplot(224)
plt.pcolormesh(xx, yy, Yq1, cmap=cmap_light)
# Mise en graphique des points d<entrainement
plt.scatter(X[0:49, 2], X[0:49, 1], c='#FFFF00', edgecolor='k',
            label = target_name[0])
plt.scatter(X[50:99, 2], X[50:99, 1], c='#0000FF', edgecolor='k',
            label = target_name[1])
plt.scatter(X[100:149, 2], X[100:149, 1], c='#FF0000', edgecolor='k',
            label = target_name[2])
plt.xlabel('Longueur des pétales')
plt.ylabel('Largeur des sépales')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(loc=1,
     fontsize=8)


## Creer un mesh pour faire le trace
Xq = np.matrix([X[:,2],X[:,3]])
Xq = np.matrix.transpose(Xq)
clf = nc.fit(Xq,y)

h = .01       # Espacement du mesh
x_min, x_max = X[:, 2].min() - 1, X[:, 2].max() + 1
y_min, y_max = X[:, 3].min() - 1, X[:, 3].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

Yq = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Mettre le resultat dans un trace couleur
cmap_light = ListedColormap(['#FFFFAA', '#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#FFFF00', '#0000FF', '#FF0000'])

Yq1 = Yq.reshape(xx.shape)
plt.figure(1,figsize = (8,6))
plt.subplot(222)
plt.pcolormesh(xx, yy, Yq1, cmap=cmap_light)
# Mise en graphique des points d<entrainement
plt.scatter(X[0:49, 2], X[0:49, 3], c='#FFFF00', edgecolor='k',
            label = target_name[0])
plt.scatter(X[50:99, 2], X[50:99, 3], c='#0000FF', edgecolor='k',
            label = target_name[1])
plt.scatter(X[100:149, 2], X[100:149, 3], c='#FF0000', edgecolor='k',
            label = target_name[2])
plt.xlabel('Longueur des pétales')
plt.ylabel('Largeur des pétales')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(loc=4,
     fontsize=8)



plt.show()