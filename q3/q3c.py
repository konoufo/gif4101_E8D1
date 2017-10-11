from sklearn import datasets
from sklearn import metrics
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Load the iris data
iris = datasets.load_iris()

# Create a variable for the feature data
X = iris.data

# Create a variable for the target data
y = iris.target

#########  erreur sur jeu d'entrainement   #############################
qda = QuadraticDiscriminantAnalysis()
qda.fit(iris.data, iris.target)
erreur_qda_jeu = 1 - metrics.accuracy_score(qda.predict(iris.data), iris.target)

############### erreur avec partition aléatoire 50/50 ######################

# Random split the data into 2 new datasets.
erreur_p = np.zeros(10)

for i in range(0, 10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    qda.fit(X_train, y_train)
    erreur_partition = metrics.accuracy_score(qda.predict(X_test), y_test)
    erreur_p[i] = erreur_partition

mean = 1 - erreur_p.mean()
ecart = (1 - erreur_p).std() / 2

###### erreur avec k-fold cross validation ###############################
from sklearn.model_selection import cross_val_score, StratifiedKFold

erreur_k = np.zeros(10)
for i in range(0, 10):
    kf = StratifiedKFold(n_splits=3, shuffle=True)
    scores = cross_val_score(qda, iris.data, iris.target, cv=kf)
    scoresm = np.mean(scores)
    erreur_k[i] = scoresm

meank = 1 - np.mean(erreur_k)
ecartk = np.std(1 - erreur_k) / 2

print('L\'erreur sur l\'entrainement est %.3f.' % erreur_qda_jeu)
print('L\'erreur sur la partition est %5.3f plus ou moins %.3f.' % (mean, ecart))
print('L\'erreur sur la validation croisée est %5.3f plus ou moins %.3f.' %
      (meank, ecartk))