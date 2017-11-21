import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

from q2b import GradientDescent

def convert_target(y):
    labels = np.unique(y)
    y[y == labels[1]] = -1
    y[y == labels[0]] = 1
    print('labels: {}'.format(np.unique(y)))
    return y

if __name__ == '__main__':
    gd = GradientDescent()
    sgd = SGDClassifier(loss='squared_loss')
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1,
                               n_classes=2)
    y = convert_target(y)
    random_slice = np.arange(100)
    np.random.shuffle(random_slice)
    trainX = X[random_slice[:50], :]
    trainR = y[random_slice[:50]]
    testX = X[random_slice[50:], :]
    testR = y[random_slice[50:]]
    sgd.fit(trainX, trainR)
    gd.fit(trainX, trainR)
    print('Score (scikit): {:.3f}'.format(sgd.score(testX, testR)))
    print('Score 1 (us): {:.3f}'.format(gd.score(testX, testR)))
    # X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1,
    #                            n_classes=3)
    # y = convert_target(y)
    # gd.fit(X, y)
    # print('Score 2: {:.3f}'.format(gd.score(X, y)))
