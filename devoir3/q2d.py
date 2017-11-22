from sklearn import preprocessing
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.model_selection import cross_val_score

from utils import Testeur
from q2c import OneVSAll as GradientDescentMulti


clfs = [GradientDescentMulti, LinearDiscriminantAnalysis, Perceptron, LogisticRegression]


def report_performances(dataset):
    X = preprocessing.minmax_scale(dataset.data)
    iris_test = Testeur(X, dataset.target)
    test_errors = []
    train_errors = []
    for clf in clfs:
        iris_test.compute_cross_val(k=3, clfactory=clf)
        test_errors.append(1 - iris_test.scores[:, 1].ravel())
        train_errors.append(1 - iris_test.scores[:, 0].ravel())
    print(''.join(['{}:: erreur_train = {}; erreur_test = {}\n'.format(clfs[i].__name__, train_errors[i], err)
                   for i, err in enumerate(test_errors)]))


if __name__=='__main__':
    iris = load_iris()
    cancer = load_breast_cancer()
    deco = ''.join(['-'] * 9)
    print('{} Iris dataset {}'.format(deco, deco))
    report_performances(iris)
    print('\n{} Breast cancer dataset {}'.format(deco, deco))
    report_performances(cancer)
