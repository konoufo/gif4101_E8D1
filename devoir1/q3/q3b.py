import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.neighbors import NearestCentroid as nc

from .utils import Testeur, Traceur


def compute_errors_for_classifiers(classifiers=(qda, lda, gnb, nc)):
    names = ['discriminant quadratique', 'discriminant linéaire', 'classifieur bayesien naif',
             'classifieur de plus proche moyenne']
    for i, clf in enumerate(classifiers):
        print('L\'erreur pour le {} est {:0.3f}.'.format(names[i], Testeur(clf).compute_error()))


def generate_the_plots(classifiers=(qda, lda, gnb, nc)):
    for i, clf in enumerate(classifiers):
        Traceur(fig_index=i + 1).tracer_tous_les_duos(clf, show=False)
    plt.show()


if __name__ == '__main__':
    compute_errors_for_classifiers()
    generate_the_plots()
