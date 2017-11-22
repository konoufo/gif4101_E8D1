import numpy as np
from sklearn.base import is_classifier, clone
from sklearn.utils import indexable
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _fit_and_score


def cross_val_score(estimator, X, y=None, groups=None, scoring=None, cv=None,
                    n_jobs=1, verbose=0, fit_params=None,
                    pre_dispatch='2*n_jobs'):
    """Une copie quasi-verbatim de la méthode de scikit-learn pour cross-validation.
    Aux exception près, que celle-ci retourne les scores pour l'ensemble d'entrainement également
    ainsi que le temps de calcul pour les évaluations de score."""
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    scores = parallel(delayed(_fit_and_score)(clone(estimator), X, y, scorer,
                                              train, test, verbose, None,
                                              fit_params, return_train_score=True)
                      for train, test in cv.split(X, y, groups))
    return np.array(scores)
