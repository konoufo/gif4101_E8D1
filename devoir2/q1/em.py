import numpy as np


class EM:
    """Algorithme EM
    Exécute et conserve les paramètres finaux de l'algorithme EM avec K composantes.
    Les probabilités a priori des observations suivent des lois de Bernoulli multivariées.

    Attributs:
        X (ndarray): (n_samples, n_dim) matrice des donnees
        n_composants (int): nombre de groupes à paramétrer pour l'algorithme EM
        pi (vecteur): (1, n_composants) vecteur des proportions de groupe P(G_j)
        pr (ndarray): (n_composants, n_dim) matrice des parametres de loi de Bernoulli  (x_i|G_j) ~ B(p_j,i)
        posteriors (ndarray): (n_samples, n_composants) matrice des probabilites a posteriori P(G_j|x)
    """
    def __init__(self, X=None, n_clusters=4):
        self.n_composants = n_clusters
        if X is not None:
            self.initialize(X)

    def initialize(self, X, n_clusters=None):
        self.X = np.array(X)
        self.n_samples, self.n_dim = X.shape[0], X.shape[1]
        if n_clusters is not None:
            self.n_composants = n_clusters
        # vecteur des P(G_j)
        self.pi = np.ones([1, self.n_composants], dtype=np.float64) / self.n_composants
        # matrice des paramètres de loi de Bernoulli.
        # L'élément (j,i) correspond au paramètre de loi (x_i|G_j) ~ B(p_j,i)
        self.pr = np.random.rand(self.n_composants, self.n_dim)
        self.pr[self.pr < 1e-6] = 1e-6
        self._is_fitted = False

    def _m_step(self):
        posteriors_sum = np.sum(self.posteriors_, axis=0)
        print('post sum', posteriors_sum[0])
        self.pi = np.reshape(posteriors_sum, (1, self.n_composants)) / self.n_samples
        self.pi[0, -1] = 1.0 - np.sum(self.pi[0, :-1])
        print('somme des pi: {}'.format(np.sum(self.pi)))

        for j in range(self.n_composants):
            for i in range(self.n_dim):
                self.pr[j, i] = np.sum(self.posteriors_[:, j] * self.X[:, i]) / posteriors_sum[j]
        # correction numérique
        self.pi[self.pi < 1e-6] = 1e-6
        self.pr[self.pr < 1e-6] = 1e-6
        self.pr[np.isnan(self.pr)] = 1e-6
        self.pi[np.isnan(self.pi)] = 1e-6

    def _e_step(self):
        self.posteriors_ = self._compute_posteriors()

    def _compute_posteriors(self, X=None):
        X = X if X is not None else self.X
        posteriors = np.zeros([X.shape[0], self.n_composants])
        for t in range(X.shape[0]):
            for j in range(self.n_composants):
                produit = self.pi[0, j]
                for i in range(X.shape[1]):
                    produit *= self.pr[j, i]**(X[t, i]) * (1 - self.pr[j, i])**(1 - X[t, i])
                posteriors[t, j] = produit
            posteriors[t, :] = posteriors[t, :] / np.sum(posteriors[t, :])
        return posteriors

    def fit(self, X=None, n_clusters=None):
        if X is not None:
            self.initialize(X, n_clusters)
        if self.X is None:
            raise AssertionError('Vous devez passer le jeu de donnees en argument de fit() ou dans le constructeur.')
        if self.n_composants is None:
            raise AssertionError('Vous devez passer le nb de clusters en argument de fit() ou dans le constructeur.')
        epsilon = 1.0
        for k in range(100):
            if epsilon <= 1e-6:
                break
            pi_initial = np.copy(self.pi)
            pr_initial = np.copy(self.pr)
            self._e_step()
            self._m_step()
            epsilon = np.min(np.concatenate((np.abs(self.pi - pi_initial).T, np.abs(self.pr - pr_initial)), axis=1))
            print('\niteration: {}'.format(k))
            print('epsilon: {}'.format(epsilon))
        self._is_fitted = True
        return self

    def _check_fitted(self):
        if not self._is_fitted:
            raise AssertionError('Executer fit() d\'abord pour entrainer.')

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        self._check_fitted()
        return self._compute_posteriors(X)


    def score(self, X):
        """ Calcule et retourne le maximum de l'espérance de vraisemblance de la paramétrisation.
        """
        self._check_fitted()
        # Pour rendre le calcul plus simple grâce au produit matriciel P -> (n_dim, n_composants)
        P = self.pr.T
        somme_interne =  np.add(np.log(self.pi), np.dot(X, np.log(P)) + np.dot(1 - X, np.log(1 - P)))
        phi = np.sum(self.predict_proba(X) * somme_interne)
        return phi
