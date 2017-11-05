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
    def __init__(self, X, n_clusters=4):
        self.initialize(X, n_clusters)

    def initialize(self, X, n_clusters):
        self.X = X
        self.n_samples, self.n_dim = X.shape[0], X.shape[1]
        self.n_composants = n_clusters
        # vecteur des P(G_j)
        self.pi = np.ones([1, n_clusters], dtype=np.float64) / n_clusters
        # matrice des paramètres de loi de Bernoulli.
        # L'élément (j,i) correspond au paramètre de loi (x_i|G_j) ~ B(p_j,i)
        self.pr = np.random.rand(n_clusters, self.n_dim)


    def _m_step(self):
        posteriors_sum = np.sum(self.posteriors, axis=0)
        self.pi = posteriors_sum / self.n_samples
        for j in range(self.n_composants):
            self.pr[j, :] = np.sum(np.multiply(self.posteriors[:, j], self.X), axis=0) / posteriors_sum
        # correction numérique
        self.pi[self.pi < 1e-6] = 1e-6
        self.pr[self.pr < 1e-6] = 1e-6

    def _e_step(self):
        try:
            posteriors = self.posteriors
        except AttributeError:
            self.posteriors = posteriors = np.zeros([self.n_samples, self.n_composants])
        for t in range(self.n_samples):
            produit1 = np.power(self.pr, self.X[t, :])
            produit2 = np.power(1 - self.pr, 1 - self.X[t, :])
            produit = np.prod(np.multiply(produit1, produit2), axis=1)
            numerator = np.multiply(self.pi, produit)
            denominator = np.sum(numerator)
            print('denominator: {}'.format(denominator))
            posteriors[t, :] = np.divide(numerator, denominator)


    def run(self):
        epsilon = 1.0
        while epsilon > 1e-6:
            pi_initial = np.copy(self.pi)
            pr_initial = np.copy(self.pr)
            self._e_step()
            self._m_step()
            epsilon = np.min(np.concatenate((np.abs(self.pi - pi_initial).T, np.abs(self.pr - pr_initial)), axis=1))
        return self.pi, self.pr
