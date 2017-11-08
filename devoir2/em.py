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
        self._is_fitted = False


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


    def fit(self, X=None, n_clusters=None):
        if X is not None:
            self.initialize(X, n_clusters)
        if self.X is None:
            raise AssertionError('Vous devez passer le jeu de donnees en argument de fit() ou dans le constructeur.')
        if self.n_composants is None:
            raise AssertionError('Vous devez passer le nb de clusters en argument de fit() ou dans le constructeur.')
        epsilon = 1.0
        while epsilon > 1e-6:
            pi_initial = np.copy(self.pi)
            pr_initial = np.copy(self.pr)
            self._e_step()
            self._m_step()
            epsilon = np.min(np.concatenate((np.abs(self.pi - pi_initial).T, np.abs(self.pr - pr_initial)), axis=1))
        self._is_fitted = True
        return self

    def _should_fit(self, X=None, n_clusters=None):
        return X is not None or n_clusters is not None or not self._is_fitted

    def predict(self, X=None, n_clusters=None):
        if self._should_fit(X, n_clusters):
            return self.fit(X, n_clusters).predict()
        return np.argmax(self.posteriors, axis=1)

    def score(self, X=None, n_clusters=None):
        """ Calcule et retourne le maximum de l'espérance de vraisemblance de la paramétrisation.
        On utilise les propriétés mathématiques de multiplication de matrice d'une part,
        et les multiplications element par element de Numpy d'autre part, pour raccourcir le calcul des sommes.
        """
        if self._should_fit(X, n_clusters):
            return self.fit(X, n_clusters).score()
        X = self.X
        # pour rendre le calcul plus simple grâce au produit matriciel P -> (n_dim, n_composants)
        P = self.pr.T
        somme_interne =  np.add(self.pi, X * np.log(P) + (1 - X) * np.log(1 - P))
        phi = np.sum(np.multiply(self.posteriors, somme_interne))
        return phi
