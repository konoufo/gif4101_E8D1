import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics.cluster import adjusted_rand_score as rand
from sklearn.metrics.cluster import adjusted_mutual_info_score as mutual
from sklearn.metrics.cluster import v_measure_score as vscore


data1 = np.loadtxt("data.txt", delimiter=',')[:, :100]
target1 = np.loadtxt("target.txt", delimiter=',')


class Question2c:
    em = None

    def compute_labels(self):
        """Calcule les P(C_i | x) en partant du clustering obtenu par l'instance `em` et
        en deduire l'etiquette
        Parameters:
            em (EM): instance d'algorithme EM complete
        Return:
            labels (ndarray): etiquette obtenue pour chaque donnee
        """
        em = self.em
        classes = np.unique(target1)
        pcX = np.zeros([target1.shape[0], len(classes)])
        for c in classes:
            # les P(Ci | Gj)
            pcg = np.sum(em.posteriors_[target1 == c], axis=0) / np.sum(em.posteriors_, axis=0)
            for t in range(data1.shape[0]):
                pcX[t, c] = np.sum(pcg * em.posteriors_[t, :])
        for t in range(data1.shape[0]):
            somme = np.sum(pcX[t, :])
            pcX[t, :] /= somme
            print('somme P(Ci|x): {}'.format(somme))
        labels = np.argmax(pcX, axis=1)
        return labels

    def run(self, EM):
        kgroups = np.arange(2, 52, 2)
        k = 0
        rand_gene = np.empty(len(kgroups))
        mutual_gene = np.empty(len(kgroups))
        v_gene = np.empty(len(kgroups))

        for i in kgroups:
            em = EM(n_clusters=i, X=data1[:, :100]).fit()
            self.em = em
            labk_eti = self.compute_labels()
            print('Labels:\n {}'.format(labk_eti))
            rand_score = rand(labk_eti, target1)
            mutual_score = mutual(labk_eti, target1)
            v_score = vscore(labk_eti, target1)
            rand_gene[k] = rand_score
            mutual_gene[k] = mutual_score
            v_gene[k] = v_score
            k = k + 1

        plt.plot(kgroups, rand_gene, 'r--', label = 'rand')
        plt.plot(kgroups, mutual_gene, 'g--', label = 'info mutuelle')
        plt.plot(kgroups, v_gene, 'b--', label = 'mesure V')
        plt.legend(loc=2, fontsize=8)
        plt.xlabel('Nombres de groupes')
        plt.ylabel('Performance de EM')
        plt.show()


if __name__ == '__main__':
    from q1.em import EM
    Question2c().run(EM)
