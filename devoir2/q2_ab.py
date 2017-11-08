import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics.cluster import adjusted_rand_score as rand
from sklearn.metrics.cluster import adjusted_mutual_info_score as mutual
from sklearn.metrics.cluster import v_measure_score as vscore

##### Importer les données du problemes  ############################
features = open("features.txt","r")


data1 = np.loadtxt("data.txt", delimiter=',')
target1 = np.loadtxt("target.txt", delimiter=',')
features1 = features.read().split()



########  Clustering pour K=5, question 2a ####################
class Question2a:
    def run(self):
        kmeans = KMeans(n_clusters=5, init = 'k-means++').fit(data1)
        labels = kmeans.labels_

        ###### Recherche des mots les plus fréquents pour chaque groupe #######
        for k in range (0,5):
           mword = []
           for i in range (0,4327):
               if labels[i] == k:
                   word1 = data1[i,:]
                   for j in range(0,1000):
                      if word1[j] == 1:
                           mword.append(features1[j])
           c = Counter(mword)
           print(c.most_common(5))
            
 
            
######## Clustering K = 2..50, question 2b ###############
class Question2b:
    def run(self):
        kgroups = np.arange(2,52,2)
        labels = np.empty([4327,0])
        k=0
        rand_gene = np.empty(len(kgroups))
        mutual_gene = np.empty(len(kgroups))
        v_gene = np.empty(len(kgroups))

        for i in kgroups:           #procede a Kmeans pour different cluster
            kmeans = KMeans(n_clusters=i, init = 'k-means++').fit(data1)
            lab = kmeans.labels_
            labk_eti = np.empty([0,2])
            for j in range(0,i):  #évalue et assigne la classe pour chaque groupe parmis les K groupes
                labels = np.append(lab,target1)
                labels = np.reshape(labels,(2,4327)).T
                labk = labels[labels[:,0]==j, :]
                b = np.asarray(Counter(labk[:,1]).most_common(1))
                labk[:,0] = b[:,0]
                labk_eti = np.concatenate([labk_eti,labk])
            rand_score = rand(labk_eti[:,0],labk_eti[:,1])
            mutual_score = mutual(labk_eti[:,0],labk_eti[:,1])
            v_score = vscore(labk_eti[:,0],labk_eti[:,1])
            rand_gene[k] = rand_score
            mutual_gene[k] = mutual_score
            v_gene[k] = v_score
            k = k+1


        plt.plot(kgroups, rand_gene,'r--',label = 'rand' )
        plt.plot(kgroups, mutual_gene,'g--',label = 'info mutuelle' )
        plt.plot(kgroups, v_gene,'b--',label = 'mesure V' )
        plt.legend(loc=2,
             fontsize=8)
        plt.xlabel('Nombres de groupes')
        plt.ylabel('Performance')
        plt.show()
        
        
if __name__=='__main__':
    # Question2a().run()
    Question2b().run()