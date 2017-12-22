import random
import functools
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import MiniBatchKMeans as KMeans


def memoize(f):
    """ Décorateur de fonction pour memoization. """
    class memodict(dict):
        def __init__(self, f):
            self.f = f
        def __call__(self, *args):
            return self[args]
        def __missing__(self, key):
            ret = self[key] = self.f(*key)
            return ret
    return memodict(f)


@memoize
def get_ak(k, Nd):
    if k == 2:
        return( 1 - 3.0 / (4.0 * Nd ) )
    else:
        previous_a = get_ak(k-1, Nd)
        return ( previous_a + (1.0-previous_a)/6.0 ) 


class DetK:
    def __init__(self, X, B=10):
        self.X = X
        self.B = B
        self.mu = None
        self.classement = None
        
    def find_centers(self, k):
        km = KMeans(n_clusters=k)
        self.classement = km.fit_predict(self.X)
        self.mu = km.cluster_centers_
            
    def fK(self, thisk, Skm1=0):
        X = self.X
        Nd = X.shape[0]
        self.find_centers(thisk)
        mu, clusters = self.mu, self.classement
        Sk = sum([np.sum(np.linalg.norm(mu[i]-X[clusters == i], axis=1)**2) for i in range(thisk)])
        if thisk == 1:
            fs = 1
        elif Skm1 == 0:
            fs = 1
        else:
            fs = Sk/(get_ak(thisk, Nd)*Skm1)
        return fs, Sk  
        
    def _bounding_box(self):
        X = self.X
        return list(zip(np.min(X, axis=0), np.max(X, axis=0)))
         
    def gap(self, thisk):
        X = self.X
        bbox = self._bounding_box()
        self.find_centers(thisk)
        mu, clusters = self.mu, self.classement
        Wk = sum([np.sum(np.linalg.norm(mu[i]-X[clusters == i], axis=1)**2)/(2*X[clusters == i].shape[0])
                  for i in range(thisk)])
        Wk = np.log(Wk)
        # Create B reference datasets
        B = self.B
        BWkbs = np.zeros(B)
        random_uniform = functools.partial(np.random.uniform, size=X.shape[0])
        for i in range(B):
            Xb = np.array([feature for feature in itertools.starmap(random_uniform, bbox)])
            Xb = Xb.T
            assert Xb.shape == X.shape
            # for n in range(len(X)):
                # Xb.append([random.uniform(xmin,xmax), \
                          # random.uniform(ymin,ymax)])
            # Xb = np.array(Xb)
            kb = KMeans(n_clusters=thisk)
            clusters = kb.fit_predict(Xb)
            mu = kb.cluster_centers_
            BWkbs[i] = sum([np.sum(np.linalg.norm(mu[j]-Xb[clusters == j], axis=1)**2)/(2*Xb[clusters == j].shape[0])
                            for j in range(thisk)])
        Wkb = sum(BWkbs)/B
        sk = np.sqrt(sum((BWkbs-Wkb)**2)/float(B))*np.sqrt(1+1/B)
        return Wk, Wkb, sk
     
    def run(self, maxk, which='both'):
        ks = range(1,maxk)
        fs = np.zeros(len(ks))
        Wks,Wkbs,sks = np.zeros(len(ks)+1),np.zeros(len(ks)+1),np.zeros(len(ks)+1)
        # Special case K=1
        if which == 'f':
            fs[0], Sk = self.fK(1)
        elif which == 'gap':
            Wks[0], Wkbs[0], sks[0] = self.gap(1)
        else:
            fs[0], Sk = self.fK(1)
            Wks[0], Wkbs[0], sks[0] = self.gap(1)
        # Rest of Ks
        for k in ks[1:]:
            if which == 'f':
                fs[k-1], Sk = self.fK(k, Skm1=Sk)
            elif which == 'gap':
                Wks[k-1], Wkbs[k-1], sks[k-1] = self.gap(k)
            else:
                fs[k-1], Sk = self.fK(k, Skm1=Sk)
                Wks[k-1], Wkbs[k-1], sks[k-1] = self.gap(k)
        if which == 'f':
            self.fs = fs
        elif which == 'gap':
            G = []
            for i in range(len(ks)):
                G.append((Wkbs-Wks)[i] - ((Wkbs-Wks)[i+1]-sks[i+1]))
            self.G = np.array(G)
        else:
            self.fs = fs
            G = []
            for i in range(len(ks)):
                G.append((Wkbs-Wks)[i] - ((Wkbs-Wks)[i+1]-sks[i+1]))
            self.G = np.array(G)
     
    def plot_all(self):
        X = self.X
        ks = range(1, len(self.fs)+1)
        fig = plt.figure(figsize=(18,5))
        # Plot 1
        # ax1 = fig.add_subplot(131)
        # ax1.set_xlim(-1,1)
        # ax1.set_ylim(-1,1)
        # ax1.plot(zip(*X)[0], zip(*X)[1], '.', alpha=0.5)
        # tit1 = 'N=%s' % (str(len(X)))
        # ax1.set_title(tit1, fontsize=16)
        # Plot 2
        ax2 = fig.add_subplot(131)
        ax2.set_ylim(0, 1.25)
        ax2.plot(ks, self.fs, 'ro-', alpha=0.6)
        ax2.set_xlabel('Number of clusters K', fontsize=16)
        ax2.set_ylabel('f(K)', fontsize=16) 
        foundfK = np.where(self.fs == min(self.fs))[0][0] + 1
        tit2 = 'f(K) finds %s clusters' % (foundfK)
        ax2.set_title(tit2, fontsize=16)
        # Plot 3
        ax3 = fig.add_subplot(133)
        ax3.bar(ks, self.G, alpha=0.5, color='g', align='center')
        ax3.set_xlabel('Number of clusters K', fontsize=16)
        ax3.set_ylabel('Gap', fontsize=16)
        foundG = np.where(self.G > 0)[0][0] + 1
        tit3 = 'Gap statistic finds %s clusters' % (foundG)
        ax3.set_title(tit3, fontsize=16)
        ax3.xaxis.set_ticks(range(1,len(ks)+1))
        plt.savefig('detK_N{}.png'.format(X.shape[0]), bbox_inches='tight', dpi=100)
        plt.show()