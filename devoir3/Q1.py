import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm
from sklearn.neighbors import KernelDensity


# Loi de probabilite sous-jacente
def pdf(X):
    return 0.4 * norm(0,1).pdf(X[:]) + 0.6 * norm(5,1).pdf(X[:])


def sample(n):
    x = norm(0,1).rvs(n)
    # Simulation a partir d'une loi normal
    x[int(0.4 * n):] += 5
    return x

# a)

# Tracer des histogrammes (n=50)
plt.hist(sample(50),bins=25)
plt.title('Histogramme de X pour N=50 avec 25 bins')
plt.show()
# Tracer des histogrammes (n=10000)
plt.hist(sample(10000),bins=25)
plt.title('Histogramme de X pour N=10000 avec 25 bins')
plt.show()


# b)
# Pour N = 50
bandwidths = [0.3,1,5]
fig, ax = plt.subplots()
xs = sample(50)

for bandwidth in bandwidths:
    # tracez des estimationde densite en fct du bandwidth
    ax.plot( np.linspace(-5, 10, 50), np.exp(KernelDensity(bandwidth=bandwidth,kernel='tophat').fit(xs[:, np.newaxis]).score_samples(np.linspace(-5, 10, 50)[:,np.newaxis])),
            label=bandwidth)
# Tracez de la distribution sous-jacente
ax.fill(np.linspace(-5, 10, 50), pdf(np.linspace(-5, 10, 50)), fc='grey')
ax.set_xlim(-5, 10)
ax.legend(loc='upper right')
plt.title('Estimation du noyau par top-hat pour b = {0.3, 1, 5} et pour N=50')
plt.show()

# Pour N = 10 000
bandwidths = [0.3,1,5]
fig, ax = plt.subplots()
xs = sample(10000)

for bandwidth in bandwidths:
    # tracez des estimationde densite en fct du bandwidth
    ax.plot( np.linspace(-5, 10, 10000), np.exp(KernelDensity(bandwidth=bandwidth,kernel='tophat').fit(xs[:, np.newaxis]).score_samples(np.linspace(-5, 10, 10000)[:,np.newaxis])),
            label=bandwidth)
# Tracez de la distribution sous-jacente
ax.fill(np.linspace(-5, 10, 10000), pdf(np.linspace(-5, 10, 10000)),fc='grey')
ax.set_xlim(-5, 10)
ax.legend(loc='upper right')
plt.title('Estimation du noyau par top-hat pour b = {0.3, 1, 5} et pour N=10000')
plt.show()