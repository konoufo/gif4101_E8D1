from os import path
import numpy as np
import tensorflow as tf

from kmeans import DetK
from autoencoder import encoder_op, X

# importer données traitées
root = path.join(path.dirname(path.abspath(__file__)), 'output')


def transformer(data, identifier):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, join(root, 'autoencodeur_{}.ckpt'.format(identifier)))
        return encoder_op.eval({X: data})
    
    
def generate_clusters(data, identifier):
    dt = transformer(data[:, 3:])
    clusters = KMeans(k_optimal[identifier]).fit_predict(dt)
    clusters = np.rec.fromarrays(data[:, 0], np.c_[clusters])
    np.savetxt('clustering_{}'.format(identifier), clusters)
    return clusters


def load_data(filename):
    return np.genfromtxt(filename, delimiter='\t')
    
    
if __name__ == '__main__':
    X = load_data(path.join(root, '..', 'parameters_STED640.csv'))
    X = transformer(X, 'STED640')
    print('Data:', X)
    determiner_k = DetK(X)
    print('Bounding box:', determiner_k._bounding_box())
    determiner_k.run(5)
    determiner_k.plot_all()
    # k_optimal = {'STD640': 3}
    # for k, v in k_optimal.items():
        # data = load_data('parameters_{}.csv'.format(k))
        # print('Jeu {}:\n'.format(identifier), generate_clusters(data, k))
