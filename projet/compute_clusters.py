from os import path
import pickle
import numpy as np
import tensorflow as tf
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn import metrics

from kmeans import DetK
from autoencoder import encoder_op, X as tfX


# importer données traitées
root = path.join(path.dirname(path.abspath(__file__)), 'output')


def transformer(data, identifier):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, path.join(root, 'autoencodeur_{}.ckpt'.format(identifier)))
        return encoder_op.eval({tfX: data})
    
    
def generate_clusters(data, identifier):
    dt = transformer(data[:, 4:].astype(np.float64), identifier)
    km = KMeans(k_optimal[identifier])
    clusters = km.fit_predict(dt).astype(int)
    print('\nShape:\n', np.c_[clusters].shape, data.shape, '\n')
    dump = np.concatenate((np.c_[clusters], data), axis=1)
    np.savetxt(path.join(root, 'clustering_{}.csv'.format(identifier)), dump, 
               fmt='%s', delimiter='\t')
    return clusters, km.cluster_centers_


def load_data(filename):
    return np.genfromtxt(filename, delimiter='\t', dtype='U')
    
    
if __name__ == '__main__':
    # Sauver le générateur aléatoire
    
    with open('compute_clusters_rs.state', 'wb') as f:
                pickle.dump(np.random.get_state(), f)
    
    # identifier = 'STED561'
    # X = load_data(path.join(root, '..', 'parameters_{}.csv'.format(identifier)))
    # X = transformer(X[:, 4:], identifier)
    # print('Data:', X)
    # determiner_k = DetK(X)
    # print('Bounding box:', determiner_k._bounding_box())
    # determiner_k.run(5)
    # determiner_k.plot_all()
    k_optimal = {'STED640': 2, 'STED561': 3}
    for key, v in k_optimal.items():
        data = load_data('parameters_{}.csv'.format(key))
        print('\nJeu {}:\n'.format(key), 'Data:', data.shape)
        classement, centres = generate_clusters(data, key)
        score = metrics.calinski_harabaz_score(transformer(data[:, 4:].astype(float), key), classement)
        print(*['Clus.{}: {:.3f}\n'.format(k, classement[classement==k].shape[0]/classement.shape[0]) 
               for k in np.unique(classement)])
        print('Calinski-Harabaz Score: {}'.format(score), '\n')       
        
        