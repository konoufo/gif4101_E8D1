import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Importer les données
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Paramètres d'entraînement
taux_apprentissage = 0.01
num_steps = 30000
batch_size = 256

periode_logging = 1000

# Paramètres du réseau de neurone
# nombre de neurones dans les couches cachées
num_hidden_1 = 256
num_hidden_2 = 128
num_input = 784 # MNIST data input (img shape: 28*28)

# Définition des données pour le graphe computationnel
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Générer le modèle computationnel
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prédiction
y_pred = decoder_op
# On construit un autoencodeur donc l'objectif est d'imiter les données
y_true = X

# Definir fonction de perte
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(taux_apprentissage).minimize(loss)

# Initialiser les variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # Entraînement
    for i in range(1, num_steps+1):
        # Générer prochain batch
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Backpropagation et coût pour calculer la fonction de perte
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % periode_logging == 0 or i == 1:
            print('Étape %i: Perte du Minibatch: %f' % (i, l))
