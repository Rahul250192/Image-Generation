import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#Hyperparameters
learning_rate = 0.001
num_steps = 300
batch_size = 64

#Network
image_dim = 784 # MNIST images are 28x28 pixels
hidden_dim = 512
latent_dim = 2

#using Xavier Glorot initialization
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# Weights and Biases
weights = {
    'encoder_h1': tf.Variable(glorot_init([image_dim, hidden_dim])),
    'mean': tf.Variable(glorot_init([hidden_dim, latent_dim])),
    'std': tf.Variable(glorot_init([hidden_dim, latent_dim])),
    'decoder_h1': tf.Variable(glorot_init([latent_dim, hidden_dim])),
    'decoder_out': tf.Variable(glorot_init([hidden_dim, image_dim]))
}
biases = {
    'encoder_b1': tf.Variable(glorot_init([hidden_dim])),
    'mean': tf.Variable(glorot_init([latent_dim])),
    'std': tf.Variable(glorot_init([latent_dim])),
    'decoder_b1': tf.Variable(glorot_init([hidden_dim])),
    'decoder_out': tf.Variable(glorot_init([image_dim]))
}

#### Building the Encoder ####
input_image = tf.placeholder(tf.float32, shape=[None, image_dim])
encoder = tf.matmul(input_image, weights['encoder_h1']) + biases['encoder_b1']
encoder = tf.nn.tanh(encoder)
mean = tf.matmul(encoder, weights['mean']) + biases['mean']
std = tf.matmul(encoder, weights['std']) + biases['std']

# Distribution
dis = tf.random_normal(tf.shape(std), dtype=tf.float32, mean=0., stddev=1.0,
                       name='epsilon')

z = mean + tf.exp(std / 2) * dis

#### Decoder####
decoder = tf.matmul(z, weights['decoder_h1']) + biases['decoder_b1']
decoder = tf.nn.tanh(decoder)
decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
decoder = tf.nn.sigmoid(decoder)

