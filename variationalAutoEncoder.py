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

# variational autoencoder loss #####
def va_loss(x_reconstruct, x_true):
    loss = x_true * tf.log(1e-10 + x_reconstruct) + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstruct)
    loss = -tf.reduce_sum(loss, 1)

    kl_div_loss = 1 + std - tf.square(mean) - tf.exp(std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(loss + kl_div_loss)

loss_op = va_loss(decoder, input_image)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

### Training####
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for i in range(1, num_steps+1):
        batch_x, _ = mnist.train.next_batch(batch_size)
        feed_dict = {input_image: batch_x}
        _, l = sess.run([train_op, loss_op], feed_dict=feed_dict)
        if i % 1000 == 0 or i == 1:
            print('Step %i, Loss: %f' % (i, l))

#### Testing ####
    noise_input = tf.placeholder(tf.float32, shape=[None, latent_dim])
    decoder = tf.matmul(noise_input, weights['decoder_h1']) + biases['decoder_b1']
    decoder = tf.nn.tanh(decoder)
    decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
    decoder = tf.nn.sigmoid(decoder)

# Building a manifold of generated digits
    n = 20
    x_axis = np.linspace(-3, 3, n)
    y_axis = np.linspace(-3, 3, n)

    canvas = np.empty((28 * n, 28 * n))
    for i, yi in enumerate(x_axis):
        for j, xi in enumerate(y_axis):
            z_mu = np.array([[xi, yi]] * batch_size)
            x_mean = sess.run(decoder, feed_dict={noise_input: z_mu})
            canvas[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = \
            x_mean[0].reshape(28, 28)

    plt.figure(figsize=(8, 10))
    Xi, Yi = np.meshgrid(x_axis, y_axis)
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.show()