import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#MNIST Data
import mnist_data
mnist = mnist_data.read_data_sets("F:\ASU\pro_auto_encoder", one_hot=True)

#Hyperparameters
learning_rate = 0.01
num_steps = 100
batch_size = 256

display_step = 100
examples_to_show = 10

#Network

num_hidden = 256
latent_dim = 128
num_input = 784

#Graph

X = tf.placeholder("float", [None, num_input])

####weights
weights= {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden, latent_dim])),
    'decoder_h1': tf.Variable(tf.random_normal([latent_dim, num_hidden])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden, num_input])),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden])),
    'encoder_b2': tf.Variable(tf.random_normal([latent_dim])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

#####Model
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1']))
    layer = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),biases['encoder_b2']))
    return layer
    
def decoder(y):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(y, weights['decoder_h1']),biases['decoder_b1']))
    layer = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2']))
    return layer
    
en_nn = encoder(X)
de_nn = decoder(en_nn)

# Prediction
y_pred = de_nn
# Targets (Labels) are the input data.
y_true = X

####loss and optimizer
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    for i in range(1, num_steps+1):
        batch_x, _ = mnist.train.next_batch(batch_size)
        
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))
            
    #Testing ------------
    n = 4
    ori = np.empty((28*n, 28*n))
    gen = np.empty((28*n, 28*n))
    
    for i in range(n):
        batch_x, _ = mnist.test.next_batch(n)
        
        dig = sess.run(de_nn, feed_dict={X: batch_x})
        for j in range(n):
            ori[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        for j in range(n):
            gen[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                dig[j].reshape([28, 28])
				
    print("Original Images")
    fig = plt.figure(figsize=(n, n))
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(ori, origin="upper", cmap="gray")
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(gen, origin="upper", cmap="gray")
    plt.show()

    
        
