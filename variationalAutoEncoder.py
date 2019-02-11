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

