import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import pickle

def MNIST(restore, mu = 0, sigma = 0.1):
    """
        Defines an CNN model LeNet-5 for Traffic Sign Recognition.
        :param restore: Decide whether to save model to specific path.
        :param mu: A 0-D Tensor or Python value of type `dtype`. The mean of the
      truncated normal distribution.
        :param sigma: A 0-D Tensor or Python value of type `dtype`. The standard deviation
      of the truncated normal distribution.
        :return:
    """
    if restore == None:
        # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
        conv1_b = tf.Variable(tf.zeros(6))

        # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
        conv2_b = tf.Variable(tf.zeros(16))

        # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
        conv3_W = tf.Variable(tf.truncated_normal(shape=(4, 4, 16, 120), mean=mu, stddev=sigma))
        conv3_b = tf.Variable(tf.zeros(120))

        # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
        fc1_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
        fc1_b = tf.Variable(tf.zeros(84))

        # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
        fc2_W = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=mu, stddev=sigma))
        fc2_b = tf.Variable(tf.zeros(10))
    else:
        conv1_W, conv1_b, conv2_W, conv2_b, conv3_W, conv3_b, fc1_W, fc1_b, fc2_W, fc2_b = [tf.constant(np.array(x,dtype=np.float32)) for x in pickle.load(open(restore,"rb"))]


    def model(x, add_dropout = False):
        """
            Defines an CNN model.
            :param x: path to folder where training data should be stored.
            :param add_dropout: if or not dropout.
            :param add_dropout: if or not use dropout to avoid over-fitting.
            :return:
        """
        conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

        conv1 = tf.nn.relu(conv1)

        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='conv1')

        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID', name='conv2') + conv2_b

        conv2 = tf.nn.relu(conv2)

        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID', name='conv3') + conv3_b

        conv3 = tf.nn.relu(conv3)

        fc0 = flatten(conv3)

        fc1 = tf.matmul(fc0, fc1_W) + fc1_b

        # SOLUTION: Activation.
        fc1 = tf.nn.relu(fc1)

        # Add dropout
        if add_dropout:
            fc1 = tf.nn.dropout(fc1, 0.5)



        logits = tf.matmul(fc1, fc2_W) + fc2_b

        regularizers = (tf.nn.l2_loss(fc1_W) + tf.nn.l2_loss(fc1_b) + tf.nn.l2_loss(fc2_W) + tf.nn.l2_loss(fc2_b))

        return logits, regularizers

    def saver(s, nn):
        def deeplist(x):
            try:
                x[0]
                return list(map(deeplist, x))
            except:
                return x

        dd = [deeplist(s.run(x)) for x in
              [conv1_W, conv1_b, conv2_W, conv2_b, conv3_W, conv3_b, fc1_W, fc1_b, fc2_W, fc2_b]]
        pickle.dump(dd, open(nn, "wb"), pickle.HIGHEST_PROTOCOL)

    if restore:
        return model
    else:
        return model, saver
