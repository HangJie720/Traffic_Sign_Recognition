import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import pickle

def LeNet(restore, mu = 0, sigma = 0.1):
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
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma))
        conv1_b = tf.Variable(tf.zeros(6))

        # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
        conv2_b = tf.Variable(tf.zeros(16))

        # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
        fc1_b = tf.Variable(tf.zeros(120))

        # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
        fc2_b = tf.Variable(tf.zeros(84))

        # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
        fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
        fc3_b = tf.Variable(tf.zeros(43))
    else:
        conv1_W, conv1_b, conv2_W, conv2_b, fc1_W, fc1_b, fc2_W, fc2_b, fc3_W, fc3_b = [tf.constant(np.array(x,dtype=np.float32)) for x in pickle.load(open(restore,"rb"))]


    def model(x, add_dropout = False):
        """
            Defines an CNN model.
            :param x: path to folder where training data should be stored.
            :param add_dropout: if or not dropout.
            :param add_dropout: if or not use dropout to avoid over-fitting.
            :return:
        """
        conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

        # SOLUTION: Activation.
        conv1 = tf.nn.relu(conv1)

        # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='conv1')

        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID', name='conv2') + conv2_b

        # SOLUTION: Activation.
        conv2 = tf.nn.relu(conv2)

        # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
        fc0 = flatten(conv2)

        fc1 = tf.matmul(fc0, fc1_W) + fc1_b

        # SOLUTION: Activation.
        fc1 = tf.nn.relu(fc1)

        # Add dropout
        # if add_dropout:
        #     fc1 = tf.nn.dropout(fc1, 0.75)

        fc2 = tf.matmul(fc1, fc2_W) + fc2_b

        # SOLUTION: Activation.
        fc2 = tf.nn.relu(fc2)

        # Add dropout
        if add_dropout:
            fc2 = tf.nn.dropout(fc2, 0.75)

        logits = tf.matmul(fc2, fc3_W) + fc3_b

        # regularizers = (tf.nn.l2_loss(fc1_W) + tf.nn.l2_loss(fc1_b) + tf.nn.l2_loss(fc2_W) + tf.nn.l2_loss(fc2_b) + tf.nn.l2_loss(fc3_W) + tf.nn.l2_loss(fc3_b))

        return logits

    def saver(s, nn):
        def deeplist(x):
            try:
                x[0]
                return list(map(deeplist, x))
            except:
                return x

        dd = [deeplist(s.run(x)) for x in
              [conv1_W, conv1_b, conv2_W, conv2_b, fc1_W, fc1_b, fc2_W, fc2_b, fc3_W, fc3_b]]
        pickle.dump(dd, open(nn, "wb"), pickle.HIGHEST_PROTOCOL)

    if restore:
        return model
    else:
        return model, saver
