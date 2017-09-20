import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import pickle

def AlexNet(restore, mu = 0, sigma = 0.1):
    """
        Defines an CNN model AlexNet for Traffic Sign Recognition.
        :param restore: Decide whether to save model to specific path.
        :param mu: A 0-D Tensor or Python value of type `dtype`. The mean of the
      truncated normal distribution.
        :param sigma: A 0-D Tensor or Python value of type `dtype`. The standard deviation
      of the truncated normal distribution.
        :return:
    """
    if restore == None:
        conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 64), mean=mu, stddev=sigma))
        conv1_b = tf.Variable(tf.zeros(64))

        conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), mean=mu, stddev=sigma))
        conv2_b = tf.Variable(tf.zeros(128))

        conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 256), mean=mu, stddev=sigma))
        conv3_b = tf.Variable(tf.zeros(256))

        fc1_W = tf.Variable(tf.truncated_normal(shape=(4*4*256, 1024), mean=mu, stddev=sigma))
        fc1_b = tf.Variable(tf.zeros(1024))

        fc2_W = tf.Variable(tf.truncated_normal(shape=(1024, 1024), mean=mu, stddev=sigma))
        fc2_b = tf.Variable(tf.zeros(1024))

        fc3_W = tf.Variable(tf.truncated_normal(shape=(1024, 43), mean=mu, stddev=sigma))
        fc3_b = tf.Variable(tf.zeros(43))
    else:
        conv1_W, conv1_b, conv2_W, conv2_b, conv3_W, conv3_b, fc1_W, fc1_b, fc2_W, fc2_b, fc3_W, fc3_b = [tf.constant(np.array(x,dtype=np.float32)) for x in pickle.load(open(restore,"rb"))]


    def model(x):
        """
            Defines an CNN model.
            :param x: path to folder where training data should be stored.
            :param add_dropout: if or not dropout.
            :param add_dropout: if or not use dropout to avoid over-fitting.
            :return:
        """
        conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='conv1')
        conv1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='conv1')
        conv1 = tf.nn.dropout(conv1, 0.5)

        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME', name='conv2') + conv2_b
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='conv2')
        conv2 = tf.nn.dropout(conv2, 0.5)

        conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='SAME', name='conv3') + conv3_b
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='conv3')
        conv3 = tf.nn.dropout(conv3, 0.5)

        fc0 = flatten(conv3)
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b
        fc1 = tf.nn.relu(fc1)

        fc2 = tf.matmul(fc1, fc2_W) + fc2_b
        fc2 = tf.nn.relu(fc2)

        logits = tf.matmul(fc2, fc3_W) + fc3_b

        return logits

    def saver(s, nn):
        def deeplist(x):
            try:
                x[0]
                return list(map(deeplist, x))
            except:
                return x

        dd = [deeplist(s.run(x)) for x in
              [conv1_W, conv1_b, conv2_W, conv2_b, conv3_W, conv3_b, fc1_W, fc1_b, fc2_W, fc2_b, fc3_W, fc3_b]]
        pickle.dump(dd, open(nn, "wb"), pickle.HIGHEST_PROTOCOL)

    if restore:
        return model
    else:
        return model, saver
