import pickle
import numpy as np
import tensorflow as tf

def Inception(restore):
    if restore == None:
        map1 = 32
        map2 = 64
        num_fc1 = 700  # 1028
        num_fc2 = 43
        reduce1x1 = 16

        def createWeight(size, Name):
            return tf.Variable(tf.truncated_normal(size, stddev=0.1),
                               name=Name)

        def createBias(size, Name):
            return tf.Variable(tf.constant(0.1, shape=size),
                               name=Name)

        def conv2d_s1(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_3x3_s1(x):
            return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                                  strides=[1, 1, 1, 1], padding='SAME')

        # Inception Module 1
        #
        # follows input
        W_conv1_1x1_1 = createWeight([1, 1, 3, map1], 'W_conv1_1x1_1')
        b_conv1_1x1_1 = createWeight([map1], 'b_conv1_1x1_1')

        # follows input
        W_conv1_1x1_2 = createWeight([1, 1, 3, reduce1x1], 'W_conv1_1x1_2')
        b_conv1_1x1_2 = createWeight([reduce1x1], 'b_conv1_1x1_2')

        # follows input
        W_conv1_1x1_3 = createWeight([1, 1, 3, reduce1x1], 'W_conv1_1x1_3')
        b_conv1_1x1_3 = createWeight([reduce1x1], 'b_conv1_1x1_3')

        # follows 1x1_2
        W_conv1_3x3 = createWeight([3, 3, reduce1x1, map1], 'W_conv1_3x3')
        b_conv1_3x3 = createWeight([map1], 'b_conv1_3x3')

        # follows 1x1_3
        W_conv1_5x5 = createWeight([5, 5, reduce1x1, map1], 'W_conv1_5x5')
        b_conv1_5x5 = createBias([map1], 'b_conv1_5x5')

        # follows max pooling
        W_conv1_1x1_4 = createWeight([1, 1, 3, map1], 'W_conv1_1x1_4')
        b_conv1_1x1_4 = createWeight([map1], 'b_conv1_1x1_4')

        # Inception Module 2
        #
        # follows inception1
        W_conv2_1x1_1 = createWeight([1, 1, 4 * map1, map2], 'W_conv2_1x1_1')
        b_conv2_1x1_1 = createWeight([map2], 'b_conv2_1x1_1')

        # follows inception1
        W_conv2_1x1_2 = createWeight([1, 1, 4 * map1, reduce1x1], 'W_conv2_1x1_2')
        b_conv2_1x1_2 = createWeight([reduce1x1], 'b_conv2_1x1_2')

        # follows inception1
        W_conv2_1x1_3 = createWeight([1, 1, 4 * map1, reduce1x1], 'W_conv2_1x1_3')
        b_conv2_1x1_3 = createWeight([reduce1x1], 'b_conv2_1x1_3')

        # follows 1x1_2
        W_conv2_3x3 = createWeight([3, 3, reduce1x1, map2], 'W_conv2_3x3')
        b_conv2_3x3 = createWeight([map2], 'b_conv2_3x3')

        # follows 1x1_3
        W_conv2_5x5 = createWeight([5, 5, reduce1x1, map2], 'W_conv2_5x5')
        b_conv2_5x5 = createBias([map2], 'b_conv2_5x5')

        # follows max pooling
        W_conv2_1x1_4 = createWeight([1, 1, 4 * map1, map2], 'W_conv2_1x1_4')
        b_conv2_1x1_4 = createWeight([map2], 'b_conv2_1x1_4')

        # Fully connected layers
        # since padding is same, the feature map with there will be 4 32*32*map2
        W_fc1 = createWeight([32 * 32 * (4 * map2), num_fc1], 'W_fc1')
        b_fc1 = createBias([num_fc1], 'b_fc1')

        W_fc2 = createWeight([num_fc1, num_fc2], 'W_fc2')
        b_fc2 = createBias([num_fc2], 'b_fc2')

    else:
        W_conv1_1x1_1, b_conv1_1x1_1, W_conv1_1x1_2, b_conv1_1x1_2, W_conv1_1x1_3, b_conv1_1x1_3, W_conv1_3x3, b_conv1_3x3, W_conv1_5x5, b_conv1_5x5, W_conv1_1x1_4, b_conv1_1x1_4, W_conv2_1x1_1, b_conv2_1x1_1, W_conv2_1x1_2, b_conv2_1x1_2, W_conv2_1x1_3, b_conv2_1x1_3, W_conv2_3x3, b_conv2_3x3, W_conv2_5x5, b_conv2_5x5, W_conv2_1x1_4, b_conv2_1x1_4, W_fc1, b_fc1, W_fc2, b_fc2= [tf.constant(np.array(x, dtype=np.float32)) for x in pickle.load(open(restore, "rb"))]

    def model(x, keep_prob=1.0):
        # Inception Module 1
        conv1_1x1_1 = conv2d_s1(x, W_conv1_1x1_1) + b_conv1_1x1_1
        conv1_1x1_2 = tf.nn.relu(conv2d_s1(x, W_conv1_1x1_2) + b_conv1_1x1_2)
        conv1_1x1_3 = tf.nn.relu(conv2d_s1(x, W_conv1_1x1_3) + b_conv1_1x1_3)
        conv1_3x3 = conv2d_s1(conv1_1x1_2, W_conv1_3x3) + b_conv1_3x3
        conv1_5x5 = conv2d_s1(conv1_1x1_3, W_conv1_5x5) + b_conv1_5x5
        maxpool1 = max_pool_3x3_s1(x)
        conv1_1x1_4 = conv2d_s1(maxpool1, W_conv1_1x1_4) + b_conv1_1x1_4

        # concatenate all the feature maps and hit them with a relu
        inception1 = tf.nn.relu(tf.concat([conv1_1x1_1, conv1_3x3, conv1_5x5, conv1_1x1_4],3))

        # Inception Module 2
        conv2_1x1_1 = conv2d_s1(inception1, W_conv2_1x1_1) + b_conv2_1x1_1
        conv2_1x1_2 = tf.nn.relu(conv2d_s1(inception1, W_conv2_1x1_2) + b_conv2_1x1_2)
        conv2_1x1_3 = tf.nn.relu(conv2d_s1(inception1, W_conv2_1x1_3) + b_conv2_1x1_3)
        conv2_3x3 = conv2d_s1(conv2_1x1_2, W_conv2_3x3) + b_conv2_3x3
        conv2_5x5 = conv2d_s1(conv2_1x1_3, W_conv2_5x5) + b_conv2_5x5
        maxpool2 = max_pool_3x3_s1(inception1)
        conv2_1x1_4 = conv2d_s1(maxpool2, W_conv2_1x1_4) + b_conv2_1x1_4

        # concatenate all the feature maps and hit them with a relu
        inception2 = tf.nn.relu(tf.concat([conv2_1x1_1, conv2_3x3, conv2_5x5, conv2_1x1_4],3))

        # flatten features for fully connected layer
        inception2_flat = tf.reshape(inception2, [-1, 32 * 32 * 4 * map2])

        # Fully connected layers, w/ dropout

        h_fc1 = tf.nn.dropout(tf.nn.relu(tf.matmul(inception2_flat, W_fc1) + b_fc1), keep_prob)

        # Layer2
        logits = tf.matmul(h_fc1, W_fc2) + b_fc2

        return logits

    def saver(s, nn):
        def deeplist(x):
            try:
                x[0]
                return list(map(deeplist, x))
            except:
                return x

        dd = [deeplist(s.run(x)) for x in
              [W_conv1_1x1_1, b_conv1_1x1_1, W_conv1_1x1_2, b_conv1_1x1_2, W_conv1_1x1_3, b_conv1_1x1_3, W_conv1_3x3, b_conv1_3x3, W_conv1_5x5, b_conv1_5x5, W_conv1_1x1_4, b_conv1_1x1_4, W_conv2_1x1_1, b_conv2_1x1_1, W_conv2_1x1_2, b_conv2_1x1_2, W_conv2_1x1_3, b_conv2_1x1_3, W_conv2_3x3, b_conv2_3x3, W_conv2_5x5, b_conv2_5x5, W_conv2_1x1_4, b_conv2_1x1_4, W_fc1, b_fc1, W_fc2, b_fc2]]
        pickle.dump(dd, open(nn, "wb"), pickle.HIGHEST_PROTOCOL)

    if restore:
        return model
    else:
        return model, saver
