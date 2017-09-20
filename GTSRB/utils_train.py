#-*- coding:utf-8 -*-
import os
import warnings
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.python.platform import flags


FLAGS = flags.FLAGS

class _ArgsWrapper(object):

    """
    Wrapper that allows attribute access to dictionaries
    """

    def __init__(self, args):
        if not isinstance(args, dict):
            args = vars(args)
        self.args = args
    def __getattr__(self, name):
        return self.args.get(name)

class _FlagsWrapper(_ArgsWrapper):
    """
    Wrapper that tries to find missing parameters in TensorFlow FLAGS
    for backwards compatibility.

    Plain _ArgsWrapper should be used instead if the support for FLAGS
    is removed.
    """
    def __getattr__(self, name):
        val = self.args.get(name)
        if val is None:
            warnings.warn('Setting parameters ({}) from TensorFlow FLAGS is '
                          'deprecated.'.format(name))
            val = FLAGS.__getattr__(name)
        return val

def model_train(sess, x, y, model, X_train, y_train, X_valid, y_valid, args=None):
    """
    Train a TF graph
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param model: model output predictions
    :param regularizers: tf.nn.L2_loss to all variables
    :param X_train: numpy array with training inputs
    :param y_train: numpy array with training outputs
    :param X_valid: numpy array with validating inputs
    :param y_valid: numpy array with validating outputs
    :param args: dict or argparse `Namespace` object.
            Should contain factor`, `epochs`, 'model_dir'
            `learning_rate`, `batch_size`
    :return: True if model trained
    """
    args = _FlagsWrapper(args or {})

    # Check that necessary arguments were given (see doc above)
    assert args.factor, "Number of factor was not given in args dict"
    assert args.epochs, "Number of epochs was not given in args dict"
    assert args.learning_rate, "Learning rate was not given in args dict"
    assert args.batch_size, "Batch size was not given in args dict"
    eval_params = {'batch_size': args.batch_size}

    # Define array
    val_accuracy = []
    train_accuracy = []
    val_loss = []
    train_loss = []

    # Save and restore variables
    # saver = tf.train.Saver()

    # Define loss
    loss_operation = model_loss(model, y)

    # L2 regularization for the fully connected parameters. Add regularization to loss term
    # loss_operation += args.factor * regularizers

    training_operation = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss_operation)

    # Train model
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        print("Training...")
        for i in range(args.epochs):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, args.batch_size):
                end = offset + args.batch_size
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            validation_accuracy, _ , validation_loss = model_train_eval(sess, x, y, model, loss_operation, X_valid, y_valid, args=eval_params)
            training_accuracy, _ , training_loss = model_train_eval(sess, x, y, model, loss_operation, X_train, y_train, args=eval_params)
            val_accuracy.append(validation_accuracy)
            val_loss.append(validation_loss)
            train_accuracy.append(training_accuracy)
            train_loss.append(training_loss)
            print("EPOCH {}...".format(i+1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy),",Validation Loss = {:.3f}".format(validation_loss))
            print("Training Accuracy = {:.3f}".format(training_accuracy),",Training Loss = {:.3f}".format(training_loss))
        # Plot accuracy and loss results
        accuracy_plot(val_accuracy, train_accuracy, val_loss, train_loss)
        # Save model to specific position
        # saver.save(sess, args.model_dir)
        print("Complete model training.")

    return True

def cnn_train(sess, x, y, model, X_train, y_train, X_valid, y_valid, args=None):
    """
    Train a TF graph
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param model: model output predictions
    :param regularizers: tf.nn.L2_loss to all variables
    :param X_train: numpy array with training inputs
    :param y_train: numpy array with training outputs
    :param X_valid: numpy array with validating inputs
    :param y_valid: numpy array with validating outputs
    :param args: dict or argparse `Namespace` object.
            Should contain factor`, `epochs`, 'model_dir'
            `learning_rate`, `batch_size`
    :return: True if model trained
    """
    args = _FlagsWrapper(args or {})

    # Check that necessary arguments were given (see doc above)
    assert args.epochs, "Number of epochs was not given in args dict"
    assert args.learning_rate, "Learning rate was not given in args dict"
    assert args.batch_size, "Batch size was not given in args dict"
    assert args.model_dir, "Model directory was not given in args dict"
    eval_params = {'batch_size': args.batch_size}

    # Define array
    val_accuracy = []
    train_accuracy = []
    val_loss = []
    train_loss = []

    # Save and restore variables
    saver = tf.train.Saver()

    # Define loss
    loss_operation = model_loss(model, y)

    training_operation = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss_operation)

    # Train model
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        print("Training...")
        for i in range(args.epochs):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, args.batch_size):
                end = offset + args.batch_size
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            validation_accuracy, _ , validation_loss = model_train_eval(sess, x, y, model, loss_operation, X_valid, y_valid, args=eval_params)
            training_accuracy, _ , training_loss = model_train_eval(sess, x, y, model, loss_operation, X_train, y_train, args=eval_params)
            val_accuracy.append(validation_accuracy)
            val_loss.append(validation_loss)
            train_accuracy.append(training_accuracy)
            train_loss.append(training_loss)
            print("EPOCH {}...".format(i+1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy),",Validation Loss = {:.3f}".format(validation_loss))
            print("Training Accuracy = {:.3f}".format(training_accuracy),",Training Loss = {:.3f}".format(training_loss))
        # Plot accuracy and loss results
        accuracy_plot(val_accuracy, train_accuracy, val_loss, train_loss)
        # Save model to specific position
        saver.save(sess, args.model_dir)
        print("Complete model training.")

        # tf_model_load(saver, sess, args.model_dir)
        # saver.restore(sess, tf.train.latest_checkpoint('../GTSRB/models'))
        # testing_acc, preds = model_test_eval(sess, x, y, model, X_test, y_test, args=eval_params)
        # print("Testing Accuracy = {:.3f}".format(testing_acc))
    return True

def model_adv_train(sess, x, y, model, X_train, y_train, X_valid, y_valid, args=None, x_advs=None):
    """
    Train a TF graph
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param model: model output predictions
    :param regularizers: tf.nn.L2_loss to all variables
    :param X_train: numpy array with training inputs
    :param y_train: numpy array with training outputs
    :param X_valid: numpy array with validating inputs
    :param y_valid: numpy array with validating outputs
    :param args: dict or argparse `Namespace` object.
            Should contain factor`, `epochs`, 'model_dir'
            `learning_rate`, `batch_size`
    :return: True if model trained
    """
    args = _FlagsWrapper(args or {})

    # Check that necessary arguments were given (see doc above)
    assert args.factor, "Number of factor was not given in args dict"
    assert args.epochs, "Number of epochs was not given in args dict"
    assert args.learning_rate, "Learning rate was not given in args dict"
    assert args.batch_size, "Batch size was not given in args dict"
    eval_params = {'batch_size': args.batch_size}

    # Define array
    val_accuracy = []
    train_accuracy = []
    val_loss = []
    train_loss = []

    # Save and restore variables
    # saver = tf.train.Saver()

    # Define loss
    logits = model(x)
    l1 = model_loss(logits, y)

    # idx = tf.placeholder(dtype=np.int32)
    logits_adv = model(x_advs)
    l2 = model_loss(logits_adv, y)
    loss_operation = 0.5*(l1 + l2)

    # L2 regularization for the fully connected parameters. Add regularization to loss term
    # loss_operation += args.factor * regularizers

    training_operation = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss_operation)

    # Train model
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        print("Training...")
        for i in range(args.epochs):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, args.batch_size):
                end = offset + args.batch_size
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            validation_accuracy, _ , validation_loss = model_train_eval(sess, x, y, logits, loss_operation, X_valid, y_valid, args=eval_params)
            training_accuracy, _ , training_loss = model_train_eval(sess, x, y, logits, loss_operation, X_train, y_train, args=eval_params)
            val_accuracy.append(validation_accuracy)
            val_loss.append(validation_loss)
            train_accuracy.append(training_accuracy)
            train_loss.append(training_loss)
            print("EPOCH {}...".format(i+1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy),",Validation Loss = {:.3f}".format(validation_loss))
            print("Training Accuracy = {:.3f}".format(training_accuracy),",Training Loss = {:.3f}".format(training_loss))
        # Plot accuracy and loss results
        accuracy_plot(val_accuracy, train_accuracy, val_loss, train_loss)
        # Save model to specific position
        # saver.save(sess, args.model_dir)
        print("Complete model training.")

    return True

def model_train_eval(sess, x, y, model, loss, X_test, y_test, args=None):
    """
    Compute the accuracy and loss of a TF model on some training data or validation data
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param model: model output predictions
    :param loss: model loss
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :param args: dict or argparse `Namespace` object.
                Should contain `batch_size`
    :return: a float with the accuracy value，loss value.
    """
    args = _FlagsWrapper(args or {})

    assert args.batch_size, "Batch size was not given in args dict"

    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    softmax = tf.nn.softmax(model)
    prediction = tf.argmax(model, 1)
    num_examples = len(X_test)
    total_accuracy = 0
    total_loss = 0
    pred = []
    # sess = tf.get_default_session()
    for offset in range(0, num_examples, args.batch_size):
        batch_x, batch_y = X_test[offset:offset+args.batch_size], y_test[offset:offset+args.batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        loss_oper = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
        predictions = sess.run(prediction, feed_dict={x: batch_x, y: batch_y})
        pred.append(predictions)
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss_oper * len(batch_x))
    return total_accuracy / num_examples, pred, total_loss / num_examples

def model_test_eval(sess, x, y, model, X_test, y_test, args=None):
    """
    Compute the accuracy of a TF model on some testing data
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param model: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :param args: dict or argparse `Namespace` object.
                Should contain `batch_size`
    :return: a float with the accuracy value and pred
    """
    args = _FlagsWrapper(args or {})

    assert args.batch_size, "Batch size was not given in args dict"

    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    softmax = tf.nn.softmax(model)
    prediction = tf.argmax(model, 1)
    num_examples = len(X_test)
    total_accuracy = 0
    pred = []
    # sess = tf.get_default_session()
    with sess.as_default():
        for offset in range(0, num_examples, args.batch_size):
            batch_x, batch_y = X_test[offset:offset+args.batch_size], y_test[offset:offset+args.batch_size]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
            predictions = sess.run(prediction, feed_dict={x: batch_x, y: batch_y})
            pred.append(predictions)
            total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples, pred

def accuracy_plot(val_accuracy, train_accuracy, val_loss, train_loss):
    plt.figure(figsize=(25, 10))
    fig = plt.figure()
    a = fig.add_subplot(121)
    line_one, = plt.plot(val_accuracy, label="Validation")
    line_two, = plt.plot(train_accuracy, label="Training")
    plt.ylabel('Accuracy values')
    plt.xlabel('No. of epochs')
    plt.legend(handles=[line_one, line_two])

    a = fig.add_subplot(122)
    line_one, = plt.plot(val_loss, label="Validation")
    line_two, = plt.plot(train_loss, label="Training")
    plt.ylabel('Loss values')
    plt.xlabel('No. of epochs')
    plt.legend(handles=[line_one, line_two])

    plt.show()

def model_loss(model, y, loss='logloss',mean=True):
    """
    Define loss of TF graph
    :param y: correct labels
    :param model: output of the model
    :param mean: boolean indicating whether should return mean of loss
                     or vector of losses for each input of the batch
    :return: return mean of loss if True, otherwise return vector with per
                 sample loss
    """
    if loss == 'training':
        # use the model's output instead of the true labels to avoid
        # label leaking at training time
        y = tf.cast(tf.equal(model, tf.reduce_max(model, 1, keep_dims=True)), "float32")
        y = y / tf.reduce_sum(y, 1, keep_dims=True)
        out = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=model)
    elif loss == 'logloss':
        out = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=model)
        # out = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=model)
    else:
        raise ValueError("Unknown loss: {}".format(loss))

    if mean:
        out = tf.reduce_mean(out)
    else:
        out = tf.reduce_sum(out)
    return out

def tf_model_load(saver, sess, model_dir):
    """
    :param sess:
    :param train_dir:
    :param filename:
    :return:
    """
    with sess.as_default():
        saver.restore(sess, model_dir)

    return True