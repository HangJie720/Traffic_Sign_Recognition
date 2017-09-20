import os
import keras
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.examples.tutorials.mnist import input_data
from GTSRB.utils_gtsrb import read_data, split_data, load_data
from GTSRB.utils_train import model_train, model_test_eval, tf_model_load
from GTSRB.MNIST import MNIST
FLAGS = flags.FLAGS


def train(learning_rate=0.001, factor=5e-4, epochs=30, batch_size=128, mu=0, sigma=0.1, add_dropout=False):
    """
    Train a GTSRB model
    :param learning_rate: learning rate for training
    :param factor: regularization factor
    :param epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param mu: A 0-D Tensor or Python value of type `dtype`. The mean of the
    truncated normal distribution.
    :param sigma: A 0-D Tensor or Python value of type `dtype`. The standard deviation
    of the truncated normal distribution.
    :param add_dropout: if or not dropout.
    :return: a dictionary with:
             * model training accuracy and loss on training data
             * model validating accuracy and loss on validation data
             * accuracy on test dataset by class lebel
    """
    np.random.seed(0)
    assert keras.backend.backend() == "tensorflow"

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get MNIST data
    mnist = input_data.read_data_sets('/tmp/', one_hot=True, reshape=False)
    X_train, y_train, X_test, y_test =mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels
    # Split MNIST training data to training data and validating data
    X_valid, y_valid = mnist.validation.images,mnist.validation.labels

    # One-hot encode image labels
    # label_binarizer = LabelBinarizer()
    # y_train = label_binarizer.fit_transform(y_train)
    # y_valid = label_binarizer.fit_transform(y_valid)
    # y_test = label_binarizer.fit_transform(y_test)
    # print(y_train.shape)

    # Define input and output TF placeholders
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # one_hot_y = tf.one_hot(y, 43)

    model, saver = MNIST(None, mu, sigma)
    logits, regularizers = model(x, add_dropout)

    # parameters required by training
    train_params = {
        'factor': factor,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    # Train an gtsrb model
    model_train(sess, x, y, logits, regularizers, X_train, y_train, X_valid, y_valid, args=train_params)

    # Save model to specific position
    saver(sess, model_dir)
    # Print out the accuracy on legitimate data
    # Flatten list from the tensorflow
    eval_params = {'batch_size': batch_size}
    test_accuracy, pred = model_test_eval(sess, x, y, logits, X_test, y_test, args=eval_params)
    print('Test accuracy of LeNet on legitimate test '
          'examples: {:.3f}'.format(test_accuracy))


def main(argv=None):
    train(learning_rate = FLAGS.LEARNING_RATE, factor=FLAGS.REGULARIZATION_FACTOR,
          epochs = FLAGS.NUM_EPOCHS, batch_size=FLAGS.BATCH_SIZE,
          mu = FLAGS.MU, sigma = FLAGS.SIGMA, add_dropout = FLAGS.ADD_DROPOUT)


if __name__ == '__main__':
    ROOT_PATH = "../GTSRB"
    SAVE_PATH = "../GTSRB/models"
    train_data_dir = os.path.join(ROOT_PATH, "train.p")
    test_data_dir = os.path.join(ROOT_PATH, "test.p")
    model_dir = os.path.join(SAVE_PATH, "MNIST")
    # General flags
    flags.DEFINE_string('train_data_dir', train_data_dir, 'Training datasets directory')
    flags.DEFINE_string('test_data_dir', test_data_dir, 'Testing datasets directory')
    flags.DEFINE_string('model_dir', model_dir, 'Saving model path')
    flags.DEFINE_integer('BATCH_SIZE', 128, 'Size of training batches')
    flags.DEFINE_integer('NUM_EPOCHS', 30, 'Number of epochs')
    flags.DEFINE_float('LEARNING_RATE', 0.001, 'Learning rate for training')
    flags.DEFINE_float('REGULARIZATION_FACTOR', 5e-4, 'Regularization factor')
    flags.DEFINE_float('MU', 0, 'The mean of thetruncated normal distribution')
    flags.DEFINE_float('SIGMA', 0.1, 'The standard deviation of the truncated normal distribution')
    flags.DEFINE_boolean('ADD_DROPOUT', False, 'Decide if add dropout process when training')

    app.run()