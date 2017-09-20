import os
import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.preprocessing import LabelBinarizer
from utils_gtsrb import load_data
from utils_train import model_test_eval
from GTSRB.MNIST import MNIST
FLAGS = flags.FLAGS

def eval(mu, sigma, add_dropout, batch_size):
    """
    Evaluate a GTSRB model
    :param mu: A 0-D Tensor or Python value of type `dtype`. The mean of the
    truncated normal distribution.
    :param sigma: A 0-D Tensor or Python value of type `dtype`. The standard deviation
    of the truncated normal distribution.
    :param add_dropout: if or not use dropout.
    :param batch_size: size of training batches
    :return: a dictionary with:
             * model training accuracy and loss on training data
             * model validating accuracy and loss on validation data
             * accuracy on test dataset by class lebel
    """

    np.random.seed(0)
    assert keras.backend.backend() == "tensorflow"

    # # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get MNIST data
    mnist = input_data.read_data_sets('/tmp/', one_hot=True, reshape=False)
    X_train, y_train, X_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    # One-Hot Encode
    # label_binarizer = LabelBinarizer()
    # y_test = label_binarizer.fit_transform(y_test)
    # print(y_test.shape)

    # Define input and output TF placeholders
    x = tf.placeholder(tf.float32, (None, 28, 28, 1))
    y = tf.placeholder(tf.float32, (None, 10))

    # one_hot_y = tf.one_hot(y, 43)

    model = MNIST(model_dir)
    logits, _ = model(x, add_dropout)

    print('loading MNIST ...')
    # Parameters required by evaluating
    eval_params = {'batch_size': batch_size}
    accuracy, pred = model_test_eval(sess, x, y, logits, X_test, y_test, args=eval_params)
    print('Test accuracy of MNIST on legitimate test '
          'examples: {:.3f}'.format(accuracy))


def main(argv=None):
    eval(mu = FLAGS.MU, sigma = FLAGS.SIGMA, add_dropout = FLAGS.ADD_DROPOUT, batch_size=FLAGS.BATCH_SIZE)


if __name__ == '__main__':
    ROOT_PATH = "../GTSRB"
    SAVE_PATH = "../GTSRB/models"
    train_data_dir = os.path.join(ROOT_PATH, "train.p")
    test_data_dir = os.path.join(ROOT_PATH, "test.p")
    model_dir = os.path.join(SAVE_PATH, "MNIST")

    # General flags
    flags.DEFINE_integer('BATCH_SIZE', 128, 'Size of training batches')
    flags.DEFINE_float('MU', 0, 'The mean of thetruncated normal distribution')
    flags.DEFINE_float('SIGMA', 0.1, 'The standard deviation of the truncated normal distribution')
    flags.DEFINE_boolean('ADD_DROPOUT', False, 'Decide if add dropout process when training')
    app.run()