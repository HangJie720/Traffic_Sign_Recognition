import os
import keras
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from GTSRB.utils_gtsrb import read_data, split_data, load_data
from GTSRB.utils_train import model_train, model_test_eval, tf_model_load, model_adv_train
from utils_attack import generate_gradient, fgsm, iteration_fgsm, batch_eval, rand_fgsm
from GTSRB.LeNet import LeNet
from GTSRB.AlexNet import AlexNet
FLAGS = flags.FLAGS


def train(learning_rate=0.001, factor=5e-4, epochs=100, batch_size=128, mu=0, sigma=0.1, eps=0.3):
    """
    Ensemble adversarial training
    :param learning_rate: learning rate for training
    :param factor: regularization factor
    :param epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param mu: A 0-D Tensor or Python value of type `dtype`. The mean of the
    truncated normal distribution.
    :param sigma: A 0-D Tensor or Python value of type `dtype`. The standard deviation
    of the truncated normal distribution.
    :param add_dropout: if or not dropout.
    :param eps: (optional float) attack step size (input variation).
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

    # Get GTSRB data
    X_train, y_train, X_test, y_test = load_data(train_data_dir, test_data_dir)

    # Split GTSRB training data to training data and validating data
    X_train, y_train, X_valid, y_valid = split_data(X_train, y_train)

    # One-hot encode image labels
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_valid = label_binarizer.fit_transform(y_valid)
    y_test = label_binarizer.fit_transform(y_test)
    print(y_train.shape)

    # Define input and output TF placeholders
    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, (None))

    one_hot_y = tf.one_hot(y, 43)

    # if src_models is not None, we train on adversarial examples that come
    # from multiple models

    adv_model0 = AlexNet(model_dir)

    logits = adv_model0(x)
    grad = generate_gradient(x, logits, one_hot_y, loss='training')
    x_advs = fgsm(x, grad, eps, clipping=True)

    # parameters required by training
    train_params = {
        'factor': factor,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }

    model, saver = AlexNet(None, mu, sigma)
    # Train an gtsrb model
    model_adv_train(sess, x, one_hot_y, model, X_train, y_train, X_valid, y_valid, args=train_params, x_advs=x_advs)

    # Save model to specific position
    saver(sess, adv_model_dir)
    # Print out the accuracy on legitimate data
    # Flatten list from the tensorflow
    eval_params = {'batch_size': batch_size}
    logits = model(x)
    test_accuracy, pred = model_test_eval(sess, x, one_hot_y, logits, X_test, y_test, args=eval_params)
    print('Test accuracy of AlexNet on legitimate test '
          'examples: {:.3f}'.format(test_accuracy))


def main(argv=None):
    train(learning_rate = FLAGS.LEARNING_RATE, factor=FLAGS.REGULARIZATION_FACTOR,
          epochs = FLAGS.NUM_EPOCHS, batch_size=FLAGS.BATCH_SIZE,
          mu = FLAGS.MU, sigma = FLAGS.SIGMA,eps=FLAGS.EPS)


if __name__ == '__main__':
    ROOT_PATH = "../GTSRB"
    SAVE_PATH = "../GTSRB/models"
    train_data_dir = os.path.join(ROOT_PATH, "train.p")
    test_data_dir = os.path.join(ROOT_PATH, "test.p")
    model_dir = os.path.join(SAVE_PATH, "AlexNet")
    model_dir1 = os.path.join(SAVE_PATH, "AlexNet")
    adv_model_dir = os.path.join(SAVE_PATH, "AlexNet_adv")
    # General flags
    flags.DEFINE_string('train_data_dir', train_data_dir, 'Training datasets directory')
    flags.DEFINE_string('test_data_dir', test_data_dir, 'Testing datasets directory')
    flags.DEFINE_string('model_dir', model_dir, 'Saving model path')
    flags.DEFINE_integer('BATCH_SIZE', 128, 'Size of training batches')
    flags.DEFINE_integer('NUM_EPOCHS', 100, 'Number of epochs')
    flags.DEFINE_float('LEARNING_RATE', 0.001, 'Learning rate for training')
    flags.DEFINE_float('REGULARIZATION_FACTOR', 5e-4, 'Regularization factor')
    flags.DEFINE_float('MU', 0, 'The mean of thetruncated normal distribution')
    flags.DEFINE_float('SIGMA', 0.1, 'The standard deviation of the truncated normal distribution')
    flags.DEFINE_float('EPS', 0.3, 'The epsilon (input variation parameter)')
    flags.DEFINE_float('DROPOUT', 0.8, 'DROPOUT VALUE')
    flags.DEFINE_boolean('ADD_DROPOUT', False, 'Decide if add dropout process when training')

    app.run()