import os
import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from sklearn.preprocessing import LabelBinarizer
from utils_gtsrb import load_data, display_leg_adv_sample, pair_visual
from utils_train import model_test_eval
from utils_attack import generate_gradient, fgsm, iteration_fgsm, batch_eval, rand_fgsm
from GTSRB.AlexNet import AlexNet
from GTSRB.LeNet import LeNet
from l2_attack import CarliniL2
from li_attack import CarliniLi
from squeeze import binary_filter_tf
FLAGS = flags.FLAGS

def gtsrb_whitebox(attack, mu, sigma, batch_size, eps, steps, kappa, alpha):
    """
    GTSRB tutorial for the white-box attack.
    :param attack: attack type(FGSM, I-FGSM, Carlini)
    :param mu: A 0-D Tensor or Python value of type `dtype`. The mean of the
    truncated normal distribution.
    :param sigma: A 0-D Tensor or Python value of type `dtype`. The standard deviation
    of the truncated normal distribution.
    :param dropout: dropout value
    :param add_dropout: if or not dropout.
    :param batch_size: size of training batches
    :param eps: the epsilon (input variation parameter)
    :return: a dictionary with:
             * white-box model accuracy on test set
             * white-box model accuracy on adversarial examples transferred
               from the substitute model
    """

    np.random.seed(0)
    assert keras.backend.backend() == "tensorflow"

    # # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get GTSRB data
    _, _, X_test, y_test = load_data(train_data_dir, test_data_dir)

    # One-Hot Encode
    label_binarizer = LabelBinarizer()
    y_test = label_binarizer.fit_transform(y_test)

    # Define input and output TF placeholders
    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, (None))

    one_hot_y = tf.one_hot(y, 43)

    print('Preparing the white-box model LeNet.')
    model = LeNet(model_dir, mu, sigma)
    logits = model(x)

    # Parameters required by evaluating
    eval_params = {'batch_size': batch_size}

    accuracy, pred = model_test_eval(sess, x, one_hot_y, logits, X_test, y_test, args=eval_params)
    print('Test accuracy of white-box on legitimate test '
              'examples: {:.3f}'.format(accuracy))

    model = AlexNet(model_dir, mu, sigma)
    logits = model(x)
    # take the random step in the RAND+FGSM
    if attack == "rand_fgsm":
        X_test = np.clip(
            X_test + alpha * np.sign(np.random.randn(*X_test.shape)),
            0.0, 1.0)
        eps -= alpha

    grad = generate_gradient(x, logits, one_hot_y, loss='training')
    # FGSM and RAND_FGSM one-shot attack
    if attack in ["fgsm", "rand_fgsm"]:
        adv_x = fgsm(x, grad, eps, clipping=True)
    if attack == 'ifgsm':
        adv_x = iteration_fgsm(model, x, one_hot_y, steps, eps, loss='training')

    if attack == 'Carlini':
        X_test = X_test[0:128]
        Y_test = y_test[0:128]

        # cli = CarliniL2(sess, model, targeted=False, max_iterations=1000, confidence=0, batch_size=1)
        cli = CarliniLi(sess, model, targeted=False, max_iterations=1000)
        X_adv = cli.attack(X_test, Y_test)

        r = np.clip(X_adv - X_test, -eps, eps)
        X_adv = X_test + r

        accuracy, pred = model_test_eval(sess, x, one_hot_y, logits, X_adv, y_test, args=eval_params)
        print('Test accuracy of white-box on adversarial examples crafted by '
              'Carlini Attack: {:.3f}'.format(accuracy))
        display_leg_adv_sample(X_test, X_adv)
        return

    # compute the adversarial examples and evaluate
    X_adv = batch_eval(sess, [x, y], [adv_x], [X_test, y_test], args=eval_params)[0]
    accuracy, pred = model_test_eval(sess, x, one_hot_y, logits, X_adv, y_test, args=eval_params)
    print('Test accuracy of white-box on adversarial '
            'examples: {:.3f}'.format(accuracy))
    display_leg_adv_sample(X_test, X_adv)
    # pair_visual(X_test, X_adv)


def main(argv=None):
    gtsrb_whitebox(attack = FLAGS.ATTACK_TYPE, mu = FLAGS.MU, sigma = FLAGS.SIGMA,
                   batch_size=FLAGS.BATCH_SIZE, eps = FLAGS.EPS, steps = FLAGS.STEPS,
                   kappa = FLAGS.KAPPA, alpha = FLAGS.ALPHA)


if __name__ == '__main__':
    ROOT_PATH = "../GTSRB"
    SAVE_PATH = "../GTSRB/models"
    train_data_dir = os.path.join(ROOT_PATH, "train.p")
    test_data_dir = os.path.join(ROOT_PATH, "test.p")
    model_dir = os.path.join(SAVE_PATH, "LeNet_1")
    model_dir1 = os.path.join(SAVE_PATH, "AlexNet")

    # General flags
    flags.DEFINE_integer('BATCH_SIZE', 128, 'Size of training batches')
    flags.DEFINE_float('MU', 0, 'The mean of thetruncated normal distribution')
    flags.DEFINE_float('SIGMA', 0.1, 'The standard deviation of the truncated normal distribution')
    flags.DEFINE_boolean('ADD_DROPOUT', False, 'Decide if add dropout process when training')
    flags.DEFINE_boolean('DROPOUT', 0.5, 'Dropout value')
    flags.DEFINE_string('ATTACK_TYPE', 'fgsm', 'Select one attack type')
    flags.DEFINE_float('EPS', 0.3, 'The epsilon (input variation parameter)')
    flags.DEFINE_float('KAPPA', 100, 'Carlini attack confidence')
    flags.DEFINE_integer('STEPS', 10, 'Number of iteration')
    flags.DEFINE_float('ALPHA', 0.05, 'parameter for random noise')
    app.run()