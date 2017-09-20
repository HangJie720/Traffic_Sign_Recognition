import numpy as np
import keras.backend as K
import tensorflow as tf
from utils_train import _FlagsWrapper
from utils_train import model_loss
from LeNet import LeNet
def linf_loss(X1, X2):
    return np.max(np.abs(X1 - X2), axis=(1, 2, 3))

def generate_gradient(x, logits, y, loss='logloss'):
    """
    Generate the gradient of the loss function.
    :param x: input placeholder
    :param logits: model output predictions
    :param y: output placeholder (for labels)
    :param loss: Decide if or not to use the model's output instead of the true labels to avoid label leaking at training time
    """

    adv_loss = model_loss(logits, y, loss=loss, mean=False)

    # Define gradient of loss wrt input
    grad = tf.gradients(adv_loss, [x])[0]
    return grad

def fgsm(x, grad, eps=0.3, clipping=True):
    """
    TensorFlow implementation of the Fast Gradient Method.
    :param x: the input placeholder
    :param grad: the model's gradient
    :param eps: the epsilon (input variation parameter)
    :param clipping: # If clipping is True, reset all values outside of [0, 1]
    :return: a tensor for the adversarial example
    """

    # signed gradient
    normed_grad = tf.sign(grad)

    # Multiply by constant epsilon
    scaled_grad = eps * normed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_grad)

    if clipping:
        adv_x = tf.clip_by_value(adv_x, 0, 1)
    return adv_x


def iteration_fgsm(model, x, y, steps, eps, loss='logloss'):
    """
    TensorFlow implementation of the Iteration Fast Gradient Method.
    :param x: the input placeholder
    :param y: the output placeholder
    :param steps: the number of iteration
    :param eps: the epsilon (input variation parameter)
    :param loss: Decide if or not to use the model's output instead of the true labels to avoid label leaking at training time
    :return: a tensor for the adversarial example
    """

    adv_x = x
    # iteratively apply the FGSM with small step size
    for i in range(steps):
        logits = model(adv_x)
        grad = generate_gradient(adv_x, logits, y, loss=loss)
        adv_x = fgsm(adv_x, grad, eps, clipping=True)

    return adv_x

def rand_fgsm(model, x, y,  X_test, alpha, eps):
    """
    RAND+FGSM attack.
    :return: adv_x  --adversarial sample
    """
    X_test = np.clip(
        X_test + alpha * np.sign(np.random.randn(*X_test.shape)),
        0.0, 1.0)
    eps -= alpha
    logits = model(x)
    grad = generate_gradient(x, logits, y)

    adv_x = fgsm(x, grad, eps, True)

    return adv_x, X_test

def batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs, feed=None,
               args=None):
    """
    A helper function that computes a tensor on numpy inputs by batches.

    :param sess:
    :param tf_inputs:
    :param tf_outputs:
    :param numpy_inputs:
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance.
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    """
    args = _FlagsWrapper(args or {})

    assert args.batch_size, "Batch size was not given in args dict"

    n = len(numpy_inputs)
    assert n > 0
    assert n == len(tf_inputs)
    m = numpy_inputs[0].shape[0]
    for i in range(1, n):
        assert numpy_inputs[i].shape[0] == m
    out = []
    for _ in tf_outputs:
        out.append([])
    with sess.as_default():
        for start in range(0, m, args.batch_size):
            batch = start // args.batch_size

            # Compute batch start and end indices
            start = batch * args.batch_size
            end = start + args.batch_size
            numpy_input_batches = [numpy_input[start:end]
                                   for numpy_input in numpy_inputs]
            cur_batch_size = numpy_input_batches[0].shape[0]
            assert cur_batch_size <= args.batch_size
            for e in numpy_input_batches:
                assert e.shape[0] == cur_batch_size

            feed_dict = dict(zip(tf_inputs, numpy_input_batches))
            if feed is not None:
                feed_dict.update(feed)
            numpy_output_batches = sess.run(tf_outputs, feed_dict=feed_dict)
            for e in numpy_output_batches:
                assert e.shape[0] == cur_batch_size, e.shape
            for out_elem, numpy_output_batch in zip(out, numpy_output_batches):
                out_elem.append(numpy_output_batch)

    out = [np.concatenate(x, axis=0) for x in out]
    for e in out:
        assert e.shape[0] == m, e.shape
    return out
