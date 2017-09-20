import os
from tqdm import tqdm
import random
import argparse
import numpy as np
import GTSRB.load_data as ld
import matplotlib.pyplot as plt
from GTSRB.LeNet import LeNet
from skimage.color import rgb2gray
from sklearn.utils import shuffle
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.cross_validation import train_test_split
from tensorflow.python.platform import flags
import tensorflow as tf
def set_gtsrb_flags():
    """
        Set flags related to GTSRB datasets.
        :return:
    """
    try:
        flags.DEFINE_integer('BATCH_SIZE', 128, 'Size of training batches')
    except argparse.ArgumentError:
        pass

    flags.DEFINE_integer('NUM_CLASSES', 43, 'Number of classification classes')
    flags.DEFINE_integer('IMAGE_ROWS', 32, 'Input row dimension')
    flags.DEFINE_integer('IMAGE_COLS', 32, 'Input column dimension')
    flags.DEFINE_integer('NUM_CHANNELS', 3, 'Input depth dimension')

def preprocess(image):
    image = image.astype(float)
    return (image-255.0/2)/255.0
def preprocess_batch(images):
    imgs = np.copy(images)
    for i in tqdm(range(images.shape[0])):
        imgs[i] = preprocess(images[i])
    return imgs

def reduction(image):
    image = image.astype(float)
    image1 = (image + 0.5) * 255
    return image1
def reduction_batch(images):
    imgs = np.copy(images)
    for i in tqdm(range(images.shape[0])):
        imgs[i] = reduction(images[i])
    return imgs

def load_data(train_data_dir, test_data_dir):
    """
        Load GTSRB dataset.
        :param train_data_dir: path to folder where training data should be stored.
        :param test_data_dir: path to folder where testing data should be stored.
        :return: tuple of four arrays containing training data, training labels,
                     testing data and testing labels.
    """
    X_train, y_train, X_test, y_test = ld.load_data(train_data_dir, test_data_dir)

    # X_train_norm = (X_train.astype(float))/255 - 0.5
    # X_test_norm = (X_test.astype(float))/255 - 0.5
    X_train_norm = preprocess_batch(X_train.astype(float))
    X_test_norm = preprocess_batch(X_test.astype(float))

    # Transfer RGB to gray
    X_train_transfered, X_test_transfered = rgb_convert_grayscale(X_train_norm, X_test_norm)
    display_gray(X_train_transfered, X_test_transfered)
    # Shuffle training data
    X_train, y_train = shuffle(X_train_transfered, y_train)
    X_test, y_test = shuffle(X_test_transfered, y_test)

    return X_train, y_train, X_test, y_test

def read_data(train_data_dir, test_data_dir):
    """
        Load GTSRB dataset.
        :param train_data_dir: path to folder where training data should be stored.
        :param test_data_dir: path to folder where testing data should be stored.
        :return: tuple of four arrays containing training data, training labels,
                 testing data and testing labels.
    """
    train_images, train_labels = ld.load_train_data(train_data_dir)
    test_images, test_labels = ld.load_test_data(test_data_dir)

    X_train = np.array(train_images)
    y_train = np.array(train_labels)
    X_test = np.array(test_images)
    y_test = np.array(test_labels)
    # Shuffle training data
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)
    print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)

    return X_train, y_train, X_test, y_test

def read_model(model_dir, x, mu, sigma, dropout, add_dropout):
    try:
        with open(model_dir + '.json', 'r') as f:
            json_string = f.read()
            model = model_from_json(json_string)
    except IOError:
        model = LeNet(x, mu, sigma, dropout,add_dropout)

    model.load_weights(model_dir)
    return model

def split_data(X_train, y_train):
    """
        Split the data into training/validation/testing sets here.
        :param X_train: the training data for the model.
        :param y_train:  the training labels for the model.
        :return: tuple of four arrays containing training data, training labels,
                     validating data and validating labels.
    """
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=2275)
    print("train datasets", X_train.shape, y_train.shape)
    print("validation datasets", X_valid.shape, y_valid.shape)

    return X_train, y_train, X_valid, y_valid

def display(sess, x, X_train, y_train, preds):
    """
        Pick 10 random images and print prediction labels and true labels.
        :param sess: the TF session.
        :param x: the input placeholder for GTSRB.
        :param X_train: the training data for the oracle.
        :param Y_train: the training labels for the oracle.
        :param preds: model output predictions label. preds = tf.argmax(logits, 1)
        :return:
    """
    sample_indexes = random.sample(range(len(X_train)), 10)
    sample_images = [X_train[i] for i in sample_indexes]
    sample_labels = [y_train[i] for i in sample_indexes]

    # Run the "predicted_labels" op.
    predicted = sess.run([preds], feed_dict={x:sample_images})[0]
    print(sample_labels)
    print(predicted)

def display_visual(sess, x, X_train, y_train, preds):
    """
        Display the prediction result visually.
        :param sess: the TF session.
        :param X_train: the training data for the oracle.
        :param Y_train: the training labels for the oracle.
        :param preds: model output predictions label.
        :return:
    """
    # Pick 10 random images
    sample_indexes = random.sample(range(len(X_train)), 10)
    sample_images = [X_train[i] for i in sample_indexes]
    sample_labels = [y_train[i] for i in sample_indexes]

    # Run the "predicted_labels" op.
    predicted = sess.run([preds], feed_dict={x: sample_images})[0]

    # Display prediction result visually.
    fig = plt.figure(figsize=(10, 10))
    for i in range(len(sample_images)):
        truth = sample_labels[i]
        prediction = predicted[i]
        # print('truth:', truth, '\nprediction:',prediction)
        plt.subplot(5, 2, 1+i)
        plt.axis('off')
        color = 'green' if truth == prediction else 'red'
        plt.text(40, 10, 'Truth: {0}\nPrediction: {1}'.format(truth, prediction),fontsize=12, color=color)
        plt.imshow(sample_images[i])
    plt.show()

def display_gray(X_train, X_test):
    """
        Display the training sample and testing sample visually.
        :param sess: the TF session.
        :param X_test: the testing data for the oracle.
        :param X_train: the training data for the oracle.
        :return:
    """
    sample_indexes = random.sample(range(len(X_train)), 10)
    training_sample = [X_train[i] for i in sample_indexes]
    testing_sample = [X_test[i] for i in sample_indexes]
    f, axarr = plt.subplots(2, 10)
    for j in range(10):
        axarr[0, j].imshow(np.reshape(training_sample[j], (28, 28)))
        axarr[1, j].imshow(np.reshape(testing_sample[j], (28, 28)))
        plt.pause(0.01)
    plt.show()


def display_leg_adv_sample(X_test, X_adv):
    """
        Display the testing sample and adversarial sample visually.
        :param sess: the TF session.
        :param X_test: the testing data for the oracle.
        :param Y_adv: the adversarial data for the oracle.
        :param preds: model output predictions label.
        :return:
    """
    # X_test_redu = (0.5 + tf.reshape(X_test, ((X_test.shape[0], 32, 32, 3)))) * 255
    # X_train, y_train, X_test1, y_test = ld.load_data(train_data_dir, test_data_dir)
    # X_test_redu = reduction_batch(X_test)
    # X_adv_redu = reduction_batch(X_adv)

    # Pick 6 random images

    sample_indexes = random.sample(range(len(X_test)), 10)

    legitimate_sample = [X_test[i] for i in sample_indexes]
    adversarial_sample = [X_adv[i] for i in sample_indexes]
    f, axarr = plt.subplots(3, 10)
    for j in range(10):
        axarr[0, j].imshow(np.reshape(legitimate_sample[j],(28,28)), cmap='gray')
        axarr[1, j].imshow(np.reshape(adversarial_sample[j],(28,28)), cmap='gray')
        axarr[2, j].imshow(np.reshape(adversarial_sample[j]-legitimate_sample[j],(28,28)), cmap='gray')
        # plt.setp(axarr[0, j].get_xticklabels(), visible=False)
        # plt.setp(axarr[1, j].get_yticklabels(), visible=False)
        plt.pause(0.01)
    plt.show()

def pair_visual(original, adversarial, figure=None):
    """
    This function displays two images: the original and the adversarial sample
    :param original: the original input
    :param adversarial: the input after perterbations have been applied
    :param figure: if we've already displayed images, use the same plot
    :return: the matplot figure to reuse for future samples
    """
    import matplotlib.pyplot as plt

    # Ensure our inputs are of proper shape
    assert(len(original.shape) == 2 or len(original.shape) == 3)

    # To avoid creating figures per input sample, reuse the sample plot
    if figure is None:
        plt.ion()
        figure = plt.figure()
        figure.canvas.set_window_title('Cleverhans: Pair Visualization')

    # Add the images to the plot
    perterbations = adversarial - original
    for index, image in enumerate((original, perterbations, adversarial)):
        figure.add_subplot(1, 3, index + 1)
        plt.axis('off')

        # If the image is 2D, then we have 1 color channel
        if len(image.shape) == 2:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)

        # Give the plot some time to update
        plt.pause(0.01)

    # Draw the plot and return
    plt.show()
    return figure

def display_spec_label(X_train, y_train, label):
    """
        Display an image with certain specific label visually.
        :param X_train: the training data for the oracle.
        :param Y_train: the training labels for the oracle.
        :param Label: Given label.
        :return:
    """
    for i in np.where(y_train == label):
        res_27 = i[:3]
        for i in range(3):
            image = X_train[res_27[i]].squeeze()
            plt.figure(figsize=(1, 1))
            plt.imshow(image)
            print(y_train[res_27[i]])

def rgb_convert_grayscale(X_train, X_test):
    """
        Convert to grayscale from RGB
        :param X_train: the training data for the oracle.
        :param Y_train: the training labels for the oracle.
        :param Label: Given label.
        :return:
    """
    train_imgs = rgb2gray(X_train)
    test_imgs = rgb2gray(X_test)
    # Reshape images to [num_samples, rows*columns]
    train_imgs = train_imgs.reshape(train_imgs.shape[0], train_imgs.shape[1] * train_imgs.shape[2])
    test_imgs = test_imgs.reshape(test_imgs.shape[0], test_imgs.shape[1] * test_imgs.shape[2])

    return train_imgs, test_imgs

if __name__ == "__main__":
    ROOT_PATH = "../GTSRB"
    SAVE_PATH = "../GTSRB/models"
    train_data_dir = os.path.join(ROOT_PATH, "train.p")
    test_data_dir = os.path.join(ROOT_PATH, "test.p")
    load_data(train_data_dir, test_data_dir)