import argparse
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

def set_gtsrb_flags():
    try:
        flags.DEFINE_integer('BATCH_SIZE', 64, 'Size of training batches')
    except argparse.ArgumentError:
        pass

    flags.DEFINE_integer('NUM_CLASSES', 43, 'Number of classification classes')
    flags.DEFINE_integer('IMAGE_ROWS', 32, 'Input row dimension')
    flags.DEFINE_integer('IMAGE_COLS', 32, 'Input column dimension')
    flags.DEFINE_integer('NUM_CHANNELS', 3, 'Input depth dimension')


def modelA():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(32, 32, 3),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))
    return model

def modelB():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(FLAGS.NUM_CLASSES, activation='softmax'))
    return model

def model_gtsrb(type=1):
    """
    Defines GTSRB model using Keras sequential model
    """

    models = [modelA, modelB]

    return models[type]()


def data_gen_gtsrb(X_train):
    datagen = ImageDataGenerator(featurewise_center=False,
                                 featurewise_std_normalization=False,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 zoom_range=0.2,
                                 shear_range=0.1,
                                 rotation_range=10., )

    datagen.fit(X_train)
    return datagen

def load_model(model_path, type=1):

    try:
        with open(model_path+'.json', 'r') as f:
            json_string = f.read()
            model = model_from_json(json_string)
    except IOError:
        model = model_gtsrb(type=type)

    model.load_weights(model_path)
    return model