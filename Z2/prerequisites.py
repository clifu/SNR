import os
from os.path import isfile, join
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from tensorflow.keras import callbacks
from keras import optimizers
from keras.applications import VGG16
from keras.applications import vgg16
from keras import backend as K
import pickle
import numpy as np
import gc
from random import shuffle
from math import floor
from shutil import move


INPUT_SHAPE = (224, 224, 3)
TARGET_SIZE = (224, 224)
PREPROCESS_FUNCTION = vgg16.preprocess_input


def get_data_dir():
    return os.getcwd() + r"\..\data\stanford_car_dataset_by_classes"


TRAIN_DIR = get_data_dir() + r"\train"
VAL_DIR = get_data_dir() + r"\validation"
TEST_DIR = get_data_dir() + r"\test"
RESULTS_DIR = os.getcwd() + r"\results"


def split_test_data(validation_test_ratio=0.5):

    os.mkdir(VAL_DIR)
    # for every subdirectory of TEST_DIR:
    for subdir, dirs, files in os.walk(TEST_DIR):
        # get file list of a single class
        file_list = [os.path.join(subdir, file) for file in files]
        if len(file_list) == 0:
            continue
        shuffle(file_list)
        split_index = floor(len(file_list) * validation_test_ratio)
        validation_data = file_list[:split_index]
        os.mkdir(subdir.replace('test', 'validation'))
        # move every file in validation_data
        for file in validation_data:
            file_in_new_location = file.replace('test', 'validation')
            move(file, file_in_new_location)


def draw_training_info(_history, title_specific_value=''):

    acc = _history.history['accuracy']
    val_acc = _history.history['val_accuracy']
    loss = _history.history['loss']
    val_loss = _history.history['val_loss']

    epochs_plot = range(1, len(acc) + 1)
    plt.plot(epochs_plot, acc, 'bo', label='Training acc')
    plt.plot(epochs_plot, val_acc, 'b', label='Validation acc')
    plt.title(f'Training and validation accuracy {title_specific_value}')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs_plot, loss, 'bo', label='Training loss')
    plt.plot(epochs_plot, val_loss, 'b', label='Validation loss')
    plt.title(f'Training and validation loss {title_specific_value}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def get_prepared_model(class_count, first_trainable_layer='block5_conv3'):

    # get pretrained model from library
    convolutional_part = VGG16(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

    # set layers as trainable starting from first_trainable_layer
    shall_train = False
    for layer in convolutional_part.layers:
        if layer.name == first_trainable_layer:
            shall_train = True
        layer.trainable = shall_train

    # add a global spatial average pooling layer
    model = models.Sequential()
    model.add(convolutional_part)
    model.add(layers.GlobalAveragePooling2D())

    # let's add a fully-connected layer
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(1000, activation='relu'))

    # and a logistic layer
    model.add(layers.Dense(class_count, activation='softmax'))

    return model


def get_data(batch_size):

    train_datagen = ImageDataGenerator(
        preprocessing_function=PREPROCESS_FUNCTION,
        rescale=1. / 223,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=TARGET_SIZE,
        batch_size=batch_size,
        class_mode='categorical'
    )

    class_count = train_generator.num_classes

    val_datagen = ImageDataGenerator(
        preprocessing_function=PREPROCESS_FUNCTION,
        rescale=1. / 223)

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=TARGET_SIZE,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=PREPROCESS_FUNCTION,
        rescale=1. / 223)

    test_generator = test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=TARGET_SIZE,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return class_count, train_datagen, train_generator, val_datagen, val_generator, test_datagen, test_generator
