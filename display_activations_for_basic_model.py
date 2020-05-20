import json
import operator
import os

import cv2
import keract
import keras
import keras.utils as ku
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from keras import activations
from keras import backend as k
from keras import layers
from keras.applications.vgg16 import (VGG16, decode_predictions,
                                      preprocess_input)
from keras.models import Model, Sequential, load_model
from keras.preprocessing.image import (ImageDataGenerator, img_to_array,
                                       load_img)
from matplotlib.pyplot import imshow
from PIL import Image
from sklearn import metrics, svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from vis.utils import utils
from vis.visualization import overlay, visualize_cam

batch_size = 1
target_size = (224, 224)

k.clear_session()
model = load_model('C:/Users/Clifu/Desktop/SNR/SNR/learned_vgg16.h5')
model.save_weights('wagi.h5')
config = model.get_config()
weights = model.get_weights()
model.summary()

preprocess_function = preprocess_input
current_dir = os.getcwd() + r"\SNR\stanford_car_dataset_by_classes"
train_dir = current_dir + r"\train"
test_dir = current_dir + r"\test"

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_function,
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
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_function,
    rescale=1. / 223)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

x, y = test_generator.next()
feature = model.predict(x, verbose=1)

test_model = model.layers[0]
test_model.compile(loss='categorical_crossentropy',
                   optimizer=keras.optimizers.RMSprop(),
                   metrics=['accuracy'])
activations = keract.get_activations(
    test_model, x, auto_compile=True)
keract.display_activations(activations, save=False)
