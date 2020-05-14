from vis.visualization import visualize_cam, overlay
from vis.utils import utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras import activations
from keras.preprocessing.image import load_img
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
import keras.utils as ku
import os
import numpy as np
import json
from PIL import Image
from keras import layers
import keras
from IPython.display import display
from sklearn import svm
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score
import operator
from sklearn.model_selection import train_test_split
import pandas as pd
from keras import backend as k
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import keract

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

classificators = [
    svm.SVC(kernel='linear'),
    svm.SVC(kernel='poly',
            degree=2),
    svm.SVC(kernel='rbf')]
classifactors_names = ['linear', 'quadratic', 'rbf']
indexxxxxx = 0
for clf in classificators:
    print(classifactors_names[indexxxxxx])
    abc = []

    for a in y_train:
        index, value = max(enumerate(a), key=operator.itemgetter(1))
        abc.append(index)

    clf.fit(X_train, abc)
    y_pred = clf.predict(X_test)

    abc = []

    for a in y_test:
        index, value = max(enumerate(a), key=operator.itemgetter(1))
        abc.append(index)

    report = classification_report(abc, y_pred)
    print(report)

    acc = accuracy_score(abc, y_pred)
    print('Accuracy for ' + classifactors_names[indexxxxxx] + ' : ' + str(acc))
    indexxxxxx = indexxxxxx + 1
