import json
import operator
import os

import keras
import keras.utils as ku
import numpy as np
import pandas as pd
from IPython.display import display
from keras import layers
from keras.applications.vgg16 import (VGG16, decode_predictions,
                                      preprocess_input)
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import (ImageDataGenerator, img_to_array,
                                       load_img)
from sklearn import metrics, svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

batch_size = 512
target_size = (224, 224)
preprocess_function = preprocess_input
current_dir = os.getcwd() + r"\SNR\stanford_car_dataset_by_classes"
train_dir = current_dir + r"\train"
test_dir = current_dir + r"\test"

model = load_model('C:/Users/Clifu/Desktop/SNR/SNR/learned_vgg16.h5')
config = model.get_config()
weights = model.get_weights()
model.summary()

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

# Creating new base model using trained weights from
base_model = VGG16(input_shape=(224, 224, 3))
base_model.load_weights('wagi.h5', by_name=True)
optimizer = keras.optimizers.RMSprop(lr=0.0001)
model = Model(inputs=base_model.input,
              outputs=base_model.get_layer('fc1').output)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Getting data
x, y = train_generator.next()

# Getting features for SVMs
feature = model.predict(x, verbose=0)
X_train, X_test, y_train, y_test = train_test_split(
    feature, y)

# Creating classificators - SVMs
classificators = [
    svm.SVC(kernel='linear'),
    svm.SVC(kernel='poly',
            degree=2),
    svm.SVC(kernel='rbf')]
classifactors_names = ['linear', 'quadratic', 'rbf']
idx = 0

for clf in classificators:
    print(classifactors_names[idx])
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
    print('Accuracy for ' + classifactors_names[idx] + ' : ' + str(acc))
    idx = idx + 1
