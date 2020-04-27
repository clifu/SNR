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
from IPython.display import display
from sklearn.svm import SVC
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics

batch_size = 128
epochs = 1
model = load_model('C:/Users/Clifu/Desktop/SNR/fine_tune_vgg16.h5')

preprocess_function = preprocess_input
target_size = (256, 256)
batch_size = 128
datapath = os.path.abspath(
    "../SNR/stanford-car-dataset-by-classes-folder")
validation_dir = datapath + '\\car_data\\car_data\\test'
train_dir = datapath + '\\car_data\\car_data\\train'

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_function,
    rescale=1. / 255,
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

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_function, rescale=1. / 255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_steps = len(validation_generator.filenames) // batch_size

model.fit_generator(
    train_generator,
    steps_per_epoch=10,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

x, y = validation_generator.next()
pred = model.predict(x)

print('Klasy')
print(validation_generator.classes)
print('Predictions')
for i in range(0, 128):
    if np.argmax(y[i]) == np.argmax(pred[i]):
        print(str(i) + 'true')
    else:
        print(str(i) + 'false')

for i in range(0, 128):
    for key, val in validation_generator.class_indices.items():
        if val == np.argmax(pred[i]):
            print(key)
