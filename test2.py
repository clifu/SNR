import keras
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras.applications import VGG16, Xception, InceptionV3
from keras.applications.vgg16 import preprocess_input

input_shape = (256, 256, 3)
target_size = (256, 256)
batch_size = 128
lr = 0.0001
epochs = 1
preprocess_function = preprocess_input


def print_model_info(_conv_base, _model):
    print('CONV_BASE.SUMMARY:')
    _conv_base.summary()

    print('CONV_BASE LAYERS INFO:')
    for i, layer in enumerate(_conv_base.layers):
        print(i, ')', layer, layer.trainable)

    print('MODEL.SUMMARY:')
    _model.summary()
    return


def get_vgg16_fine_tune_model(_input_shape, _num_classes):
    _conv_base = VGG16(weights='imagenet', include_top=False,
                       input_shape=_input_shape)
    _conv_base.trainable = True

    for _i, _layer in enumerate(_conv_base.layers):
        if _i < 15:
            _layer.trainable = False
        else:
            _layer.trainable = True

    _model = models.Sequential()
    _model.add(_conv_base)

    # add a global spatial average pooling layer
    _model.add(layers.GlobalAveragePooling2D())

    # let's add a fully-connected layer
    _model.add(layers.Dense(1024, activation='relu'))

    # and a logistic layer -- let's say we have 200 classes
    _model.add(layers.Dense(_num_classes, activation='softmax'))

    print_model_info(_conv_base, _model)
    return _model


datapath = os.path.abspath(
    "../SNR/stanford-car-dataset-by-classes-folder")
dirInfo = os.listdir(datapath)

train_dir = datapath + '\\car_data\\car_data\\train'
validation_dir = datapath + '\\car_data\\car_data\\test'

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

num_classes = len(train_generator.class_indices)
model = get_vgg16_fine_tune_model(input_shape, num_classes)
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])

train_steps = len(train_generator.filenames) // batch_size
validation_steps = len(validation_generator.filenames) // batch_size
history = model.fit_generator(
    train_generator,
    steps_per_epoch=10,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

validation_score = model.evaluate_generator(
    validation_generator, steps=validation_steps)

x, y = train_generator.next()

pred = model.predict(x)

print(np.argmax(pred[0]))

print('Validation loss: ', validation_score[0])
print('Validation acc:  ', validation_score[1])
