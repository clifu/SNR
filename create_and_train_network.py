import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models
from keras.applications import VGG16, InceptionV3, Xception
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

input_shape = (224, 224, 3)
target_size = (224, 224)
batch_size = 64
learning_rate = 0.0001
epochs = 70
preprocess_function = preprocess_input

current_dir = os.getcwd() + r"\SNR\stanford_car_dataset_by_classes"
train_dir = current_dir + r"\train"
test_dir = current_dir + r"\test"


def draw_training_info(_history):
    acc = _history.history['accuracy']
    val_acc = _history.history['val_accuracy']
    loss = _history.history['loss']
    val_loss = _history.history['val_loss']

    epochs_plot = range(1, len(acc) + 1)
    plt.plot(epochs_plot, acc, 'bo', label='Training acc')
    plt.plot(epochs_plot, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs_plot, loss, 'bo', label='Training loss')
    plt.plot(epochs_plot, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    return


def print_model_info(_conv_base, _model):
    print('CONV_BASE.SUMMARY:')
    _conv_base.summary()

    print('CONV_BASE LAYERS INFO:')
    for i, layer in enumerate(_conv_base.layers):
        print(i, ')', layer, layer.trainable)

    print('MODEL.SUMMARY:')
    _model.summary()
    return


def get_preapared_vgg16_model(_input_shape, _num_classes):
    _conv_base = VGG16(weights='imagenet', include_top=False,
                       input_shape=_input_shape)
    _conv_base.trainable = False

    for _i, _layer in enumerate(_conv_base.layers):
        _layer.trainable = False

    _model = models.Sequential()
    _model.add(_conv_base)

    # create classification layers
    _model.add(layers.Flatten(name='flatten'))
    _model.add(layers.Dense(4096, activation='relu'))
    _model.add(layers.Dense(4096, activation='relu'))
    _model.add(layers.Dense(_num_classes, activation='softmax'))

    print_model_info(_conv_base, _model)
    return _model


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
num_classes = train_generator.num_classes

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_function,
    rescale=1. / 223)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

model = get_preapared_vgg16_model(input_shape, num_classes)

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
              metrics=['accuracy'])

train_steps = len(train_generator.filenames) // batch_size
test_steps = len(test_generator.filenames) // batch_size

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_steps
)

model.save('learned_vgg16.h5')

validation_score = model.evaluate_generator(
    test_generator, steps=test_steps)
print('Validation loss: ', validation_score[0])
print('Validation acc:  ', validation_score[1])

draw_training_info(history)
