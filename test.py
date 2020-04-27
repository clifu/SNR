from keras.models import load_model
import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def print_model_info(_model):
    print('MODEL.SUMMARY:')
    _model.summary()
    return


if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
model = load_model('fine_tune_vgg16.h5')
for l in model.layers:
    print(l.trainable)
