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

CLASS_INDEX = None
CLASS_INDEX_PATH = 'C:/Users/Clifu/Desktop/SNR/classes_index.json'


def my_decode_predictions(preds, top=1):
    global CLASS_INDEX

    if CLASS_INDEX is None:
        with open(CLASS_INDEX_PATH) as f:
            CLASS_INDEX = json.load(f)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


batch_size = 128

model = load_model('C:/Users/Clifu/Desktop/SNR/fine_tune_vgg16.h5')
config = model.get_config()
weights = model.get_weights()

model_feat = Model(inputs=model.input,
                   outputs=model.get_layer('dense_1').output)

preprocess_function = preprocess_input
target_size = (256, 256)
batch_size = 128
datapath = os.path.abspath(
    "../SNR/stanford-car-dataset-by-classes-folder")
validation_dir = datapath + '\\car_data\\car_data\\test'

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_function, rescale=1. / 255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

names = ['C:/Users/Clifu/Desktop/SNR/00019.jpg',
         'C:/Users/Clifu/Desktop/SNR/00128.jpg']
dataset_dir = os.path.abspath(
    '../SNR/stanford-car-dataset-by-classes-folder/car_data/car_data/test')
classes, classes_idx = find_classes(dataset_dir)

for i in range(0, 100):
    # # load an image from file
    # image = load_img(n, target_size=(256, 256))
    # # convert the image pixels to a numpy array
    # image = img_to_array(image)
    # # reshape data for the model
    # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # # prepare the image for the VGG model
    # image = preprocess_input(image)
    # # predict the probability across all output classes
    #yhat = model.predict_proba(image)
    x, y = validation_generator.next()
    pred = model.predict(x)
    print(np.argmax(pred[0]))
    feat_train = model_feat.predict(x)
    svm = SVC(kernel='linear')
    svm.fit(feat_train, np.argmax(y, axis=1))
    print(svm.score(feat_train, np.argmax(y, axis=1)))
