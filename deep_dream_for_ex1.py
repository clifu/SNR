import tensorflow as tf

import numpy as np
import os

import matplotlib.pyplot as plt

import IPython.display as display
import PIL.Image

from tensorflow.keras.preprocessing import image
from tensorflow_core.python.keras.models import load_model


def deprocess(img):
    img = 224*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)


def show(img):
    plt.imshow(img)
    plt.show()


current_dir = os.getcwd() + r"\SNR\stanford_car_dataset_by_classes"
image_path = current_dir + r"\test\Aston Martin V8 Vantage Convertible 2012\01633.jpg"

im = PIL.Image.open(image_path)
sqrWidth = np.ceil(np.sqrt(im.size[0]*im.size[1])).astype(int)
im_resize = im.resize((sqrWidth, sqrWidth))
im_resize.thumbnail((224, 224))
original_img = np.array(im_resize)
show(original_img)

base_model = load_model('learned_vgg16.h5').get_layer('vgg16')
base_model.summary()

# Maximize the activations of these layers
names = ['block4_pool']  # 92160
layers = [base_model.get_layer(name).output for name in names]

# Create the feature extraction model
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)


def calc_loss(img, model):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return tf.reduce_sum(losses)


class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # This needs gradients relative to img
                # GradientTape only watches tf.Variables by default
                tape.watch(img)
                loss = calc_loss(img, self.model)

            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)

            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

        return loss, img


deepdream = DeepDream(dream_model)


def run_deep_dream_simple(img, steps=100, step_size=0.01):
    # Convert from uint8 to the range expected by the model.
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining > 100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps

        loss, img = deepdream(img, run_steps, tf.constant(step_size))

        display.clear_output(wait=True)
        show(deprocess(img))
        print("Step {}, loss {}".format(step, loss))

    result = deprocess(img)
    display.clear_output(wait=True)
    show(result)

    return result


dream_img = run_deep_dream_simple(
    img=original_img, steps=100, step_size=0.01)
