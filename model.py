#%%
from tensorflow.keras import layers
import os

from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img


from tensorflow import keras 
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from utils import read_dataset

import warnings

import tensorflow as tf

from skimage.io import imsave

from model_utils import get_model, f1_m, recall_m, precision_m
from utils import read_dataset

# %%
x_train, x_test, y_train, y_test, img_names = read_dataset()
def reshape_target(target):
    new_y_ls = []
    for y in target:
        zeros = y==0
        ones = y==1
        new_y = np.zeros((y.shape[0], y.shape[1], 2))
        new_y[:, :, 0][zeros] = 1
        new_y[:, :, 1][ones] = 1
        new_y_ls.append(new_y)
    return np.array(new_y_ls)

y_train, y_test = reshape_target(y_train), reshape_target(y_test)

# %%
# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

img_size = (256, 256)
num_classes = 2
batch_size = 32
# Build model
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
from tensorflow.keras import backend as K

def f1_m(y_true, y_pred):

    mask = tf.math.argmax(y_pred, axis=-1)
    y_pred = tf.expand_dims(mask, axis=-1)
    y_pred = tf.cast(y_pred, float)

    mask = tf.math.argmax(y_true, axis=-1)
    y_true = tf.expand_dims(mask, axis=-1)
    y_true = tf.cast(y_true, float)

    y_pred = tf.reshape(y_pred, shape=(tf.shape(y_pred)[0], 256,256))
    y_true = tf.reshape(y_true, shape=(tf.shape(y_true)[0], 256,256))

    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_model_unet(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (1,))

    prev_layers = {}
    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # x = layers.Conv2D(32, 3, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)

    prev_layers[32] = x

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        # x = layers.Activation("relu")(x)
        # x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        # x = layers.BatchNormalization()(x)

        prev_layers[filters] = x

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        # x = layers.Activation("relu")(x)
        # x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        # x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        # residual = layers.UpSampling2D(2)(prev_layers[filters])
        residual = prev_layers[filters]
        # residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.concatenate([residual, x])  # Add back residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 1, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# %%
model = get_model_unet(img_size, num_classes)
#model.compile(optimizer="sgd",loss=tf.keras.losses.CategoricalHinge(), metrics=[f1_m])
from losses import weighted_cce
weights = np.array([0.75, 50])
model.compile(optimizer="adam",loss=weighted_cce(weights), metrics=[f1_m])

callbacks = [
    keras.callbacks.ModelCheckpoint("unet_skel.h5", save_best_only=True)
]
# Train the model, doing validation at the end of each epoch.
epochs = 100
model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test,y_test), callbacks=callbacks, batch_size=32)

# %%

new_model = keras.models.load_model('unet_skel.h5', custom_objects={"f1_m": f1_m})

Y = new_model.predict(x_test)

#%%
print(Y[0].shape)

#%%%
for i, y in enumerate(Y):
    print()
    mask = np.argmax(y, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    imsave(f'Y_val/{i}.png', mask)



#model.summary()