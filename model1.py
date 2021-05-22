#%%
import os
import warnings

import numpy as np
import PIL
import tensorflow as tf
from IPython.display import Image, display
# from keras.utils.vis_utils import plot_model
from PIL import ImageOps
from skimage.io import imsave
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img

from metrics import f1_m
from losses import weighted_cce
from utils import read_dataset

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # To disable using GPU
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

#%%

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (1,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

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


from tensorflow.keras import backend as K
from utils import read_dataset

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()


img_size = (256, 256)
num_classes = 2
batch_size = 32
# Build model
model = get_model_unet(img_size, num_classes)

# plot_model(model, to_file='model_plot_unet.png', show_shapes=True, show_layer_names=True)
weights = np.array([0.75, 50])
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=[f1_m])


callbacks = [
    keras.callbacks.ModelCheckpoint("unet_skel.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 100

x_train, x_test, y_train, y_test = read_dataset()


model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test,y_test), callbacks=callbacks, batch_size=32)



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




# %%
