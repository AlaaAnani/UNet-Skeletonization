#%%
import os
import warnings

import numpy as np
import PIL
import tensorflow as tf
tf.random.set_seed(73)


from IPython.display import Image, display
# from keras.utils.vis_utils import plot_model
from PIL import ImageOps
from skimage.io import imsave
from tensorflow import keras
from tensorflow.keras import layers

from metrics import f1_m
from utils import read_dataset, reshape_target, collapse_dim, write_imgs
from losses import weighted_cce
weights = np.array([1, 15])


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # To disable using GPU
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)
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



# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

x_train, x_test, y_train, y_test, names_train, names_test = read_dataset()
y_train_r, y_test_r = reshape_target(y_train), reshape_target(y_test)

write_imgs(x_test, names_test, 'test_shapes')
write_imgs(y_test, names_test, 'Y_target')
#%%
img_size = (256, 256)
num_classes = 2
batch_size = 32
# Build model
model1 = get_model_unet(img_size, num_classes)
model2 = get_model_unet(img_size, num_classes)

model1.compile(optimizer="adam", loss=weighted_cce(weights), metrics=[f1_m])
model2.compile(optimizer="adam", loss=weighted_cce(weights), metrics=[f1_m])


callbacks1 = [
    keras.callbacks.ModelCheckpoint("unet_skel_init.h5", save_best_only=True)
]

callbacks2 = [
    keras.callbacks.ModelCheckpoint("unet_skel_next.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 64

#%%


# model1.fit(x_train, y_train_r, epochs=epochs, validation_data=(x_test,y_test_r), callbacks=callbacks1, batch_size=32)
model1 = keras.models.load_model('unet_skel_init.h5', custom_objects={"f1_m": f1_m, "loss": weighted_cce(weights)})

#%%

x_train_new = model1.predict(x_train)
x_test_new = model1.predict(x_test)

x_train_new = collapse_dim(x_train_new)
x_test_new = collapse_dim(x_test_new)


# model2.fit(x_train_new, y_train_r, epochs=epochs, validation_data=(x_test_new, y_test_r), callbacks=callbacks2, batch_size=32)


model2 = keras.models.load_model('unet_skel_next.h5', custom_objects={"f1_m": f1_m, "loss": weighted_cce(weights)})

Y = model2.predict(x_test_new)

write_imgs(Y, names_test, 'Y_pred_new', collapse=True)


# %%
