import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from metrics import f1_m
from losses import weighted_cce
import numpy as np

class UNet_Like():
    def __init__(self, load=False, manual=False):
        if load == True:
            self.load_best()
        else:
            if not manual:
                self.build()
                self.compile()
    
    def compile(self, 
            optimizer="adam", 
            loss=weighted_cce(np.array([1, 15])), 
            metrics=[f1_m]
            ):
        if self.model is not None:
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def load_best(self, 
            filepath='model_defs/Unet_Like.h5', 
            custom_objects={"f1_m": f1_m,
            "loss": weighted_cce(np.array([1, 15]))}
            ):
        self.model = keras.models.load_model(filepath, custom_objects=custom_objects)

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, 
            x_train, y_train, 
            filepath="model_defs/UNet_Like.h5", 
            epochs=100, 
            validation_data=None, 
            batch_size=32):
        callbacks = [keras.callbacks.ModelCheckpoint(filepath, save_best_only=True)]
        self.model.fit(x_train,
         y_train,
         epochs=epochs,
         validation_data=validation_data,
         batch_size=batch_size, callbacks=callbacks
         )

    def build(self, img_size=(256, 256, 1), 
            num_classes=2):
        inputs = keras.Input(shape=img_size)

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
        self.model = keras.Model(inputs, outputs)
        

