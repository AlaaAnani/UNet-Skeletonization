

class UNet_MoreLike():

    def __init__(self, img_size=(256,256), num_classes=2):
        self.model = self.get_model(num_classes)

    def get_model(img_size, num_classes):
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

    def train(self, x_train, y_train, val_data):
        
