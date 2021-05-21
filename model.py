#%%
from tensorflow.keras import layers
import os

from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps

from keras.utils.vis_utils import plot_model

from tensorflow import keras 
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from utils import read_dataset

import warnings

import tensorflow as tf

from skimage.io import imsave

from model_utils import get_model, f1_m, recall_m, precision_m
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # To disable using GPU
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')



from utils import read_dataset

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()


img_size = (256, 256)
num_classes = 2
batch_size = 32
# Build model
model = get_model(img_size, num_classes)

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer="sgd", loss=tf.losses.SigmoidFocalCrossEntropy(), metrics=[f1_m])

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
