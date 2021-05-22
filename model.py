#%%
from tensorflow.keras import layers
import os

from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps


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

# %%
x_train, x_test, y_train, y_test = read_dataset()
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
from functools import partial
from tensorflow.keras import activations
from itertools import product

# %%
model = get_model(img_size, num_classes)
#model.compile(optimizer="sgd",loss=tf.keras.losses.CategoricalHinge(), metrics=[f1_m])
from losses import weighted_cce
weights = np.array([1, 50])
opt = tf.keras.optimizers.SGD(learning_rate=0.05,  name="SGD")
model.compile(optimizer=opt,loss=weighted_cce(weights), metrics=[f1_m])

callbacks = [
    keras.callbacks.ModelCheckpoint("unet_skel.h5", save_best_only=True)
]
# Train the model, doing validation at the end of each epoch.
# %%
epochs = 200
model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test,y_test), callbacks=callbacks, batch_size=32)

# %%
modelFile = "unet_skel.h5"
new_model = tf.keras.models.model_from_json(open(modelFile).read())
new_model.load_weights(os.path.join(os.path.dirname(modelFile), 'model_weights.h5'))
# %%
def loss(y_true, y_pred):
    weights = K.variable( np.array([1, 50]))
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss        

w_loss = partial(weighted_cce, weights)
new_model = keras.models.load_model('unet_skel.h5', custom_objects={"loss":loss,"f1_m": f1_m})
new_model.fit(x_train, y_train, epochs=200, validation_data=(x_test,y_test), callbacks=callbacks, batch_size=32)
# %%
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
