# %%

import os
import warnings

import numpy as np
import tensorflow as tf
from skimage.io import imsave
from tensorflow import keras

from losses import weighted_cce
from metrics import f1_m
from model_defs.UNet_MoreLike import UNet_MoreLike
from post_process import dist_transform
from utils import (collapse_dim, load_data, read_dataset, reshape_target,
                   write_imgs)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # To disable using GPU
tf.get_logger().setLevel('FATAL')
tf.autograph.set_verbosity(1)
warnings.filterwarnings('ignore')
tf.random.set_seed(73)

weights = np.array([1, 15])
img_size = (256, 256)
num_classes = 2
batch_size = 32


NO_TEST = True
TRAIN_1 = True
TRAIN_2 = True

# Free up RAM in case the model definition cells were run multiple times
# keras.backend.clear_session()

x_train, x_val, y_train, y_val, names_train, names_val, x_test, names_test = load_data(NO_TEST=NO_TEST)

# write_imgs(x_val, names_val, 'val_shapes')
# write_imgs(x_val, names_val, 'val_shapes')
# write_imgs(y_val, names_val, 'Y_target')

# Build 2 models
model1 = UNet_MoreLike(load= ~TRAIN_1)
model2 = UNet_MoreLike(load= ~TRAIN_2)

if TRAIN_1:
    model1.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=50)
    model1.load_best()

x_train2 = collapse_dim(model1.predict(x_train))
x_val2 = collapse_dim(model1.predict(x_val))

if TRAIN_2:
    model2.fit(x_train2, y_train, validation_data=(x_val2, y_val), epochs=50)
    model2.load_best()


Y_pred_val = model2.predict(x_val2)
write_imgs(Y_pred_val, names_val, 'Y_pred_val', collapse=True)

if NO_TEST is False:
    x_test2 = collapse_dim(model1.predict(x_test))
    Y_pred_test = model2.predict(x_test2)
    write_imgs(Y_pred_test, test_names, 'Y_pred_test', collapse=True)

# %%
