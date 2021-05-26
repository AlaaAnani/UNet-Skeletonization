import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # To disable using GPU

import warnings

import numpy as np
import tensorflow as tf
from skimage.io import imsave
from tensorflow import keras
import matplotlib.pyplot as plt


from losses import weighted_cce
from metrics import f1_m, f1_i
from model_defs.UNet_MoreLike import UNet_MoreLike
from model_defs.UNet_Thick import UNet_Thick

from utils import (collapse_dim, load_data, read_dataset, reshape_target, dist_transform,
                   write_imgs)

tf.get_logger().setLevel('FATAL')
tf.autograph.set_verbosity(1)
warnings.filterwarnings('ignore')
tf.random.set_seed(73)

weights = np.array([1, 15])
img_size = (256, 256)
num_classes = 2
batch_size = 32


NO_TEST = True
SHOW_VAL = True
TRAIN_1 = False
TRAIN_2 = False

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

x_train, x_val, y_train, y_val, names_train, names_val, x_test, names_test = load_data(DIST=False, NO_TEST=NO_TEST)

# write_imgs(x_val, names_val, 'val_shapes')
# write_imgs(x_val, names_val, 'val_shapes')
# write_imgs(y_val, names_val, 'Y_target')

# Build 2 models
model1 = UNet_Thick('unet_thick1', loss=weighted_cce(np.array([1, 25])), load= not TRAIN_1)
model2 = UNet_Thick('unet_thick2', loss=weighted_cce(np.array([1, 25])), load= not TRAIN_2)

if TRAIN_1:
    model1.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=40)
    model1.load_best()

if TRAIN_2:
    x_train2 = model1.predict(x_train)

    x_val2 = model1.predict(x_val)

    model2.fit(x_train2, y_train, validation_data=(x_val2, y_val), epochs=40)
    model2.load_best()

if SHOW_VAL:
    I1 = model1.predict(x_val)
    I2 = model2.predict(I1)

    write_imgs(I1, names_val, 'I1')
    write_imgs(collapse_dim(y_val), names_val, 'y_val')
    write_imgs(I2, names_val, 'I2')
    write_imgs(x_val, names_val, 'x_val')