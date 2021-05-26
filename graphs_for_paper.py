# %%

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # To disable using GPU

import warnings

import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import cv2

from losses import weighted_cce
from metrics import f1_m, f1_i, template_matching_i
from model_defs.UNet_MoreLike import UNet_MoreLike
from utils import (collapse_dim, show, load_data, read_dataset, reshape_target, dist_transform,
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
TRAIN_3 = False

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

x_train, x_val, y_train, y_val, names_train, names_val, x_test, names_test = load_data(DIST=False, NO_TEST=NO_TEST)

# write_imgs(x_val, names_val, 'val_shapes')
# write_imgs(x_val, names_val, 'val_shapes')
# write_imgs(y_val, names_val, 'Y_target')

# Build 2 models
model1 = UNet_MoreLike('unet_thick_stage1', loss=weighted_cce(np.array([1, 17])), load= not TRAIN_1)
model2 = UNet_MoreLike('unet_thick_stage2', loss=weighted_cce(np.array([1, 17])), load= not TRAIN_2)
# model3 = UNet_MoreLike('unet_more_like_stage3', loss=weighted_cce(np.array([1, 17])), load= not TRAIN_3)


I1 = model1.predict(x_val)
I2 = model2.predict(I1)
F1, F1_scores = f1_i(y_val, I2)
CORR, CORR_ls = template_matching_i(y_val, I2)

for i in range(len(x_val)):
    images = [x_val[i], collapse_dim(y_val[i]), I2[i]]
    titles = ['Original image', 'Target Skeleton', 'Predicted Skeleton: f1='+str(F1_scores[i])[0:5]+", corr="+str(CORR_ls[i])[0:5]]
    show(1, 3, images, titles, save=True, path="graphs/f1_corr"+str(i)+".png")

