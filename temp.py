# %%
from utils import read_dataset

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
import numpy as np
x_train, x_test, y_train, y_test, _, _ = read_dataset()
x_test, x_train, y_train, y_test = reshape_target(x_test), reshape_target(x_train), reshape_target(y_train), reshape_target(y_test)
from tensorflow.keras.models import load_model
from utils import collapse_dim
new_model = load_model('model_pix2pix.h5')

Y = new_model.predict(x_test)
from skimage.io import imsave
for i, y in enumerate(Y):
	y = collapse_dim(y)
	imsave(f'Y_pix2pix_val/{i}.png', y)
# %%
