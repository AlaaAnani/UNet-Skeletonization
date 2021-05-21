
import os
from skimage.io import imread
from sklearn.model_selection import train_test_split
import numpy as np

def read_dataset(path='dataset/'):
    xpath = f'{path}/img_train_shape'
    ypath = f'{path}/img_train_skeletons'

    img_names = [name.name for name in os.scandir(
        xpath) if name.is_file()]

    x_train = []
    y_train = []
    for j, file in enumerate(img_names):
        if file.find(".png") == -1:
            continue
        shape_img = imread('/'.join([xpath, file]), as_gray=True)
        skel_img = imread('/'.join([ypath, file]), as_gray=True)
        x_train.append(shape_img)
        y_train.append(skel_img)

    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=0.33, random_state=37)
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print('x_train = ', x_train.shape)
    print('y_train = ', y_train.shape)
    
    return x_train, x_test, y_train, y_test


