
import os
import skimage
from skimage.io import imread, imsave
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_dataset(path='dataset'):
    xpath = f'{path}/img_train_shape'
    ypath = f'{path}/img_train_skeletons'

    img_names = [name.name for name in os.scandir(
        xpath) if name.is_file() and name.name.find(".png") != -1]

    x_train = []
    y_train = []

    for j, file in enumerate(img_names):
        if file.find(".png") == -1:
            img_names.pop(j)
            continue
        shape_img = imread('/'.join([xpath, file]), as_gray=True)
        skel_img = imread('/'.join([ypath, file]), as_gray=True)
        x_train.append(shape_img)
        y_train.append(skel_img)

    x_train, x_test, y_train, y_test, names_train, names_test = train_test_split(
        x_train, y_train, img_names, test_size=0.33, random_state=37)
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print('x_train = ', x_train.shape)
    print('y_train = ', y_train.shape)
    
    return x_train, x_test, y_train, y_test, names_train, names_test


def read_tests(path='dataset'):
    xpath = f'{path}/img_test_shape'

    img_names = [name.name for name in os.scandir(
        xpath) if name.is_file()]

    x_test = []
    for j, file in enumerate(img_names):
        if file.find(".png") == -1:
            continue
        shape_img = imread('/'.join([xpath, file]), as_gray=True)
        x_test.append(shape_img)
    
    x_test = np.array(x_test)

    return x_test, img_names

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


def collapse_dim(y):
    mask = np.argmax(y, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    mask = mask.reshape(*mask.shape[:-1])
    return mask

def write_imgs(imgs, img_names, path, collapse=False):
    for i, y in enumerate(imgs):
        if collapse:
            y = collapse_dim(y)
        timg = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        imsave(f'{path}/{img_names[i]}', timg)

def dist_transform(X):
    dist_imgs = []
    for img in X:
        timg = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        dist_img = cv2.distanceTransform(timg, cv2.DIST_C, 3)
        dist_imgs.append(dist_img)
    
    return np.array(dist_imgs)

def distance_transform(path):
    out_dir = 'distanced'

    img_names = [name.name for name in os.scandir(
        path) if name.is_file()]
    

    
    for j, file in enumerate(img_names):
        if file.find(".png") == -1:
            continue
        img = imread('/'.join([path, file]), as_gray=True)
        dist_img = cv2.distanceTransform(img, cv2.DIST_C, 5)

        imsave(f'{out_dir}/{file}', dist_img)


def load_data(DIST=False, NO_TEST=True):
    x_train, x_val, y_train, y_val, names_train, names_val = read_dataset()
    y_train, y_val = reshape_target(y_train), reshape_target(y_val)

    if NO_TEST:
        x_test, names_test = None, None
    else:
        x_test, names_test = read_tests()

    if DIST:
        x_train = dist_transform(x_train)
        x_val = dist_transform(x_val)
        x_test = dist_transform(x_test)
    


    return x_train, x_val, y_train, y_val, names_train, names_val, x_test, names_test