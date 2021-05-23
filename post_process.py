
import cv2
from skimage.io import imread, imsave
import os
import numpy as np


def erode_imgs():
    Y_pred = 'Y_pred'
    # Y_target = 'Y_target'

    img_names = [name.name for name in os.scandir(
        Y_pred) if name.is_file()]

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    # Create an empty output image to hold values
    for j, file in enumerate(img_names):
        if file.find(".png") == -1:
            continue
        pred_img = imread('/'.join([Y_pred, file]), as_gray=True)
        thin = np.zeros(pred_img.shape,dtype='uint8')
        # Erosion
        erode = cv2.erode(pred_img,kernel)
        # Opening on eroded image
        opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
        # Subtract these two
        subset = erode - opening
        # Union of all previous sets
        thin = cv2.bitwise_or(subset,thin)

        imsave(f'Y_eroded/{file}', thin)


def overlap_skeleton(shape_path='test_shapes', skel_path='Y_pred_new'):
    out_dir = 'overlapped'

    img_names = [name.name for name in os.scandir(
        shape_path) if name.is_file()]
    
    for j, file in enumerate(img_names):
        if file.find(".png") == -1:
            continue
        shape_img = imread('/'.join([shape_path, file]), as_gray=True)
        skel_img = imread('/'.join([skel_path, file]), as_gray=True)

        overlp = cv2.bitwise_xor(shape_img,skel_img)
        imsave(f'overlapped/{file}', overlp)


def dist_transform(X):
    dist_imgs = []
    for img in X:
        dist_img = cv2.distanceTransform(img, cv2.DIST_C, 3)
        dist_imgs.append(dist_img)
    
    return np.array(dist_imgs)

def distance_transorm(path):
    out_dir = 'distanced'

    img_names = [name.name for name in os.scandir(
        path) if name.is_file()]
    

    
    for j, file in enumerate(img_names):
        if file.find(".png") == -1:
            continue
        img = imread('/'.join([path, file]), as_gray=True)
        dist_img = cv2.distanceTransform(img, cv2.DIST_C, 5)

        imsave(f'{out_dir}/{file}', dist_img)

# overlap_skeleton()
distance_transorm('test_shapes')