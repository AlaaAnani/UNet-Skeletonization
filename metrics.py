import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from utils import collapse_dim

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(tf.math.multiply(y_true, y_pred), 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(tf.math.multiply(y_true, y_pred), 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    
    mask = tf.math.argmax(y_true, axis=-1)
    y_true = tf.expand_dims(mask, axis=-1)
    y_true = tf.cast(y_true, float)
    
    mask = tf.math.argmax(y_pred, axis=-1)
    y_pred = tf.expand_dims(mask, axis=-1)
    y_pred = tf.cast(y_pred, float)

    

    y_pred = tf.reshape(y_pred, shape=(tf.shape(y_pred)[0], 256,256))
    y_true = tf.reshape(y_true, shape=(tf.shape(y_true)[0], 256,256))

    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def recall_i(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(np.multiply(y_true, y_pred), 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_i(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(np.multiply(y_true, y_pred), 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_i(y_true, y_pred):
    
    y_true = collapse_dim(y_true)
    # y_true = np.cast(y_true, float)
    
    f1_scores = []
    for yt, yp in zip(y_true, y_pred):
        ytf = yt.flatten()
        ypf = yp.flatten()

        precision = precision_i(ytf, ypf)
        recall = recall_i(ytf, ypf)
        f1_scores.append(2*((precision*recall)/(precision+recall+K.epsilon())))

    return np.average(f1_scores), f1_scores