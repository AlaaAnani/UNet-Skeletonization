# %%
# Python program to illustrate
# template matching
import cv2
import numpy as np
 
# Read the main image
img_gray = cv2.imread('ouroutput.png', 0)
img_gray = cv2.copyMakeBorder(img_gray, 50, 50, 50, 50, cv2.BORDER_CONSTANT)
# Convert it to grayscale
 
# Read the template
template = cv2.imread('target.png',0)

# Store width and height of template in w and h
w, h = template.shape[::-1]
# %%
print(img_gray.shape, template.shape)
# Perform match operations.
res = cv2.matchTemplate(img_gray, template,cv2.TM_CCORR_NORMED)
 
# Specify a threshold
threshold = 0.5
print(np.max(res))
# Store the coordinates of matched area in a numpy array
loc = np.where( res >= threshold)
 
# Draw a rectangle around the matched region.
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_gray, pt, (pt[0] + w, pt[1] + h), (255,255,255), 2)
    break
# Show the final image with the matched area.
cv2.imwrite('Detected.png', img_gray)

print(len(loc[0]))
# %%
import tensorflow as tf
a_tensor = tf.constant([[1, 2, 3], [4, 5, 6], [4, 5, 6]])
an_array = a_tensor.numpy()

print(tf.shape(a_tensor))
# %%
y_pred = tf.constant([[1, 2]])
proto_tensor = tf.make_tensor_proto(y_pred)
img_gray = tf.make_ndarray(proto_tensor)
# %%
print(img_gray)
# %%
import tensorflow as tf
from tensorflow.python.keras import backend as K
sess = K.get_session()
array = sess.run(a_tensor)
# %%
