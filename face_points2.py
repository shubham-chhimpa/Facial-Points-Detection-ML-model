#####  TEST YOUR IMAGE FILE WITH THE MODEL  #####


from __future__ import absolute_import, division, print_function

import imutils
from tensorflow import keras

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import time

# Load the saved model
model = keras.models.load_model('facial_points.h5')  # <-- Saved model path

output_pipe = make_pipeline(
    MinMaxScaler(feature_range=(-1, 1))
)

y = output_pipe.fit([[66.03356391, 39.00227368, 30.22700752, 36.4216782, 59.58207519, 39.64742256,
                      73.13034586, 39.96999699, 36.35657143, 37.3894015, 23.45287218, 37.3894015,
                      56.95326316, 29.03364812, 80.22712782, 32.22813835, 40.22760902, 29.0023218,
                      16.35637895, 29.64747068, 44.42057143, 57.06680301, 61.19530827, 79.97016541,
                      28.61449624, 77.38899248, 43.3126015, 72.93545865, 43.13070677, 84.48577444]
                     ])


def detect_points(face_img):
    me = np.array(face_img) / 255
    # x_test = np.expand_dims(me, axis=0)
    # x_test = np.expand_dims(x_test, axis=3)
    x_test = np.reshape(me, [1, 96, 96])
    y_test = model.predict(x_test)
    label_points = output_pipe.inverse_transform(y_test).reshape(15, 2)
    # label_points = (np.squeeze(label_points) * 48) + 48

    return label_points


# Load haarcascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
dimensions = (96, 96)

# Enter the path to your test image
img = cv2.imread('test_face.jpg')

default_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
# faces = face_cascade.detectMultiScale(gray_img, 4, 6)

faces_img = np.copy(gray_img)

plt.rcParams["axes.grid"] = False

all_x_cords = []
all_y_cords = []

for i, (x, y, w, h) in enumerate(faces):

    just_face = cv2.resize(gray_img[y:y + h, x:x + w], dimensions)
    cv2.rectangle(faces_img, (x, y), (x + w, y + h), (255, 0, 0), 1)

    scale_val_x = w / 96
    scale_val_y = h / 96
    label_point = detect_points(just_face)
    print('-------------------------------')
    print(label_point[::2])
    print('-------------------------------')

    print(label_point[1::2])
    print('-------------------------------')

    for points in label_point:
        all_x_cords.append((label_point[0] * scale_val_x) + x)
        all_y_cords.append((label_point[1] * scale_val_y) + y)
        print(label_point)
    plt.imshow(just_face, cmap='gray')

    plt.plot(all_x_cords, all_y_cords, 'ro', markersize=5)
    plt.show()

plt.imshow(default_img)
plt.plot(all_x_cords, all_y_cords, 'wo', markersize=3)
plt.show()
