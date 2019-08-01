from __future__ import absolute_import, division, print_function

import imutils
import numpy as np
from PIL import Image
import cv2
from tensorflow import keras
from numpy import genfromtxt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

cap = cv2.VideoCapture(0)
new_model = keras.models.load_model('facial_points.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# y = np.loadtxt('y.txt')
output_pipe = make_pipeline(
    MinMaxScaler(feature_range=(-1, 1))
)

y_arr = genfromtxt('y.txt', delimiter=',')

y = output_pipe.fit(y_arr)



while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        cv2.imwrite('test_gray.jpg', roi_gray)
        basewidth = 96
        img = Image.open('test_gray.jpg')
        test_gray_width, test_gray_height = img.size
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, 96), Image.ANTIALIAS)
        img.save('predict.png')
        pre_img = cv2.imread('predict.png')
        pre_gray_image = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)
        pre_gray_image = pre_gray_image / 255

        pre_gray_image = np.reshape(pre_gray_image, [1, 96, 96,1])
        predictions = new_model.predict(pre_gray_image)
        predictions = output_pipe.inverse_transform(predictions).reshape(15, 2)
        for point in predictions:
            cv2.circle(frame,
                       (int(x + (point[0] * (test_gray_width / 96))), int(y + (point[1] * (test_gray_height / 96)))), 2,
                       (0, 255, 0),thickness=-1)

    cv2.imshow('frame', cv2.flip(frame,flipCode =1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
