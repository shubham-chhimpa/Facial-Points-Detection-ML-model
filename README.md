# Facial-Points-Detection-ML-model


## Demo


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/j40Ubx7uUEs/0.jpg)]

(https://www.youtube.com/watch?v=j40Ubx7uUEs)

## Description

It is a neural network model that detects 15 facial points.

The Dataset :

 Kaggle Dataset : 
https://lnkd.in/fp9nhXg

The Model :

created a basic NN using Keras sequential containing 1 cov2d and 1 max pool and 2  hidden layer (128 and 64) and output layer (30)

Training accuracy : 74-75 percent 

Then saved the model into .h5 file

The Detector and Predictor :

 I used opencv Haar cascade 
To detect faces

loaded the above created model(h5 file) to create a instance of model 

 Faces based on the detector are feeded to the model.predict() function to get the prediction and (15)key points.

The model is ready to be used by anybody through the (h5 file) with 5-6 lines of code 
