import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(X_train.shape[0],-1)/255 #Normalizing image data
X_test = X_test.reshape(X_test.shape[0],-1)/255 #Normalizing image data
y_train = pd.get_dummies(y_train).values #Encode targets as one-hot encodings to implement softmax
y_test = pd.get_dummies(y_test).values #Encode targets as one-hot encodings to implement softmax

