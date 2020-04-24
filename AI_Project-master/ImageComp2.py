import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

# import keras tools
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import optimizers

# read in the csv file (windows)
labels = pd.read_csv(r"C:\Users\daxpa\Downloads\Comp2\train_image_labels.csv")
sample_solution = pd.read_csv(r"C:\Users\daxpa\Downloads\Comp2\sample_submission.csv")

# read in the file that has images (windows)
Xtrain = np.load(r"C:\Users\daxpa\Downloads\Comp2\train_images.npy")
Xtest = np.load(r"C:\Users\daxpa\Downloads\Comp2\test_images.npy")

print (Xtrain.shape)
print (Xtest.shape)
print (np.max(labels['type']))

Y_train = np_utils.to_categorical(labels['type']-1, 102)
print (Y_train[0])

# reshaping the data
Xtrain = Xtrain.reshape(-1, 50, 50, 3)
Xtest = Xtest.reshape(-1, 50, 50, 3)
Xtrain /= 255
Xtest /= 255

#model = Sequential()
#model.add(Dense(50, activation = 'softmax', input_shape = (50*50*3, )))
#model.add(Dense(102, activation = 'softmax', input_shape = (28*28, )))

# adding convulation layers
#The objective of Max Pooling is to down-sample an input representation, reducing its dimensionality and allowing for assumptions to be made about features.
model = Sequential()
model.add(Conv2D(32, kernel_size = (5, 5), activation = 'relu', input_shape = (50, 50, 3)))
model.add(MaxPooling2D(pool_size = (3, 3)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, kernel_size = (2, 2), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# flatten the model
model.add(Flatten())
model.add(Dense(102, activation = 'relu'))
model.add(Dropout(0.5))

model.summary()

# using optimizers (others: sgd, adam, rms_prop())
sgd = optimizers.SGD(lr=.01)

model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
#model.fit(Xtrain, Y_train, batch_size=100, epochs=100, verbose=1)
model.fit(Xtrain, Y_train, batch_size=100, epochs=100,verbose=1)

# predicting the results
Ytest = model.predict(Xtest)
sample_solution['type'] = np.argmax(Ytest, axis = 1) + 1

# writing the results in csv files
sample_solution.to_csv(r"C:\Users\daxpa\Downloads\Comp2\cnn_solution.csv", index = False)