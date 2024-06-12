# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AhH3OjsZGppdnROcQxFUZkZSH19Q8KO9
"""

import numpy as np
import matplotlib.pyplot as plt

import keras

from keras.datasets import mnist

mnist.load_data

(X_train,y_train),(X_test,y_test)=mnist.load_data()

X_train.shape,y_train.shape,X_test.shape,y_test.shape

plt.imshow(X_train[0])

plt.imshow(X_train[0],cmap='binary')

def plot_input_img(i):
    plt.imshow(X_train[i],cmap='binary')
    plt.title(y_train[i])
    plt.show()

for i in range(10):
    plot_input_img(i)

X_train=X_train.astype(np.float32)/255
X_test=X_test.astype(np.float32)/255

X_train.shape

# Pre-process the image

# Normalising the image 
X_train=X_train.astype(np.float32)/255
X_test=X_test.astype(np.float32)/255

#Reshape or Expand the size of image to(28,28)
X_train=np.expand_dims(X_train,-1)
X_test=np.expand_dims(X_test,-1)



X_train.shape

y_train = keras.utils.to_categorical(y_train)

y_train

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

#covert classes to one hot vectors
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dropout(0.25))

model.add(Dense(10, activation='softmax'))

import tensorflow as tf
from tensorflow import keras

# Define a Keras model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

#Callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint

#Earlystopping

es = EarlyStopping(monitor='val_acc', min_delta= 0.01 , patience= 4, verbose= 1)

#Model Check point

mc = ModelCheckpoint("./bestmodel.h5",monitor="val_acc",verbose= 1, save_best_only= True)

cb = [es,mc]

"""Model training"""

his = model.fit(X_train, y_train, epochs= 5, validation_split= 0.3, callbacks = cb)

model_S = keras.models.load_mode("E://python_project//Python Projects//bestmodel.h5")