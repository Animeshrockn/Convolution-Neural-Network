# following https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# and https://www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras/

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import numpy as np
import argparse
import cv2
import os, os.path

import matplotlib.pyplot as plt

# ---- load data ----

# path to training images
train_path = 'train'


# path to validation images
validate_path = 'validate'

# images to be resized to (image_dim) x (image_dim)
image_dim = 128

x_train = []
y_train = []
x_valid = []
y_valid = []

# load training data
for filename in next(os.walk(train_path))[2]:
        # full path is path to filename + '/' + filename
        image = cv2.imread(''.join([train_path, '/', filename]))
        # append resized image
        print(filename)
        x_train1 = cv2.resize(image, (image_dim, image_dim))
        x_train.append(x_train1)
        # filenames are of the form {class}.{image_num}.jpg
        label = filename.split(os.path.sep)[-1].split(".")[0]
        # record label
        y_train.append(label)

# load validation data
for filename in next(os.walk(validate_path))[2]:
        # full path is path to filename + '/' + filename
        image = cv2.imread(''.join([validate_path, '/', filename]))
        # append resized image
        print(filename)
        x_valid.append(cv2.resize(image, (image_dim, image_dim)))
        # filenames are of the form {class}.{image_num}.jpg
        label = filename.split(os.path.sep)[-1].split(".")[0]
        # record label
        y_valid.append(label)


# change labels from strings to integers, e.g 'cat' -> 0, 'dog' -> 1
le = LabelEncoder()
y_train = le.fit_transform(y_train)  
y_valid = le.fit_transform(y_valid)  

# convert data to NumPy array of floats
x_train = np.array(x_train, np.float32)
x_valid = np.array(x_valid, np.float32)



# ---- define data generator ----
datagen = ImageDataGenerator(rescale=1./255) # rescaling pixel values from [0,255] to [0,1]
datagen.fit(x_train)




# ---- define the model ----
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(image_dim, image_dim, 3),kernel_regularizer=regularizers.l2(0.0001)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3),kernel_regularizer=regularizers.l2(0.0001)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3),kernel_regularizer=regularizers.l2(0.0001)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this produces a 1D feature vector
model.add(Dense(64),kernel_regularizer=regularizers.l2(0.0001))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])


model.summary()


# ---- train the model ----
# batch_size = 128
# num_epochs = 10

# history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
#                     steps_per_epoch=len(x_train) / batch_size, epochs=num_epochs,
#                     validation_data=datagen.flow(x_valid, y_valid, batch_size=batch_size),
#                     validation_steps = len(x_valid) / batch_size)



# # ---- save the model and the weights ----
# model.save('convnet_catsdogs.h5')
# model.save_weights('convnet_catsdogs_weights.h5')
# print('model saved')


# # ---- display history ----
# # list all data in history
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('train_test_accuracy.png')
# plt.clf() # clear figure
# plt.show()
# # summarize history for loss (binary cross-entropy)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.ylabel('binary cross-entropy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('train_test_loss.png')
# plt.clf()
# plt.show()
