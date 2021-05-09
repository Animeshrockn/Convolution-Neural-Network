# following https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

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

# VGG16
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input
from keras import backend as K
from keras import optimizers

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
        x_train.append(cv2.resize(image, (image_dim, image_dim)))
        # filenames are of the form {class}.{image_num}.jpg
        label = filename.split(os.path.sep)[-1].split(".")[0]
        # record label
        y_train.append(label)

# load validation data
for filename in next(os.walk(validate_path))[2]:
        # full path is path to filename + '/' + filename
        image = cv2.imread(''.join([validate_path, '/', filename]))
        # append resized image
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
datagen = ImageDataGenerator() # VGG16 already rescales input images, no need for further rescaling

datagen.fit(x_train)




# ---- define the model ----
# VGG16
base_model = VGG16(input_shape=(image_dim, image_dim, 3), include_top=False, weights='imagenet')
# base_model = VGG16(input_shape=(image_dim, image_dim, 3), include_top=False)
x = base_model.output
x = Flatten()(x)
d1 = Dense(64, activation='relu')(x)
d1 = Dropout(0.5)(d1)
predictions = Dense(1, activation='sigmoid')(d1)
model = Model(inputs=base_model.input, outputs=predictions) # final model

opt  = optimizers.SGD(lr=0.0001)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


model.summary()


# ---- train the model ----
batch_size = 32
num_epochs = 10

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train) / batch_size, epochs=num_epochs,
                    validation_data=datagen.flow(x_valid, y_valid, batch_size=batch_size),
                    validation_steps = len(x_valid) / batch_size)



# ---- save the model and the weights ----
model.save('saved_model/vgg16_catsdogs.h5')
model.save_weights('saved_weight/vgg16_catsdogs_weights.h5')
print('model saved')



# ---- display history ----
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('graph/train_test_accuracy_vgg16_augmentation.png')
plt.clf() # clear figure

# summarize history for loss (binary cross-entropy)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('binary cross-entropy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('graph/train_test_loss_vgg16_augmentation.png')
plt.clf()