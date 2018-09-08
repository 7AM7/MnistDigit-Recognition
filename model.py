import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np
from keras.models import model_from_json
import cv2

#load json and create model
def load_model():
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model/model.h5")
    print("Loaded model from disk")
    return loaded_model

def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model/model.h5")
    print("Saved model to disk")
    
def build_model():
    model = Sequential()

    model.add(Conv2D(
        filters = 64,
        kernel_size = (3, 3),
        padding = 'same',
        activation = 'relu',
        input_shape = (28, 28, 1)))

    model.add(Conv2D(
        filters = 128,
        kernel_size = (3,3),
        padding = 'same',
        activation = 'relu'))

    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(
        filters = 128,
        kernel_size = (3,3),
        padding = 'same',
        activation = 'relu'))

    model.add(Conv2D(
        filters = 128,
        kernel_size = (3,3),
        padding = 'same',
        activation = 'relu'))

    model.add(Conv2D(
        filters = 128,
        kernel_size = (3,3),
        padding = 'same',
        activation = 'relu'))

    model.add(Conv2D(
        filters = 256,
        kernel_size = (3,3),
        padding = 'same',
        activation = 'relu'))

    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(
        filters = 256,
        kernel_size = (3,3),
        padding = 'same',
        activation = 'relu'))

    model.add(Conv2D(
        filters = 256,
        kernel_size = (3,3),
        padding = 'same',
        activation = 'relu'))

    model.add(Conv2D(
        filters = 256,
        kernel_size = (3,3),
        padding = 'same',
        activation = 'relu'))

    model.add(Conv2D(
        filters = 512,
        kernel_size = (3, 3),
        padding = 'same',
        activation = 'relu'))

    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(
        filters = 512,
        kernel_size = (3, 3),
        padding = 'same',
        activation = 'relu'))

    model.add(Conv2D(
        filters = 512,
        kernel_size = (3, 3),
        padding = 'same',
        activation = 'relu'))

    model.add(Conv2D(
        filters = 512,
        kernel_size = (3, 3),
        padding = 'same',
        activation = 'relu'))

    model.add(Conv2D(
        filters = 512,
        kernel_size = (3, 3),
        padding = 'same',
        activation = 'relu'))

    model.add(Conv2D(
        filters = 1024,
        kernel_size = (1, 1),
        padding = 'same',
        activation = 'relu'))

    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation = 'softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    return model
