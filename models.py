from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.layers import *


def lenet5(input_shape=(32, 32, 3), nb_classes=43, drop=True):

    model = Sequential()
    # Layer 1: Conv Layer 1
    model.add(Conv2D(filters=6,
                     kernel_size=5,
                     strides=1,
                     activation='relu',
                     input_shape=input_shape))
    # Pooling layer 1
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # Layer 2: Conv Layer 2
    model.add(Conv2D(filters=16,
                     kernel_size=5,
                     strides=1,
                     activation='relu',
                     input_shape=(14, 14, 6)))
    # Pooling Layer 2
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # Flatten
    model.add(Flatten())

    if drop:
        model.add(Dropout(0.5))

    # Layer 3: fully connected layer 1
    model.add(Dense(120, activation='relu'))

    # Layer 4: fully connected layer 2
    model.add(Dense(84, activation='relu'))

    # Layer 5: output Layer
    model.add(Dense(nb_classes, activation='softmax'))

    return model


def lenet5_advanced(input_shape=(32, 32, 3), nb_classes=43, drop=True):

    model = Sequential()
    # Layer 1: Conv Layer 1
    model.add(Conv2D(filters=108,
                     kernel_size=5,
                     strides=1,
                     activation='relu',
                     input_shape=input_shape))
    # Pooling layer 1
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # Layer 2: Conv Layer 2
    model.add(Conv2D(filters=200,
                     kernel_size=5,
                     strides=1,
                     activation='relu',
                     input_shape=(14, 14, 6)))
    # Pooling Layer 2
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # Flatten
    model.add(Flatten())

    if drop:
        model.add(Dropout(0.5))

    # Layer 3: fully connected layer 1
    model.add(Dense(1000, activation='relu'))

    # Layer 4: fully connected layer 2
    model.add(Dense(200, activation='relu'))

    # Layer 5: output Layer
    model.add(Dense(nb_classes, activation='softmax'))

    return model