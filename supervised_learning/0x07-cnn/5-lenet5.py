#!/usr/bin/env python3
""" LeNet-5- Keras"""

import tensorflow.keras as K


def lenet5(X):
    """Function that builds a modified version of the LeNet-5
        architecture using Keras"""
    conv1 = K.layers.Conv2D(
                            filters=6,
                            kernel_size=(5, 5),
                            padding='same',
                            activation='relu',
                            kernel_initializer='he_normal')
    conv1 = conv1(X)
    pool1 = K.layers.MaxPool2D(
                               pool_size=(2, 2),
                               strides=(2, 2))
    pool1 = pool1(conv1)
    conv2 = K.layers.Conv2D(
                            filters=16,
                            kernel_size=(5, 5),
                            padding='valid',
                            activation='relu',
                            kernel_initializer='he_normal')
    conv2 = conv2(pool1)
    pool2 = K.layers.MaxPool2D(
                               pool_size=(2, 2),
                               strides=(2, 2))
    pool2 = pool2(conv2)
    flatten = K.layers.Flatten()(pool2)
    layer1 = K.layers.Dense(
                            units=120,
                            activation='relu',
                            kernel_initializer='he_normal')
    layer1 = layer1(flatten)
    layer2 = K.layers.Dense(
                            units=84,
                            activation='relu',
                            kernel_initializer='he_normal')
    layer2 = layer2(layer1)
    layer3 = K.layers.Dense(
                            units=10,
                            activation='softmax',
                            kernel_initializer='he_normal')
    layer3 = layer3(layer2)
    optimizer = K.optimizers.Adam()
    model = K.Model(inputs=X, outputs=layer3)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    return model
