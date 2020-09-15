#!/usr/bin/env python3
""" Inception Network"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Function that builds the inception network"""
    inputs = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(
                             filters=64,
                             kernel_size=(7, 7),
                             strides=(2, 2),
                             padding='same',
                             activation='relu',
                             kernel_initializer='he_normal')
    conv1 = conv1(inputs)
    pool1 = K.layers.MaxPool2D(
                               pool_size=(3, 3),
                               padding='same',
                               strides=(2, 2))
    pool1 = pool1(conv1)
    conv2_1 = K.layers.Conv2D(
                             filters=64,
                             kernel_size=(1, 1),
                             strides=(1, 1),
                             padding='same',
                             activation='relu',
                             kernel_initializer='he_normal')
    conv2_1 = conv2_1(pool1)
    conv2_2 = K.layers.Conv2D(
                             filters=192,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding='same',
                             activation='relu',
                             kernel_initializer='he_normal')
    conv2_2 = conv2_2(conv2_1)
    pool2 = K.layers.MaxPool2D(
                               pool_size=(3, 3),
                               padding='same',
                               strides=(2, 2))
    pool2 = pool2(conv2_2)
    inc_3a = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    inc_3b = inception_block(inc_3a, [128, 128, 192, 32, 96, 64])
    pool3 = K.layers.MaxPool2D(
                               pool_size=(3, 3),
                               padding='same',
                               strides=(2, 2))
    pool3 = pool3(inc_3b)
    inc_4a = inception_block(pool3, [192, 96, 208, 16, 48, 64])
    inc_4b = inception_block(inc_4a, [160, 112, 224, 24, 64, 64])
    inc_4c = inception_block(inc_4b, [128, 128, 256, 24, 64, 64])
    inc_4d = inception_block(inc_4c, [112, 144, 288, 32, 64, 64])
    inc_4e = inception_block(inc_4d, [256, 160, 320, 32, 128, 128])

    pool4 = K.layers.MaxPool2D(
                               pool_size=(3, 3),
                               padding='same',
                               strides=(2, 2))
    pool4 = pool4(inc_4e)
    inc_5a = inception_block(pool4, [256, 160, 320, 32, 128, 128])
    inc_5b = inception_block(inc_5a, [384, 192, 384, 48, 128, 128])
    pool5 = K.layers.AveragePooling2D(
                                      pool_size=(7, 7),
                                      strides=(1, 1))
    pool5 = pool5(inc_5b)
    dropout = K.layers.Dropout(0.4)(pool5)
    linear = K.layers.Dense(1000, activation='relu')(dropout)
    model = K.Model(inputs=inputs, outputs=linear)
    return model
