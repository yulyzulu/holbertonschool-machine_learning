#!/usr/bin/env python3
""" DenseNet-121"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Function that builds the DenseNet-121 architecture"""
    Inputs = K.Input(shape=(224, 224, 3))
    B_norm1 = K.layers.BatchNormalization(axis=3)(Inputs)
    act_Relu1 = K.layers.Activation('relu')(B_norm1)
    conv1 = K.layers.Conv2D(
                             filters=64,
                             kernel_size=(7, 7),
                             strides=(2, 2),
                             padding='same',
                             activation='relu',
                             kernel_initializer='he_normal')
    conv1 = conv1(act_Relu1)
    pool1 = K.layers.MaxPool2D(
                               pool_size=(3, 3),
                               padding='same',
                               strides=(2, 2))
    pool1 = pool1(conv1)
    d_block1, filters = dense_block(pool1, 64, growth_rate, 6)
    trans_layer1, filters = transition_layer(d_block1, filters, compression)
    d_block2, filters = dense_block(trans_layer1, filters, growth_rate, 12)
    trans_layer2, filters = transition_layer(d_block2, filters, compression)
    d_block3, filters = dense_block(trans_layer2, filters, growth_rate, 24)
    trans_layer3, filters = transition_layer(d_block3, filters, compression)
    d_block4, filters = dense_block(trans_layer3, filters, growth_rate, 16)

    pool_avg = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1))
    pool_avg = pool_avg(d_block4)
    dense = K.layers.Dense(1000, activation='softmax')
    dense = dense(pool_avg)
    model = K.Model(inputs=Inputs, outputs=dense)
    return model
