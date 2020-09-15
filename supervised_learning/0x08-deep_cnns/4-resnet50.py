#!/usr/bin/env python3
""" ResNet-50 """

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Function that builds the ResNet architecture"""
    Inputs = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(
                             filters=64,
                             kernel_size=(7, 7),
                             strides=(2, 2),
                             padding='same',
                             kernel_initializer='he_normal')
    conv1 = conv1(Inputs)
    B_norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    act_Relu1 = K.layers.Activation('relu')(B_norm1)
    pool1 = K.layers.MaxPool2D(
                               pool_size=(3, 3),
                               padding='same',
                               strides=(2, 2))
    pool1 = pool1(act_Relu1)
    proje_block1 = projection_block(pool1, filters=(64, 64, 256), s=1)
    id_block1 = identity_block(proje_block1, filters=(64, 64, 256))
    id_block2 = identity_block(id_block1, filters=(64, 64, 256))

    proje_block2 = projection_block(id_block2, filters=(128, 128, 512), s=2)
    id_block3 = identity_block(proje_block2, filters=(128, 128, 512))
    id_block4 = identity_block(id_block3, filters=(128, 128, 512))
    id_block5 = identity_block(id_block4, filters=(128, 128, 512))

    proje_block3 = projection_block(id_block5, filters=(256, 256, 1024), s=2)
    id_block6 = identity_block(proje_block3, filters=(256, 256, 1024))
    id_block7 = identity_block(id_block6, filters=(256, 256, 1024))
    id_block8 = identity_block(id_block7, filters=(256, 256, 1024))
    id_block9 = identity_block(id_block8, filters=(256, 256, 1024))
    id_block10 = identity_block(id_block9, filters=(256, 256, 1024))

    proje_block4 = projection_block(id_block10, filters=(512, 512, 2048))
    id_block11 = identity_block(proje_block4, filters=(512, 512, 2048))
    id_block12 = identity_block(id_block11, filters=(512, 512, 2048))

    pool_avg = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1))
    pool_avg = pool_avg(id_block12)
    dense = K.layers.Dense(1000, activation='softmax')
    dense = dense(pool_avg)
    model = K.Model(inputs=Inputs, outputs=dense)
    return model
