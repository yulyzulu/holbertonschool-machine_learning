#!/usr/bin/env python3
""" LeNet-5- Tensorflow """

import tensorflow as tf
import numpy as np


def lenet5(x, y):
    """Function thatthat builds a modified version of the
         LeNet-5 architecture using tensorflow"""
    W = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.Conv2D(
                             filters=6,
                             kernel_size=(5, 5),
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=W)
    conv1 = conv1(x)
    pool1 = tf.layers.MaxPooling2D(
                                   pool_size=(2, 2),
                                   strides=(2, 2))
    pool1 = pool1(conv1)
    conv2 = tf.layers.Conv2D(
                             filters=16,
                             kernel_size=(5, 5),
                             padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=W)
    conv2 = conv2(pool1)
    pool2 = tf.layers.MaxPooling2D(
                                   pool_size=(2, 2),
                                   strides=(2, 2))
    pool2 = pool2(conv2)
    flatten = tf.layers.Flatten()(pool2)
    layer1 = tf.layers.Dense(
                             units=120,
                             activation=tf.nn.relu,
                             kernel_initializer=W)
    layer1 = layer1(flatten)
    layer2 = tf.layers.Dense(
                             units=84,
                             activation=tf.nn.relu,
                             kernel_initializer=W)
    layer2 = layer2(layer1)
    layer3 = tf.layers.Dense(
                             units=10,
                             kernel_initializer=W)
    layer3 = layer3(layer2)
    y_pred = tf.nn.softmax(layer3)
    prediction = tf.equal(tf.argmax(layer3, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=layer3)
    Optimizer = tf.train.AdamOptimizer()
    training = Optimizer.minimize(loss)
    return y_pred, training, loss, accuracy
