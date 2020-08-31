#!/usr/bin/env python3
"""Input"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that builds a neural network with the Keras library"""
    inputs = K.Input(shape=(nx,))
    regula = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            x = K.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=regula)(inputs)
            x = K.layers.Dropout(1 - keep_prob)(x)
        elif i < (len(layers) - 1):
            x = K.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=regula)(x)
            x = K.layers.Dropout(1 - keep_prob)(x)
        else:
            outputs = K.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=regula)(x)
    model = K.Model(inputs=inputs, outputs=outputs)

    return model
