#!/usr/bin/env python3
"""Train"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """Function to also train the model using early stopping"""
    if validation_data:
        early = [K.callbacks.EarlyStopping(monitor="val_loss", mode='min',
                                           patience=patience)]
    else:
        early = None

    history = network.fit(x=data, y=labels, epochs=epochs, shuffle=shuffle,
                          batch_size=batch_size, verbose=verbose,
                          validation_data=validation_data, callbacks=early)
    return history.history
