#!/usr/bin/env python3
"""Train"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """Function to also train the model using learning decay"""
    if validation_data and early_stopping:
        early = K.callbacks.EarlyStopping(monitor="val_loss", mode='min',
                                          patience=patience)
        callback_list = [early]

        if learning_rate_decay is True:
            def scheduler(epoch):
                return alpha / (1 + decay_rate * epoch)
            lr = K.callbacks.LearningRateScheduler(schedule=scheduler,
                                                   verbose=1)
            callback_list.append(lr)
    else:
        early = None

    history = network.fit(x=data, y=labels, epochs=epochs, shuffle=shuffle,
                          batch_size=batch_size, verbose=verbose,
                          validation_data=validation_data,
                          callbacks=callback_list)
    return history
