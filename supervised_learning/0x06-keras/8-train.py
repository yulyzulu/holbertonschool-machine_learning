#!/usr/bin/env python3
"""Train"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """Function to also save the best iteration of the model"""
    callback_list = []
    if validation_data:
        early = K.callbacks.EarlyStopping(monitor="val_loss",
                                          mode='min', patience=patience)
        callback_list.append(early)

        if learning_rate_decay is True:
            def scheduler(epoch):
                return alpha / (1 + decay_rate * epoch)
            lr = K.callbacks.LearningRateScheduler(schedule=scheduler,
                                                   verbose=1)
            callback_list.append(lr)
#    else:
#        early = None
    if filepath:
        check = K.callbacks.ModelCheckpoint(filepath=filepath,
                                            monitor='val_loss', verbose=0,
                                            save_best_only=save_best)
        callback_list.append(check)
    if len(callback_list) == 0:
        callback_list = None

    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callback_list)
    return history
