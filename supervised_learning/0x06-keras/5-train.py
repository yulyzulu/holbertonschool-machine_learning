#!/usr/bin/env python3
"""Train"""


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """Function to also analyze validaiton data"""
    history = network.fit(x=data, y=labels, epochs=epochs, shuffle=shuffle,
                          batch_size=batch_size, verbose=verbose,
                          validation_data=validation_data)
    return history.history
