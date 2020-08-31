#!/usr/bin/env python3
"""Train"""

def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """Function that trains a model using mini-batch gradient
        descent"""
    history = network.fit(x=data, y=labels, epochs=epochs, shuffle=shuffle,
                        batch_size=batch_size, verbose=verbose)
    return history.history
