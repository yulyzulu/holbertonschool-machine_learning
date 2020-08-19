#!/usr/bin/env python3
"""Learning Rate Decay"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ """
    if global_step % decay_rate == 0:
        alpha = alpha / (1 + decay_rate * np.floor(global_step / decay_step))
        return alpha
    else:
        return alpha
