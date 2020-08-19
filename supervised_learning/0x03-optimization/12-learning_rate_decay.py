#!/usr/bin/env python3
"""Learning Rate Decay Upgrade"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Function that creates a learning rate decay operation in tensorflow
        inverse time decay"""
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        global_s = sess.run(global_step)
        if global_s % decay_rate == 0:
            alpha = tf.train.inverse_time_decay(alpha, global_step, decay_step, decay_rate, staircase=True)
            return alpha
        else:
            return alpha
