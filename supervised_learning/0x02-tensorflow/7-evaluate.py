#!/usr/bin/env python3
"""Evaluate"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """Function that evaluates the output of a neural network"""
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(save_path+'.meta')
        new_saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')
        train_op = tf.get_collection('train_op')[0]

        y_pred2, accuracy2, loss2 = sess.run([y_pred, accuracy, loss],
                                             feed_dict={x: X, y: Y})
    return y_pred2, accuracy2, loss2
