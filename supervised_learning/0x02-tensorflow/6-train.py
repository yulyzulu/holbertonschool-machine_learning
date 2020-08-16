#!/usr/bin/env python3
""" Train """

import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """ Function that builds, trains, and saves a neural network classifier"""
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_pred", y_pred)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("train_op", train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
#        step = 0
        for i in range(iterations + 1):
            cost, acc = sess.run([loss, accuracy],
                                 feed_dict={x: X_train, y: Y_train})
            cost2, acc2 = sess.run([loss, accuracy],
                                   feed_dict={x: X_valid, y: Y_valid})

            if i == 0 or i % 100 == 0 or i == iterations:
                print('After', i, 'iterations:')
                print('\tTraining Cost:', cost)
                print('\tTraining Accuracy:', acc)
                print('\tValidation Cost:', cost2)
                print('\tValidation Accuracy:', acc2)
#                step = step + 100

            if i != iterations:
                sess.run([train_op], feed_dict={x: X_train, y: Y_train})

#        path = saver.save(sess, save_path)
    return saver.save(sess, save_path)
