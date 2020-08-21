#!/usr/bin/env python3
"""Model"""

import tensorflow as tf
import numpy as np


def create_batch_norm_layer(prev, n, activation):
    """Function that creates a batch normalization layer for a
         neural network in tensorflow"""
    W = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            kernel_initializer=W,
                            name="layer")
    X = layer(prev)
    mean, variance = tf.nn.moments(X, [0])
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    X_norm = tf.nn.batch_normalization(X, mean, variance, offset=beta,
                                       scale=gamma, variance_epsilon=1e-8)
#    if not activation:
#        return X_norm
    return activation(X_norm)


def create_layer(prev, n, activation):
    """Function that create layers"""
    W = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=W,
                            name="layer")
    return layer(prev)


def forward_prop(x, layer_sizes=[], activations=[]):
    """Function that creates the forward propagation
        graph for the Neural Network"""
    layers = len(layer_sizes)
    prediction = x
    for i in range(layers):
        if i != layers - 1:
            prediction = create_batch_norm_layer(prediction, layer_sizes[i],
                                                 activations[i])
        else:
            prediction = create_layer(prediction, layer_sizes[i],
                                      activations[i])
    return prediction


def calculate_accuracy(y, y_pred):
    """Function that calculates the accuracy of a preddiction"""
    prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    return accuracy


def calculate_loss(y, y_pred):
    """ Function that calculates the softmax cross-entropy
        loss of a predicction """
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    return loss


def shuffle_data(X, Y):
    """Function that suffles the data poins in two matrices
       the same way"""
    m = X.shape[0]
    shuff = np.random.permutation(m)
    return X[shuff], Y[shuff]


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Function that creates the training operation for a neural network
       in tensorflow using the Adam optimization algorithm"""
    optimizer = tf.train.AdamOptimizer(alpha, beta1=beta1, beta2=beta2,
                                       epsilon=epsilon)
    return optimizer.minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Function that creates a learning rate decay operation in tensorflow
        inverse time decay"""
    alpha = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                        decay_rate, staircase=True)
    return alpha


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """Function that builds, trains, and saves a neural network model,
        in tensorflow using Adam optimization, mini-batch gradient
        descent, learning rate decay, and batch normalization"""
    x = tf.placeholder(tf.float32, shape=(None, Data_train[0].shape[1]))
    y = tf.placeholder(tf.float32, shape=(None, Data_train[1].shape[1]))
    y_pred = forward_prop(x, layers, activations)
    accuracy = calculate_accuracy(y, y_pred)
#    loss = calculate_loss(y, y_pred)
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    global_step = tf.Variable(0, trainable=False)
    global_step_inc = tf.assign(global_step, global_step + 1)
    alpha_n = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha_n, beta1, beta2, epsilon)

    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_pred", y_pred)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("train_op", train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

#    batches = Data_train[0].shape[1] // batch_size
#    if batches % batch_size != 0:
#        batches = batches + 1
#        flag = True
#    else:
#        flag = False

    with tf.Session() as sess:
        sess.run(init)
        batches = Data_train[0].shape[0] // batch_size
        if batches % batch_size != 0:
            batches = batches + 1
            flag = True
        else:
            flag = False

        for i in range(epochs + 1):
            cost, acc = sess.run([loss, accuracy],
                                 feed_dict={x: Data_train[0],
                                            y: Data_train[1]})
            cost2, acc2 = sess.run([loss, accuracy],
                                   feed_dict={x: Data_valid[0],
                                              y: Data_valid[1]})
            print('After {} epochs:'.format(i))
            print('\tTraining Cost: {}'.format(cost))
            print('\tTraining Accuracy: {}'.format(acc))
            print('\tValidation Cost: {}'.format(cost2))
            print('\tValidation Accuracy: {}'.format(acc2))

            if i < epochs:
                X_train_S, Y_train_S = shuffle_data(Data_train[0],
                                                    Data_train[1])

                for j in range(batches):
                    start = j * batch_size
                    if j == batches - 1 and flag:
                        final = Data_train[0].shape[0]
                    else:
                        final = j * batch_size + batch_size

                    X_train_mini = X_train_S[start:final]
                    Y_train_mini = Y_train_S[start:final]
                    sess.run(train_op,
                             feed_dict={x: X_train_mini, y: Y_train_mini})

                    if j != 0 and (j + 1) % 100 == 0:
                        cost_B, acc_B = sess.run([loss, accuracy],
                                                 feed_dict={x: X_train_mini,
                                                            y: Y_train_mini})
                        print('\tStep {}:'.format(j + 1))
                        print('\t\tCost: {}'.format(cost_B))
                        print('\t\tAccuracy: {}'.format(acc_B))
#            sess.run(tf.assign(global_step, global_step + 1))
            sess.run(global_step_inc)
        return saver.save(sess, save_path)
