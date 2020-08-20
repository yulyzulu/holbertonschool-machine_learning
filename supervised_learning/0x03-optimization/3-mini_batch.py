#!/usr/bin/env python3
"""Mini-batch """

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ Function that trains a loaded NN model using mini-batch
       gradient descend"""
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(load_path+'.meta')
        new_saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
#        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        batches = X_train.shape[0] // batch_size
        if batches % batch_size != 0:
            batches = batches + 1
            flag = True
        else:
            flag = False

        for i in range(epochs + 1):
            cost, acc = sess.run([loss, accuracy],
                                 feed_dict={x: X_train, y: Y_train})
            cost2, acc2 = sess.run([loss, accuracy],
                                   feed_dict={x: X_valid, y: Y_valid})
            print('After {} epochs:'.format(i))
            print('\tTraining Cost: {}'.format(cost))
            print('\tTraining Accuracy: {}'.format(acc))
            print('\tValidation Cost: {}'.format(cost2))
            print('\tValidation Accuracy: {}'.format(acc2))

            if i < epochs:
                X_train, Y_train = shuffle_data(X_train, Y_train)

                for j in range(batches):
                    start = j * batch_size
                    if j == batches - 1 and flag:
                        end = X_train.shape[0]
                    else:
                        end = j * batch_size + batch_size

                    X_train_mini = X_train[start:end]
                    Y_train_mini = Y_train[start:end]
                    sess.run([train_op],
                             feed_dict={x: X_train_mini, y: Y_train_mini})
#                cost_B, acc_B = sess.run([loss, accuracy],
#                             feed_dict={x: X_train_mini, y: Y_train_mini})
#                sess.run([train_op],
# feed_dict={x: X_train_mini, y: Y_train_mini})
                    if j != 0 and j % 100 == 0:
                        cost_B, acc_B = sess.run([loss, accuracy],
                                                 feed_dict={x: X_train_mini,
                                                            y: Y_train_mini})
                        print('\tStep {}:'.format(j))
                        print('\t\tCost: {}'.format(cost_B))
                        print('\t\tAccuracy: {}'.format(acc_B))
        saver = tf.train.Saver()
        return saver.save(sess, save_path)
