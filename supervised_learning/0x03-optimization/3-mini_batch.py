#!/usr/bin/env python3
"""Optimization module"""
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        batch_size=32,
        epochs=5,
        load_path="/tmp/model.ckpt",
        save_path="/tmp/model.ckpt"
):
    """Trains a loaded neural network model using mini-batch gradient descent

    Args:
        X_train (np.ndarray): Is the training data of shape (m, 784), where
        m is the number of data points and 784 is the number of features.
        Y_train (np.ndarray): Is the one-hot training labels with shape (m, 10)
        where 10 is the number of classes.
        X_valid (np.ndarray): Is the validation data of shape (m, 784)
        Y_valid (np.ndarray): Is the one-hot validation labels with shape
        (m, 10)
        batch_size (int): Is the number of data point in batch.
        epochs (int): Is the number of times the training should pass through
        the whole dataset.
        load_path (str): Is the path from which to load the model.
        save_path (str): Is the path where the model should saved after
        training

    Returns:
        str: The path where the model was saved

    """
    m = X_train.shape[0]
    batches = m / batch_size
    if batches % 1 != 0:
        batches += 1
    with tf.Session() as session:
        saver = tf.train.import_meta_graph("{}.meta".format(load_path))
        saver.restore(session, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        for epoch in range(epochs + 1):
            train_cost, train_accuracy = session.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train}
            )
            valid_cost, valid_accuracy = session.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid}
            )
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))
            if epoch < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
                for batch in range(batches):
                    start = batch * batch_size
                    if start + batch_size < m:
                        end = start + batch_size
                    else:
                        end = m - 1
                    indices = np.arange(start, end)
                    X_batch = X_shuffled[indices]
                    Y_batch = Y_shuffled[indices]
                    session.run(
                        train_op,
                        feed_dict={
                            x: X_batch,
                            y: Y_batch
                        }
                    )
                    if batch % 100 == 0 or batch == batches:
                        batch_cost, batch_accuracy = session.run(
                            [loss, accuracy],
                            feed_dict={x: X_batch, y: Y_batch}
                        )
                        print("\tStep {}:".format(batch))
                        print("\t\tCost: {}".format(batch_cost))
                        print("\t\tAccuracy: {}".format(batch_accuracy))
        return saver.save(session, save_path)
