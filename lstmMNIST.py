###################################################
#
# Build a feed forward RNN (lstm) in tensorflow ANN to predict digits from MNIST data
# Jason Dean
# 09/16/17
#
###################################################

import tensorflow as tf
print(tf.__version__)
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn

def lstm(mnist):
    # each image is 28X28, or 784 pixels
    # we will input a 56 sequences 14 times into the RNN
    # define placeholders
    x = tf.placeholder(tf.float32, [None, 14, 56])
    y = tf.placeholder(tf.float32, [None, 10])

    # we will make a RNN with 25 nodes
    weights = tf.Variable(tf.random_normal([25, 10], stddev=0.01))
    biases = tf.Variable(tf.random_normal([10], stddev=0.01))

    # make a LSTM
    # this function will be receiving a set of 14x56 images.
    # we want to re-group these by layers of 56 pixels so that the first layer for
    # all of the images is passed to the first cell, and so forth down the line
    x_ = tf.unstack(x, 14, 1)

    # define a lstm_cell with 25 hidden nodes
    lstm_cell = rnn.BasicLSTMCell(25, forget_bias=1.0)

    # Get lstm cell outputs
    outputs, states = rnn.static_rnn(lstm_cell, x_, dtype=tf.float32)

    # get last output for prediction
    logits = tf.add(tf.matmul(outputs[-1], weights), biases)

    # loss functions
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    prediction = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    epochs = 100
    losses = []

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        init = tf.global_variables_initializer()
        sess.run(init)

        # load the image data
        images = mnist.train.images
        images = images.reshape(images.shape[0], 14, 56)

        # load the labels
        labels = mnist.train.labels

        for i in range(epochs):
            # train!
            sess.run(train_op, feed_dict={x: images, y: labels})
            loss = sess.run(loss_op, feed_dict={x: images, y: labels})
            losses.append(loss)

        # evalute test accuracy
        test = mnist.test.images
        test = test.reshape(test.shape[0], 14, 56)
        testlabels = mnist.test.labels

        acc = sess.run(accuracy, feed_dict={x: test, y: testlabels})
        print('accuracy:  ', acc)

    return losses


def run():

    # load the mnist data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    loss = lstm(mnist)

    # plot loss verus epoc
    epoch = list(range(len(loss)))
    plt.scatter(epoch, loss)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss vs. Epoch for Feed Forward ANN', fontsize=20)
    plt.show()

# -------- go time --------
if __name__ == '__main__':
    run()
