###################################################
#
# Build a feed forward ANN in tensorflow ANN to predict digits from MNIST data
# Jason Dean
# 09/16/17
#
###################################################


import tensorflow as tf
print(tf.__version__)
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def ann(mnist):
    # define placeholders
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    # build a netowrk with 1 hidden layer and 25 nodes
    w1 = tf.Variable(tf.random_normal([784, 25], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([25], stddev=0.01))

    # build output layer
    output = tf.add(tf.matmul(x, w1), b1)

    # activation
    activated = tf.nn.relu(output)

    # second layer
    w2 = tf.Variable(tf.random_normal([25, 10], stddev=0.01))
    b2 = tf.Variable(tf.random_normal([10], stddev=0.01))

    # output of second layer
    y_out = tf.add(tf.matmul(activated, w2), b2)

    # add cross entropy loss function
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_out))

    # optimize like a boss
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    # some metric nodes
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_out, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    epochs = 100

    with tf.Session() as sess:
        # initiliaze variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        loss = []

        # train!
        for i in range(epochs):
            sess.run(train_step, feed_dict={x: mnist.train.images, y: mnist.train.labels})
            loss.append(sess.run(cross_entropy, feed_dict={x: mnist.train.images, y: mnist.train.labels}))

        # evaluate test data
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("test accuracy:  ", acc)

    return loss


def run():

    # load the mnist data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    loss = ann(mnist)

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