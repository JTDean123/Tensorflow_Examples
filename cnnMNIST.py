###################################################
#
# Build a CNN in tensorflow ANN to predict predict digits from MNIST data
# Jason Dean
# 09/16/17
#
###################################################



import tensorflow as tf
print(tf.__version__)
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def cnn(mnist):
    # define placeholders
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, 10])

    # conv layer 1
    # input is: [batchsize, 28, 28, 1]
    # output is:  [batchsize, 28, 28, 10]
    conv1 = tf.layers.conv2d(inputs=x,
                             filters=10,
                             kernel_size=[4, 4],
                             padding="same",
                             activation=tf.nn.relu)

    # pool layer 1
    # input is: [batchsize, 28, 28, 10]
    # output is:  [batchsize, 14, 14, 10]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # conv layer 2
    # input is: [batchsize, 14, 14, 10]
    # output is:  [batchsize, 14, 14, 40]
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=40,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)

    # pool layer 2
    # input is: [batchsize, 14, 14, 40]
    # output is:  [batchsize, 7, 7, 40]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # last layer, fully connected
    flattened = tf.reshape(pool2, [-1, 7 * 7 * 40])
    dense = tf.layers.dense(inputs=flattened, units=512, activation=tf.nn.relu)

    # logits layer for prediction
    y_out = tf.layers.dense(inputs=dense, units=10)

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

        num_train = 10000
        features = mnist.train.images[0:num_train]
        features = features.reshape(features.shape[0], 28, 28, 1)

        # train!
        for i in range(epochs):
            sess.run(train_step, feed_dict={x: features, y: mnist.train.labels[0:num_train]})
            losses = sess.run(cross_entropy, feed_dict={x: features, y: mnist.train.labels[0:num_train]})
            # print(losses)
            loss.append(losses)

        # evaluate test data
        features = mnist.test.images[0:int(num_train / 10)]
        features = features.reshape(features.shape[0], 28, 28, 1)
        acc = sess.run(accuracy, feed_dict={x: features, y: mnist.test.labels[0:int(num_train / 10)]})
        print("test accuracy:  ", acc)

    return loss


def run():

    # load the mnist data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # train the model
    loss = cnn(mnist)

    # plot loss verus epoc
    epoch = list(range(len(loss)))
    plt.scatter(epoch, loss)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss vs. Epoch for CNN', fontsize=20)
    plt.show()


# -------- go time --------
if __name__ == '__main__':
    run()
