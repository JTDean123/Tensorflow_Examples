###################################################
#
# Build a feed forward in tensorflow ANN to predict flower type based on features in the iris data set
# Jason Dean
# 09/16/17
#
###################################################

import tensorflow as tf
print(tf.__version__)
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets


def getIris():

    # load iris data
    iris = datasets.load_iris()

    # turn this dataset into a dataframe
    iris_pd = pd.DataFrame()

    for i, j in enumerate(iris.feature_names):
        iris_pd[j] = iris.data[:, i]

    # add label
    iris_pd['target'] = iris.target
    iris_pd.sample(10)

    for i, j in enumerate(iris.feature_names):
        iris_pd[j] = iris.data[:, i]

    iris_pd['target'] = iris.target

    return iris_pd


def ann(iris_pd):
    # let us make a simple neural network to predict target
    # first a test/train split
    X_train, X_test, y_train, y_test = train_test_split(iris_pd.iloc[:,0:4], iris_pd.iloc[:,-1], test_size=0.33, random_state=42)


    # one hot encode the labels
    y_train = pd.get_dummies(y_train, prefix=['target'])
    y_test = pd.get_dummies(y_test, prefix=['target'])

    # placeholder for the input and the ground truth classes
    # note there are 4 features and three classes
    x = tf.placeholder(tf.float32, [None, 4])
    y = tf.placeholder(tf.float32, [None, 3])

    # variables for the hidden layer - we will use 25 nodes in the hidden layer
    w1 = tf.Variable(tf.random_normal([4, 25], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([25], stddev=0.01))

    # variables for the output layer
    w2 = tf.Variable(tf.random_normal([25,3], stddev=0.01))
    b2 = tf.Variable(tf.random_normal([3], stddev=0.01))

    # set up the forward passes
    hidden1 = tf.add(tf.matmul(x, w1), b1)
    # activation
    hidden1A = tf.nn.relu(hidden1)

    # output layer
    y_out = tf.add(tf.matmul(hidden1A, w2), b2)
    #y_out = tf.nn.softmax(output)


    # add cross entropy loss function
    cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_out))

    # optimize like a boss
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    # some metric nodes
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_out,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # some training variables
    epochs = 100

    with tf.Session() as sess:
        # initiliaze variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        loss = []

        for i in range(epochs):
            sess.run(train_step, feed_dict={x: X_train, y: y_train})
            loss.append(sess.run(cross_entropy, feed_dict={x: X_train, y: y_train}))

        # evaluate test accuracy
        print("\ntest accuracy:  ", sess.run(accuracy, feed_dict={x: X_test, y: y_test}))

    return loss


def run():

    # load data
    iris_pd = getIris()

    # run the model
    loss = ann(iris_pd)

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