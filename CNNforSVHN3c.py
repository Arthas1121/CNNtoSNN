'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import numpy as np

import tensorflow as tf
import scipy.io as sio
import matplotlib as plt
sess = tf.InteractiveSession()

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import it_data

#mnist = input_data.read_data_sets("tmp/data/MNIST/", one_hot=True)
train_location ='Data/train.mat'
test_location = 'Data/test.mat'


def load_train_data():
    train_dict = sio.loadmat(train_location)
    X = np.asarray(train_dict['X'])

    X_train = []
    for i in range(X.shape[3]):
        X_train.append(X[:,:,:,i])
    X_train = np.asarray(X_train)/255.0

    Y_train = train_dict['y']
    for i in range(len(Y_train)):
        if Y_train[i]%10 == 0:
            Y_train[i] = 0
    Y_train = tf.one_hot(Y_train,10)
    Y_train=tf.reshape(Y_train,[73257,10])
    return (X_train,Y_train)

def load_test_data():
    test_dict = sio.loadmat(test_location)
    X = np.asarray(test_dict['X'])

    X_test = []
    for i in range(X.shape[3]):
        X_test.append(X[:,:,:,i])
    X_test = np.asarray(X_test)/255.0

    Y_test = test_dict['y']
    for i in range(len(Y_test)):
        if Y_test[i]%10 == 0:
            Y_test[i] = 0
    Y_test = tf.one_hot(Y_test,10)
    Y_test=tf.reshape(Y_test,[26032,10])
    return (X_test,Y_test)
X_train, Y_train = load_train_data()
X_test,Y_test=load_test_data()




# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 50
display_step = 10

# Network Parameters
 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.5 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, 32,32,3]) #定义一个占位符 代表一种数据
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
   # x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def avgpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')


# Create model
def conv_net(x, weights,dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 32, 32, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'])
    # Max Pooling (down-sampling)
    conv1 = avgpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'])
    # Max Pooling (down-sampling)
    conv2 = avgpool2d(conv2, k=2)
    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'])
    # Max Pooling (down-sampling)
    conv3 = avgpool2d(conv3, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.matmul(fc1, weights['wd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
   #out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    out=tf.matmul(fc1,weights['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 3, 32],stddev=0.1)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64],stddev=0.1)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc3': tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=0.1)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.truncated_normal([4*4*64, 100],stddev=0.1)),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.truncated_normal([100, n_classes],stddev=0.1))
}

a = tf.Variable(tf.truncated_normal([1,1]))

'''biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}'''

# Construct model
pred = conv_net(x, weights,keep_prob)

# Define loss and optimizer
#交叉熵 一种表达两个向量之间差距的的指标
#reduce_mean:Computes the mean of elements across dimensions of a tensor.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#亚当算法 一种基于梯度下降的优化算法 但比较稳定
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
#argmax()返回最大值的索引
#correct_pred是一个一维的只有true or false的序列
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    current = 0;
    y_train = sess.run(Y_train)
    y_test = sess.run(Y_test)
    for i in range(1):
        if (current >= 73200):
            batch_data = X_train[current:current + 56, :, :, :]
            batch_label = y_train[current:current + 56, :]
            current = 0
        else:
            batch_data = X_train[current:current + 50, :, :, :]
            batch_label = y_train[current:current + 50, :]
            current = current + 50

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch_data, y: batch_label, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        sess.run(optimizer, feed_dict={x: batch_data, y: batch_label,
                                       keep_prob: dropout})
    WC1 = weights['wc1'].eval()
    WC2 = weights['wc2'].eval()
    WD1 = weights['wd1'].eval()
    OUT = weights['out'].eval()

    sio.savemat("W_conv1.mat", mdict={'W_conv1': WC1})
    sio.savemat("W_conv2.mat", mdict={'W_conv2': WC2})
    sio.savemat("W_fc1.mat", mdict={'W_fc1': WD1})
    sio.savemat("W_fc2.mat", mdict={'W_fc2': OUT})
    test_x = X_test[1:1000, :, :, :]
    test_y = y_test[1:1000, :]
    print("test accuracy %g" % accuracy.eval(feed_dict={x: test_x, y: test_y, keep_prob: 1.0}))
