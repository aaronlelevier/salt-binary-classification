""" Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
import tensorflow as tf
import numpy as np


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def deconv2d(x, W, b, stride=2):
    # conv2d_transpose wrapper
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(
        x, W, output_shape, strides=[1, stride, stride, 1])

# hyper parameters
epochs = 2
batch_size = 32

# tf Graph input
X = tf.placeholder(tf.float32, [None, 128, 128, 1])
Y = tf.placeholder(tf.float32, [None, 128, 128, 1])

# weights & biases
weights = {
    'w1': tf.Variable(tf.random_normal([3, 3, 1, 16])),
    'w2': tf.Variable(tf.random_normal([3, 3, 16, 32])),
    'w3': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    'w4': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'w5': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'w6': tf.Variable(tf.random_normal([3, 3, 256, 128])),
    'w7': tf.Variable(tf.random_normal([3, 3, 128, 64])),
    'w8': tf.Variable(tf.random_normal([3, 3, 64, 32])),
    'w9': tf.Variable(tf.random_normal([3, 3, 32, 16])),
    'w10': tf.Variable(tf.random_normal([3, 3, 16, 1])),
}

biases = {
    # conv
    'b1': tf.Variable(tf.random_normal([16])),
    'b2': tf.Variable(tf.random_normal([32])),
    'b3': tf.Variable(tf.random_normal([64])),
    'b4': tf.Variable(tf.random_normal([128])),
    # middle
    'b5': tf.Variable(tf.random_normal([256])),
    # deconv
    'b6': tf.Variable(tf.random_normal([128])),
    'b7': tf.Variable(tf.random_normal([64])),
    'b8': tf.Variable(tf.random_normal([32])),
    'b9': tf.Variable(tf.random_normal([16])),
    'b10': tf.Variable(tf.random_normal([1])),
}

def conv_net(x, weights, bias):
    # 128 -> 64
    conv1 = conv2d(x, weights['w1'], biases['b1'])
    conv1 = maxpool2d(conv1)
    
    # 64 -> 32
    conv2 = conv2d(conv1, weights['w2'], biases['b2'])
    conv2 = maxpool2d(conv2)

    # 32 -> 16
    conv3 = conv2d(conv2, weights['w3'], biases['b3'])
    conv3 = maxpool2d(conv3)

    # 16 -> 8
    conv4 = conv2d(conv3, weights['w4'], biases['b4'])
    conv4 = maxpool2d(conv4)

    # middle
    convm = conv2d(conv4, weights['w5'], biases['b5'])

    # 8 -> 16
    deconv4 = deconv2d(convm, weights['w6'], biases['b6'])

    # 16 -> 32
    deconv3 = deconv2d(deconv4, weights['w7'], biases['b7'])

    # 32 -> 64
    deconv2 = deconv2d(deconv3, weights['w8'], biases['b8'])

    # 64 -> 128
    deconv1 = deconv2d(deconv2, weights['w9'], biases['b9'])

    # flatten filters for final output
    # (None, 128, 128, 16) -> (None, 128, 128, 1)
    out = deconv2d(deconv1, weights['w10'], biases['b10'])
    return out


logits = conv_net(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()

# Start training
# with tf.Session() as sess:
isess.run(init)

for e in range(epochs):
    print(f'epoch: {e+1}')
    shuffle_x, shuffle_y = shuffle(x_train, y_train)
    iterations = np.int(np.ceil(shuffle_x.shape[0] / batch_size))

    for step in tqdm(range(iterations)):
        start = step * batch_size
        stop = (step+1) * batch_size
        batch_x, batch_y = shuffle_x[start:stop], shuffle_y[start:stop]

        isess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        # if step % 10 == 0:
        # Calculate batch loss and accuracy
        loss, acc = isess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                             Y: batch_y})
        print("Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))
