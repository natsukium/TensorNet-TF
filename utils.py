import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split


def down_scale(x, scale=2):
    # order 2 -> order 4
    h = int(np.sqrt(x.shape[1]))
    img = x.astype("float32").reshape(x.shape[0], h, h, 1)
    scaled_img = tf.nn.avg_pool(img, ksize=[1, scale, scale, 1],
                                strides=[1, scale, scale, 1],
                                padding='VALID')
    h //= scale

    return tf.reshape(scaled_img, [x.shape[0], h ** 2])


def quantize(x):
    phi = tf.concat(
        [tf.expand_dims(tf.cos(x) * np.pi/2, 2),
         tf.expand_dims(tf.sin(x) * np.pi/2, 2)], 2)

    return phi


def load_mnist(one_hot=True, random_state=42):
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=one_hot)
    mnist_X = np.concatenate((mnist.train.images, mnist.test.images), axis=0)
    mnist_y = np.concatenate((mnist.train.labels, mnist.test.labels), axis=0)

    return train_test_split(mnist_X, mnist_y, test_size=0.2,
                            random_state=random_state)
