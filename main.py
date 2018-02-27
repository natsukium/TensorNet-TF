import tensorflow as tf
from sklearn.utils import shuffle
from tqdm import tqdm

from tensornet import TensorNetworks
from utils import down_scale, quantize, load_mnist
from config import FLAGS


def train():
    config = tf.ConfigProto(
        device_count={"GPU": FLAGS.GPU},
        log_device_placement=True)
    random_state = 42

    print("loading data...")
    X, testX, Y, testY = load_mnist()

    X = quantize(down_scale(X, FLAGS.scale))
    testX = quantize(down_scale(testX, FLAGS.scale))

    print("defining graph...")
    x = tf.placeholder(tf.float32, [None, X.shape[1], X.shape[2]])
    delta = tf.placeholder(tf.float32, [None, 10])
    m = FLAGS.rank

    TN = TensorNetworks(x, m)
    flx = TN()

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.reduce_sum(
            tf.square(flx - delta) + 1e-10, axis=1)) / 2

    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(flx, 1), tf.argmax(delta, 1)), "float"))

    with tf.name_scope("train"):
        train = tf.train.AdamOptimizer(FLAGS.eta).minimize(loss)
        accuracy_summary_train = tf.summary.scalar("train_accuracy", accuracy)
        loss_summary_train = tf.summary.scalar("train_loss", loss)

    with tf.name_scope("test"):
        accuracy_summary_test = tf.summary.scalar("test_accuracy", accuracy)
        loss_summary_test = tf.summary.scalar("test_loss", loss)

    n_epoch = FLAGS.num_epochs
    batch_size = FLAGS.batch_size
    n_batches = X.shape[0] // batch_size

    print("start train...")
    init = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        X = sess.run(X)
        file_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        sess.run(init)
        train_list = [accuracy, accuracy_summary_train, loss_summary_train]
        test_list = [accuracy, accuracy_summary_test, loss_summary_test]
        for epoch in tqdm(range(n_epoch)):
            X, Y = shuffle(X, Y, random_state=random_state)
            for i in tqdm(range(n_batches)):
                start = i * batch_size
                end = start + batch_size
                sess.run(
                    train, feed_dict={x: X[start:end], delta: Y[start:end]})

            train_summaries = sess.run(train_list,
                                       feed_dict={x: X, delta: Y})
            for i in range(1, len(train_summaries)):
                file_writer.add_summary(train_summaries[i], epoch+1)
            if (epoch+1) % 10 == 0:
                test_summaries = sess.run(test_list,
                                          feed_dict={x: testX, delta: testY})
                for i in range(1, len(test_summaries)):
                    file_writer.add_summary(test_summaries[i], epoch+1)


if __name__ == "__main__":
    train()
