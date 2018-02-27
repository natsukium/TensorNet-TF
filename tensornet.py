import tensorflow as tf


class TensorNetworks(object):
    def __init__(self, X, m, l=10, seed=1):
        self.X = tf.transpose(tf.convert_to_tensor(X, dtype=tf.float32),
                              perm=[1, 0, 2])
        self.n, _, self.d = self.X.shape
        self.j = self.n // 2
        self.m = m
        self.l = l
        self.seed = seed

    def __call__(self):
        a1, an, ai, aj = self._weight()

        edge_l = tf.einsum('jk,ik->ij', a1, self.X[0])  # shape (N, m)
        block_l = tf.scan(
            lambda a, x: tf.einsum('ijk,li,lk->lj', x[0], a, x[1]),
            (ai[:self.j-1], self.X[1:self.j]), initializer=edge_l)

        edge_r = tf.einsum('jk,ik->ij', an, self.X[-1])
        block_r = tf.scan(
            lambda a, x: tf.einsum('ijk,li,lk->lj', x[0], a, x[1]),
            (ai[::-1][:self.j], self.X[::-1][1:self.j+1]), initializer=edge_r)

        flx = tf.einsum('ijkl,mi,mj,mk->ml',
                        aj, block_l[-1], block_r[-1], self.X[self.j])

        return flx

    def _weight(self):
        initializer = tf.orthogonal_initializer(seed=self.seed)

        with tf.variable_scope('tensor'):
            a1 = tf.get_variable(name='a1', shape=(self.m, self.d),
                                 initializer=initializer)  # Left tensor

            an = tf.get_variable(name='an', shape=(self.m, self.d),
                                 initializer=initializer)  # Right tensor

            ai = tf.transpose(
                tf.get_variable(name='ai',
                                shape=(self.m, self.m, self.d, self.n-3),
                                initializer=initializer),
                perm=[3, 0, 1, 2])  # Central tensor

            aj = tf.get_variable(name='aj',
                                 shape=(self.m, self.m, self.d, self.l),
                                 initializer=initializer)  # Order 4 tensor

        return a1, an, ai, aj
