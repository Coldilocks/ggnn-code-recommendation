import numpy as np
import tensorflow as tf


# 定义MLP层(3层)
# 很牛逼的变量命名以及存储方法
def init_weights(shape):
    return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (2 * np.random.rand(*shape).astype(np.float32) - 1)


class MLP(object):
    def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = dropout_keep_prob
        self.params = self.make_network_params()

    # MLP层的参数以及存储方式
    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        # in_size -> hid_sizes -> out_size
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        weights = [tf.Variable(init_weights(s), name='MLP_W_layer%i' % i)
                   for (i, s) in enumerate(weight_sizes)]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='MLP_b_layer%i' % i)
                  for (i, s) in enumerate(weight_sizes)]
        network_params = {
            "weights": weights,
            "biases": biases
        }
        return network_params

    # 这里设计了两层的神经网络，因此直接调用就可以输出last_hidden
    def __call__(self, inputs):
        acts = inputs
        for W, b in zip(self.params['weights'], self.params['biases']):
            hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob)) + b
            acts = tf.nn.relu(hid)
        last_hidden = hid
        return last_hidden
