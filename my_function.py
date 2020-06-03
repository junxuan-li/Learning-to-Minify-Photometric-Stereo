import keras

from keras import backend as K
import tensorflow as tf
from keras.constraints import Constraint
from keras.callbacks import Callback
import os
import numpy as np
import matplotlib.pyplot as plt


def mean_sqrt_error(y_true, y_pred):
    abss = K.abs(y_pred - y_true)
    return K.mean(K.sqrt(abss), axis=-1)


def weight_crossentropy(y_true, y_pred):
    eps = 1e-10
    a = K.clip(1 - K.abs(y_pred - y_true) / 1.2, eps, 1. - eps)
    return K.mean(-K.log(a), axis=-1)


def focal_weight_crossentropy(y_true, y_pred):
    gamma = 2
    eps = 1e-10
    b = K.abs(y_pred - y_true)
    a = K.clip(1 - b / 1.2, eps, 1. - eps)

    return K.mean(- K.pow(b, gamma) * K.log(a), axis=-1)


def mean_angular_error(y_true, y_pred):
    d = K.clip(K.sum(y_true * y_pred, -1), -1+K.epsilon(), 1-K.epsilon())
    ang = 180 * tf.acos(d) / 3.1415926
    return K.mean(ang, axis=-1)


def cosine_similarity_error(y_true, y_pred):
    d = K.sum(y_true * y_pred, -1)
    return K.mean(1-d, axis=-1)


class LogRegularizer(object):
    def __init__(self, k=0.):
        self.k = K.cast_to_floatx(k)

    def __call__(self, x):
        return K.sum(self.k * K.log(1 + x))

    def get_config(self):
        return {'k': float(self.k)}


class MaxL2Regularizer(object):
    def __init__(self, k=0.):
        self.k = K.cast_to_floatx(k)

    def __call__(self, x):
        m = K.max(x)
        l = m * x - 0.5 * x * x
        return K.sum(self.k * l)

    def get_config(self):
        return {'k': float(self.k)}


class MaxL2Regularizer2(object):
    def __init__(self, k=0.):
        self.k = K.cast_to_floatx(k)

    def __call__(self, x):
        m = 0.5 / K.max(x)
        l = x - m * K.square(x)
        return K.sum(self.k * l)

    def get_config(self):
        return {'k': float(self.k)}


class MaxL2Regularizer3(object):
    def __init__(self, k=0.):
        self.k = K.cast_to_floatx(k)

    def __call__(self, x):
        m = 1 / K.max(x)
        l = x - m * K.square(x) + K.pow(x, 3) / (3 * m * m)
        return K.sum(self.k * l)

    def get_config(self):
        return {'k': float(self.k)}


class MaxL2Regularizerk(object):
    def __init__(self, k=0., p=3):
        self.k = K.cast_to_floatx(k)
        self.p = K.cast_to_floatx(p)

    def __call__(self, x):
        m = K.max(x)
        a = 2 - x / m
        p = self.p + 1
        l = m * (K.pow(K.constant(value=2), p) - K.pow(a, p)) / p
        return K.sum(self.k * l)

    def get_config(self):
        return {'k': float(self.k)}


class NonNeg_MaxMinNorm(Constraint):
    def __init__(self, min_value=0.0, max_value=1.0, rate=1.0):
        self.min_value = min_value
        self.max_value = max_value
        self.rate = rate

    def __call__(self, w):
        # non negative
        w *= K.cast(K.greater_equal(w, 0.), K.floatx())
        # min max norm
        norms = K.sqrt(K.sum(K.square(w)))
        desired = (self.rate * K.clip(norms, self.min_value, self.max_value) +
                   (1 - self.rate) * norms)
        w *= (desired / (K.epsilon() + norms))
        return w

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value,
                'rate': self.rate}


class NonNeg_MinTopK(Constraint):
    def __init__(self, min_value=0.0, topk=10, rate=1):
        self.min_value = min_value
        self.topk = topk
        self.rate = rate

    def __call__(self, w):
        # non negative
        w *= K.cast(K.greater_equal(w, 0.), K.floatx())
        # min max norm
        top_k, _ = tf.nn.top_k(K.flatten(w), k=self.topk)
        desired = (self.rate * K.clip(top_k[-1], self.min_value, 100) +
                   (1 - self.rate) * top_k[-1])
        w *= (desired / (K.epsilon() + top_k[-1]))
        return w

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.topk,
                'rate': self.rate}


class Connection_map_plot(Callback):
    def __init__(self, save_path):
        super(Connection_map_plot, self).__init__()
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        c_map_w = self.model.get_weights()[0][0, :, :, 0]
        np.save(os.path.join(self.save_path, "cm_%03d.npy" % (epoch + 1)), c_map_w)

    def on_train_end(self, logs=None):
        c_map_w = self.model.get_weights()[0][0, :, :, 0]
        f = plt.figure()
        plt.imshow(c_map_w)
        f.savefig(os.path.join(self.save_path, "map.png"))
        plt.close(f)


def normalize(x):
    return K.l2_normalize(x, axis=-1)


def update_ZeroMap(ws, percent_keep=1):
    zero_map = ws[1]
    conn_map = ws[0]

    threshold = np.percentile(conn_map, (1 - percent_keep) * 100)
    zero_map[np.where(conn_map < threshold)] = 0
    return


def update_ZeroMap_from_npy(ws, npy_path=""):
    zm = np.load(npy_path).reshape((1, 14, 14, 1))
    zm[np.where(zm > 0)] = 1
    ws[1] = zm
    return


def rescale_max_one(x):
    new_max = K.max(x, axis=[-1, -2, -3], keepdims=True)
    return x / (new_max + K.epsilon())


def gen_random_connect_map(num_inputs=16, num_random_map=100):
    cm = np.zeros((num_random_map, 12, 8))
    p = np.arange(96)
    for i in range(num_random_map):
        rp = np.random.permutation(p)
        c = np.zeros(96)
        c[rp[:num_inputs]] = 1
        cm[i, :, :] = c.reshape((12, 8))

    return np.pad(cm, ((0, 0), (1, 1), (3, 3)), 'constant', constant_values=0)


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 40:
        lr = 1e-4
    if epoch > 80:
        lr = 1e-5
    return lr




