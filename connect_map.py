import keras
from keras.engine import Layer
from keras import regularizers
from keras import backend as K
from my_function import NonNeg_MaxMinNorm
import numpy as np
import tensorflow as tf


class ConnectMap(Layer):
    def __init__(self,
                 kernel_regularizer=None,
                 kernel_trainable=False,
                 kernel_constraint=keras.constraints.NonNeg(),
                 **kwargs):
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_trainable = kernel_trainable
        self.kernel_constraint = kernel_constraint
        super(ConnectMap, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = (1, input_shape[1], input_shape[2], 1)  # channel last
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='c_map',
                                      shape=weight_shape,
                                      initializer='ones',
                                      regularizer=self.kernel_regularizer,
                                      trainable=self.kernel_trainable,
                                      constraint=self.kernel_constraint)
        self.zero_map = self.add_weight(name='zero_map',
                                        shape=weight_shape,
                                        initializer='ones',
                                        trainable=False,
                                        constraint=keras.constraints.NonNeg())

        super(ConnectMap, self).build(input_shape)

    def call(self, x):
        return x * self.kernel * self.zero_map

    def compute_output_shape(self, input_shape):
        return input_shape

    # def update_zero_map(self, percent_keep=1):
    #     zero_map = K.get_value(self.zero_map)
    #     conn_map = K.get_value(self.kernel)
    #
    #     threshold = np.percentile(conn_map, percent_keep*100)
    #     zero_map[np.where(conn_map < threshold)] = 0
    #     K.set_value(self.zero_map, zero_map)
    #     return

