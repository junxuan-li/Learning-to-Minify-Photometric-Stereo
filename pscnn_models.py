# From https://github.com/satoshi-ikehata/CNN-PS
# By Satoshi Ikeahta
# Paper: CNN-PS: CNN-based Photometric Stereo for General Non-Convex Surfaces, ECCV2018.
import keras
from keras import backend as K
from keras.models import Input, Model
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute, Lambda
from keras.layers import Merge, merge, Concatenate, concatenate, MaxPooling1D, multiply
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Conv1D, Conv2D, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam, Adadelta
from connect_map import ConnectMap
import my_function


def get_densenet_2d_channel_last_2dense(rows, cols, classes, kernel_regu, kernel_constraint, weight_decay=1e-4, weights_path=None):
    if weight_decay == 0:
        conv_regu = None
    else:
        conv_regu = keras.regularizers.l2(weight_decay)

    inputs1 = Input((rows, cols, 1))
    in_c_map = Activation(my_function.rescale_max_one)(inputs1)  # make sure after occlusion, input is still max one
    in_c_map = ConnectMap(kernel_regularizer=kernel_regu, kernel_trainable=kernel_regu is not None, kernel_constraint=kernel_constraint)(in_c_map)

    # commented the next line if input is full (i.e. 96 images)
    in_c_map = Activation(my_function.rescale_max_one)(in_c_map)  # make sure after connection, input is still max one

    x0 = in_c_map
    x1 = Conv2D(16, (3, 3), padding='same', name='conv1', kernel_regularizer=conv_regu, bias_regularizer=conv_regu)(x0)

    # 1st Denseblock
    x1a = Activation('relu')(x1)
    x2 = Conv2D(16, (3, 3), padding='same', name='conv2', kernel_regularizer=conv_regu, bias_regularizer=conv_regu)(x1a)
    x2 = Dropout(0.2)(x2)

    xc1 = concatenate([x2, x1], axis=3)

    xc1a = Activation('relu')(xc1)
    x3 = Conv2D(16, (3, 3), padding='same', name='conv3', kernel_regularizer=conv_regu, bias_regularizer=conv_regu)(xc1a)
    x3 = Dropout(0.2)(x3)

    xc2 = concatenate([x3,x2,x1], axis=3)

    # Transition
    xc2a = Activation('relu')(xc2)
    x4 = Conv2D(48, (1, 1), padding='same', name='conv4', kernel_regularizer=conv_regu, bias_regularizer=conv_regu)(xc2a)
    x4 = Dropout(0.2)(x4)
    x1 = AveragePooling2D((2, 2), strides=(2, 2))(x4)

    # 2nd Dense block
    x1a = Activation('relu')(x1)
    x2 = Conv2D(16, (3, 3), padding='same', name='conv5', kernel_regularizer=conv_regu, bias_regularizer=conv_regu)(x1a)
    x2 = Dropout(0.2)(x2)

    xc1 = concatenate([x2, x1], axis=3)

    xc1a = Activation('relu')(xc1)
    x3 = Conv2D(16, (3, 3), padding='same', name='conv6', kernel_regularizer=conv_regu, bias_regularizer=conv_regu)(xc1a)
    x3 = Dropout(0.2)(x3)
    xc2 = concatenate([x3,x2,x1], axis=3)

    xc2a = Activation('relu')(xc2)
    x4 = Conv2D(80, (1, 1), padding='same', name='conv7', kernel_regularizer=conv_regu, bias_regularizer=conv_regu)(xc2a)

    x = Flatten()(x4)
    x = Dense(128,activation='relu',name='dense1b', kernel_regularizer=conv_regu, bias_regularizer=conv_regu)(x)
    x = Dense(classes, name='dense2', kernel_regularizer=conv_regu, bias_regularizer=conv_regu)(x)

    normalize = Lambda(lambda x: K.l2_normalize(x, axis=-1))
    x = normalize(x)

    outputs = x

    model = Model(inputs = inputs1, outputs = outputs)
    if weights_path is not None:
        model.load_weights(weights_path)
    return model
