import numpy as np
import keras
from read_training_dataset import read_npy


def random_zero_out(ob_map):
    """
    
    :param ob_map: shape: (w,w)
    :return: Randomly zeroed out ob_map, shape:(w,w)
    """
    w_size = ob_map.shape[0]
    line_params, greater = get_random_line(w_size)
    zero_filter = get_filter(line_params, greater, w_size)
    zero_ob_map = zero_filter * ob_map
    return zero_ob_map


def random_fail_point(ob_map, k):
    r = np.random.rand(1) * k + 1 - k
    ratio_map = np.random.rand(ob_map.shape[0], ob_map.shape[1]).astype(np.float32)
    fail_filter = np.ones(ob_map.shape, np.float32)
    fail_filter[np.where(ratio_map > r)] = 0
    return ob_map * fail_filter


def get_random_line(w_size):
    """
    Line is defined: ax+by+c=0    If greater, the ax+by+c>0, [x,y]=0
    :param w_size: window size
    :return: [a,b,c], True or False
    """
    l = np.arange(4)
    np.random.shuffle(l)
    r = np.random.rand(2)
    x1, y1 = get_xy(l[0], r[0], w_size)
    x2, y2 = get_xy(l[1], r[1], w_size)
    line_params = get_line_params(x1, y1, x2, y2)

    judge = (w_size - 1) / 2 * line_params[0] + (w_size - 1) / 2 * line_params[1] + line_params[2]
    if judge > 0:
        greater = False
    else:
        greater = True
    return line_params, greater


def get_filter(line_params, greater, w_size):
    """
    Line is defined: ax+by+c=0    If greater, the ax+by+c>0, [x,y]=0
    :param w_size: window size
    :param line_params: [a,b,c]
    :param greater: boolean
    :return: shape: (w_size, w_size)
    """
    zero_filter = np.ones((w_size, w_size), np.float32)
    for i in range(w_size):
        for j in range(w_size):
            xyz = i * line_params[0] + j * line_params[1] + line_params[2]
            if greater:
                if xyz > 0:
                    zero_filter[i, j] = 0
                else:
                    pass
            else:
                if xyz < 0:
                    zero_filter[i, j] = 0
                else:
                    pass
    return zero_filter


def get_xy(l, r, w_size):
    w_size = w_size - 1
    if l == 0:
        x = 0
        y = w_size * r
    elif l == 1:
        y = 0
        x = w_size * r
    elif l == 2:
        x = w_size
        y = w_size * r
    elif l == 3:
        y = w_size
        x = w_size * r
    return x, y


def get_line_params(x1, y1, x2, y2):
    a = x2 - x1
    b = y2 - y1

    m = b
    n = -a
    p = a * y1 - b * x1
    return np.array([m, n, p])


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, datapath, data_path_suff, batch_size=32, shuffle=True, zero_outer=0, random_fail=0, sample_weight=0):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.zero_outer = zero_outer
        self.random_fail = random_fail
        self.sample_weight = sample_weight

        ob_train, self.nor_train, self.brdf_train = read_npy(datapath, data_path_suff)
        img_rows, img_cols = ob_train.shape[1], ob_train.shape[2]
        num_img_train = ob_train.shape[0]

        self.ob_train = ob_train.reshape(num_img_train, img_rows, img_cols, 1)
        self.dim = (img_rows, img_cols, 1)
        self.indexes = np.arange(ob_train.shape[0])
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.nor_train.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        nor = self.nor_train[indexes,]
        # brdf = self.brdf_train[indexes,]
        # Generate data
        if self.random_fail == 0 and self.zero_outer == 0:
            ob = self.ob_train[indexes,]
        else:
            ob = self.__data_generation(indexes)

        if self.sample_weight==0:
            return ob, nor
        else:
            z = nor[:, 2].copy()
            low_thres = 0.3
            sample_weights = ((low_thres-z)/low_thres) * self.sample_weight + 1
            sample_weights[np.where(z > low_thres)] = 1

            lar_weight = 1.5 - z
            lar_weight[np.where(z < 0.5)] = 1
            return ob, nor, sample_weights*lar_weight

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        ob = np.empty((len(indexes), *self.dim), np.float32)

        # Generate data
        for i, ID in enumerate(indexes):
            if np.random.rand(1) < self.zero_outer:
                ob[i, :, :, 0] = random_zero_out(self.ob_train[ID, :, :, 0])
            else:
                ob[i, :, :, 0] = self.ob_train[ID, :, :, 0]

            if self.random_fail > 0:
                ob[i, :, :, 0] = random_fail_point(ob[i, :, :, 0], self.random_fail)

        return ob
