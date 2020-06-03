import numpy as np
import cv2
import scipy.io as scio
import os
import glob

from read_training_dataset import normalize2D


def read_testing_dataset(dirpath, w_size=14, polar_cor=True):
    object_name = os.path.basename(dirpath)
    if polar_cor:
        npy_path = os.path.join("./testing_data", str(w_size), object_name)
    else:
        npy_path = os.path.join("./testing_data", "xy", str(w_size), object_name)
    ob_map_path = npy_path + "_ob_map.npy"
    nor_path = npy_path + "_nor.npy"
    v2inx_path = npy_path + "_v2inx.npy"
    im_shape_path = npy_path + "_shape.npy"
    if os.path.exists(ob_map_path):
        ob_map = np.load(ob_map_path)
        nor = np.load(nor_path)
        v2inx = np.load(v2inx_path)
        im_shape = np.load(im_shape_path)
    else:
        ob_map, nor, v2inx, im_shape = get_testing_dataset(dirpath, w_size, polar_cor)
        np.save(ob_map_path, normalize2D(ob_map))
        np.save(nor_path, nor)
        np.save(v2inx_path, v2inx)
        np.save(im_shape_path, im_shape)
    nor[np.where(np.isnan(nor))] = 0
    return normalize2D(ob_map), nor, v2inx, im_shape


def get_testing_dataset(dirpath, w_size=14, polar_cor=True):
    """
    
    :param polar_cor: build ob_map base on polar coordinate or not
    :param w_size: ob_map's window size
    :param dirpath: 
    :return ob_map: np.array   shape:(w_size,w_size, num_valid_points)  np.float32
    :return valid_normal: np.array   shape:(3, num_valid_points)   np.float32
    :return validinx: np.array   shape:(num_valid_points, 2)   np.int64  (For rebuild the normal map)
    """
    normal_path = dirpath + '/Normal_gt.png'
    mask_path = dirpath + '/mask.png'
    ld_path = dirpath + '/light_directions.txt'
    li_path = dirpath + '/light_intensities.txt'

    light_d = read_light_directions(ld_path)
    light_i = read_light_intensities(li_path)
    # normal = read_surface_normal(normal_path)
    normal = read_normal_from_png(normal_path)
    mask = read_mask(mask_path)
    images = read_images(dirpath, light_i, mask)

    validinx = np.array(np.where(mask > 0.5)).transpose()
    num_valid_points = validinx.shape[0]

    ob_map = np.zeros((w_size, w_size, num_valid_points), np.float32)
    valid_normal = np.zeros((3, num_valid_points), np.float32)
    for i in range(num_valid_points):
        ob_seq = images[validinx[i, 0], validinx[i, 1], :]
        ob_map[:, :, i] = get_ob_map(ob_seq, light_d, w_size, polar_cor)
        valid_normal[:, i] = normal[validinx[i, 0], validinx[i, 1], :]

    return np.rollaxis(ob_map, -1, 0), np.rollaxis(valid_normal[0:3, :], -1, 0), validinx, mask.shape


def read_light_directions(filepath):
    """
    Return normalized light directions. 
    :param filepath: String
    :return: np.array   shape:(num_lights, 3)  dtype=float32
    """
    f = open(filepath)
    data = f.read()
    f.close()
    lines = data.strip().split('\n')
    for i in range(len(lines)):
        lines[i] = lines[i].split(' ')
    temp = np.array(lines)
    light_direction = temp.astype(np.float32)
    norm = normalized(light_direction, 1)
    return norm


def read_light_intensities(filepath):
    f = open(filepath)
    data = f.read()
    f.close()
    lines = data.strip().split('\n')
    for i in range(len(lines)):
        lines[i] = lines[i].split(' ')
    temp = np.array(lines)
    light_i = temp.astype(np.float32)
    return light_i


def read_surface_normal(filepath):
    """
    Read normalized surface normal from a *.mat file.  Shape: (512,612,3)
    :param filepath: String
    :return: np.array   shape:(height , width , 3)   dtype=float32
    """
    data = scio.loadmat(filepath)
    n = data['Normal_gt']
    normal = n.astype(np.float32)
    norm = normalized(normal, 2)
    return norm

def read_normal_from_png(filepath):
    img = cv2.imread(filepath)[:,:,::-1]
    normal = img/255. * 2 - 1
    return normal

def read_images(filepath, light_i, mask):
    """
    Output will first be normalized to 1. Then divided by lighting intensities. 
    Finally convert to single channel.
    :param filepath: 
    :param light_i: 
    :param mask: 
    :return: np.array   shape:(height , width , num_lights)   dtype=float32
    """
    num_lights = light_i.shape[0]
    images = np.zeros((mask.shape[0], mask.shape[1], num_lights), np.float32)
    for i in range(num_lights):  # i = 0 ~ 95
        image_path = filepath + '/%03d.png' % (i+1)
        # cv2 load images as BGR, convert it to RGB
        color_im = cv2.imread(image_path, -1)[:, :, ::-1] / 65535.0
        grey_im = (color_im[:, :, 0] / light_i[i, 0] + color_im[:, :, 1] / light_i[i, 1] + color_im[:, :, 2] / light_i[
            i, 2]) / 3
        images[:, :, i] = grey_im
    return images


def read_mask(filepath):
    """ Output will be  0(invalid)   or   1(valid)
    :param filepath: 
    :return: np.array   shape:(height , width)  dtype=float32
    """
    im = cv2.imread(filepath, -1) / 255
    return im.astype(np.float32)


def get_ob_map(ob_seq, light_d, w_size, polar_cor):
    """
    :param ob_seq: shape:(num_lights,)
    :param light_d: shape:(num_lights, 3)
    :param w_size: output window size
    :return: shape: (w_size,w_size)   np.float32
    """
    ob_map = np.zeros((w_size, w_size), np.float32)
    for i in range(len(ob_seq)):
        index = get_index_light_direction(light_d[i, :], polar=polar_cor, w=w_size)
        ob_map[index] = ob_seq[i]
    return ob_map


def get_index_light_direction(vector, polar=True, w=14):
    """
    :param w: window size
    :param polar: Coordinate system
    :param vector: shape: (3,); value:[x,y,z]
    :return: 
    """
    if polar:
        bound = np.pi / 4
        x = np.arcsin(vector[0] / np.sqrt(vector[0] ** 2 + vector[2] ** 2))
        y = np.arcsin(vector[1] / np.sqrt(vector[1] ** 2 + vector[2] ** 2))
    else:
        bound = np.sqrt(2) / 2
        x = vector[0]
        y = vector[1]
    x = x + bound
    y = y + bound
    g = 2 * bound / (w - 1)
    ix = np.rint(x / g).astype(np.int)
    iy = np.rint(y / g).astype(np.int)

    ix = np.max((0, ix))
    ix = np.min((w - 1, ix))
    iy = np.max((0, iy))
    iy = np.min((w - 1, iy))
    return ix, iy


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def combine_npy(nor_2d_map=True):
    ob_files = sorted(glob.glob("./testing_data/14/val/*_ob_map.npy"))
    nor_files = sorted(glob.glob("./testing_data/14/val/*_nor.npy"))
    for i in range(len(ob_files)):
        ob_map = np.load(ob_files[i])
        if nor_2d_map:
            ob_map = normalize2D(ob_map)
        normal = np.load(nor_files[i])
        try:
            oms = np.append(oms, ob_map, 0)
            vns = np.append(vns, normal, 0)
        except NameError:
            oms, vns = ob_map, normal
    np.save("./testing_data/14/val/ob_map_diligent.npy", oms)
    np.save("./testing_data/14/val/valid_normal_diligent.npy", vns)


def compute_all_observation_map():
    path = "/media/li191/Backup/PhotometricStereoDataset/DiLiGenT/DiLiGenT_train/pmsData/*PNG"
    objs = glob.glob(path)
    for o in objs:
        read_testing_dataset(o,w_size=14,polar_cor=True)



