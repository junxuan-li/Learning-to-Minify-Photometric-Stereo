import numpy as np
import os


def read_npy(path, num_str=None):
    ob_map = np.load(os.path.join(path, "ob_map%s.npy" % num_str))
    normal = np.load(os.path.join(path, "valid_normal%s.npy" % num_str))
    brdf = np.load(os.path.join(path, "brdf%s.npy" % num_str))
    return ob_map, normal, brdf


def combine_npy_files(path, scopeID):
    for i in scopeID:
        num_str = "%02d" % (i + 1)
        ob_map, normal, brdf = read_npy(path, num_str)
        try:
            oms = np.append(oms, ob_map, -1)
            vns = np.append(vns, normal, -1)
            bs = np.append(bs, brdf, -1)
        except NameError:
            oms, vns, bs = ob_map, normal, brdf
    return oms, vns, bs


def get_trainable_data(path, scopeID, normalize=True):
    """
    
    :param path: 
    :param low: 
    :param up: 
    :return:  (num_img, size,size);  (num_img, 3);  (num_img,num_catog)
    """
    ob_map, normal, brdf = combine_npy_files(path, scopeID)
    brdf_catog = np.zeros((100, brdf.shape[-1]), np.float32)
    for i in range(brdf.shape[-1]):
        for j in range(brdf.shape[0]):
            b_index = int(brdf[j, 0, i])
            brdf_catog[b_index, i] = brdf[j, 1, i]

    out_ob_map = np.rollaxis(ob_map, -1, 0)
    if normalize:
        out_ob_map = normalize2D(out_ob_map)  # for normalize ob_map to max=1
    return out_ob_map, np.rollaxis(normal[0:3, :], -1, 0), np.rollaxis(brdf_catog, -1, 0)


def normalize2D(map):
    """
    :param map: (num_img, size,size)
    :return: 
    """
    nor_map = np.zeros(map.shape, np.float32)
    for i in range(map.shape[0]):
        m = map[i, :, :]
        m_max = np.max(m)
        if m_max > 1e-6:
            nor_map[i, :, :] = m / m_max
    return nor_map


def shave_training():
    o = np.load("./ob_dataset/scul8_14polar/not-shaved/ob_map_train.npy")
    o[:, :, 0:3] = 0
    o[:, :, -3:] = 0
    np.save("./ob_dataset/scul8_14polar/shaved/ob_map_train.npy", o)

    # o = np.load("./ob_dataset/4_14polar_normalize/not-shaved/ob_map_val.npy")
    # o[:, :, 0:3] = 0
    # o[:, :, -3:] = 0
    # np.save("./ob_dataset/4_14polar_normalize/shaved/ob_map_val.npy", o)


if __name__ == "__main__":
    data_path = "/media/li191/Backup/PhotometricStereoDataset/my_ps_dataset/ob_dataset/3_14polar"
    path = "./ob_dataset/" + os.path.basename(data_path)
    normalize = True
    if normalize:
        path = path + '_normalize'
    if not os.path.exists(path):
        os.mkdir(path)

    oms1, vns1, bs1 = get_trainable_data(data_path, range(0,10), normalize=normalize)
    np.save(os.path.join(path, "ob_map_train.npy"), oms1)
    np.save(os.path.join(path, "valid_normal_train.npy"), vns1)
    np.save(os.path.join(path, "brdf_train.npy"), bs1)
    del oms1, vns1, bs1
    # oms2, vns2, bs2 = get_trainable_data(data_path, range(9,10), normalize=normalize)
    # np.save(os.path.join(path, "ob_map_val.npy"), oms2)
    # np.save(os.path.join(path, "valid_normal_val.npy"), vns2)
    # np.save(os.path.join(path, "brdf_val.npy"), bs2)
