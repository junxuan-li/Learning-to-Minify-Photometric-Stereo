import numpy as np
import os
import keras
import cv2
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
from read_diligent_dataset import read_testing_dataset
from custom_layers import Scale
from my_function import mean_angular_error, mean_sqrt_error, normalize
from connect_map import ConnectMap
from pscnn_models import get_densenet_2d_channel_last_2dense


def test_model(model, ob_map, gt_normal=None, gt_brdf=None, nor2index=None, img_shape=None):
    img_rows, img_cols = ob_map.shape[1], ob_map.shape[2]
    num_img_train = ob_map.shape[0]
    if K.image_data_format() == 'channels_first':
        ob_map = ob_map.reshape(num_img_train, 1, img_rows, img_cols)
    else:
        ob_map = ob_map.reshape(num_img_train, img_rows, img_cols, 1)
    pred_normal = model.predict(ob_map)

    gt_nor_map = vector2map(gt_normal, nor2index, img_shape)
    pred_nor_map = vector2map(pred_normal, nor2index, img_shape)
    ang_error_map = compute_angular_error(gt_nor_map, pred_nor_map)
    return pred_nor_map, gt_nor_map, ang_error_map


def nor2_to_nor3(normal):
    z2 = 1 - (normal[:, 0] ** 2) - (normal[:, 1] ** 2)
    z = np.sqrt(np.clip(z2, 0, 1))
    z = z.reshape((normal.shape[0], 1))
    xyz = np.append(normal, z, axis=-1)
    return xyz


def vector2map(v, v2i, shape):
    map = np.zeros((shape[0], shape[1], 3))
    for inx in range(v.shape[0]):
        i = v[inx, :]
        map[v2i[inx, 0], v2i[inx, 1], :] = i
    return map


def compute_angular_error(gt, pred):
    a_map = np.zeros(shape=gt.shape[0:2])
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if np.linalg.norm(gt[i, j, :]) == 0:
                continue
            if np.linalg.norm(pred[i, j, :]) == 0:
                # print(pred[i, j, :], " gt: ", gt[i, j, :])
                a_map[i, j] = 180
                continue
            a_map[i, j] = 180 * np.arccos(np.clip(np.dot(gt[i, j, :], pred[i, j, :])/(np.linalg.norm(gt[i, j, :])*np.linalg.norm(pred[i, j, :])), -1, 1)) / np.pi
    return a_map


def compute_mean_error_all_diligent(model, w_size=14, data_path="./DiLiGenT"):
    path = data_path
    objs = sorted(os.listdir(path))
    mae = np.zeros(10)
    pred_n_map = np.zeros((10, 512, 612, 3))
    gt_n_map = np.zeros((10, 512, 612, 3))
    ang_error_map = np.zeros((10, 512, 612))
    for i, o in enumerate(objs):
        data_path = os.path.join(path, o)
        ob_map, gt_normal, v2inx, im_shape = read_testing_dataset(data_path, w_size=w_size)
        pred_n_map[i, :, :, :], gt_n_map[i, :, :, :], ang_error_map[i, :, :] = test_model(model, ob_map, gt_normal,
                                                                                          nor2index=v2inx,
                                                                                          img_shape=im_shape)
        # plt.matshow(ang_error_map[i, :, :])
        mae[i] = np.sum(ang_error_map[i, :, :]) / np.where(ang_error_map[i, :, :] > 1e-5)[0].shape[0]
        print("MAE for %s: %f" % (os.path.basename(data_path), mae[i]))
    print("Mean MAE: %f" % np.mean(mae))
    return pred_n_map, gt_n_map, ang_error_map, mae


def show_diff_ang_error_map(map1, map2, m=25, save_path=None):
    """
    Red for map2 better. Blue for map1 better
    """
    if len(map1.shape) == 3:
        for i in range(map1.shape[0]):
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.imsave(os.path.join(save_path, '%d_diff.png' % i), map1[i,:,:] - map2[i,:,:], vmin=-m, vmax=m, cmap="bwr")
            else:
                plt.figure(i)
                plt.imshow(map1[i, :, :] - map2[i, :, :], vmin=-m, vmax=m, cmap="bwr")
                plt.show()
    else:
        plt.figure()
        plt.imshow(map1 - map2, vmin=-m, vmax=m, cmap="bwr")
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.imsave(os.path.join(save_path, 'diff.png'),map1 - map2, vmin=-m, vmax=m, cmap="bwr")



def get_model_from_path(path):
    if "weights" in path:
        model = get_densenet_2d_channel_last_2dense(14, 14,
                                                    classes=3,
                                                    weights_path=path,
                                                    kernel_regu=None,
                                                    kernel_constraint=None,
                                                    weight_decay=0)
        model.compile(loss=keras.losses.mean_squared_error,
                      optimizer=keras.optimizers.Adam(),
                      metrics=[mean_angular_error])
    else:
        model = load_model(path, custom_objects={"Scale": Scale,
                                                 "mean_angular_error": mean_angular_error,
                                                 "mean_sqrt_error": mean_sqrt_error,
                                                 "ConnectMap": ConnectMap,
                                                 "normalize": normalize})
    return model


def write_normal_to_png(pred_n_map, gt_n_map, ang_error_map, mae=None, path='./test_results/synt-inputs-10', maxerror=90):
    if not os.path.exists(path):
        os.makedirs(path)
    if len(gt_n_map.shape) == 4:
        for i in range(gt_n_map.shape[0]):
            p_n = pred_n_map[i,:,:,:].copy()
            g_n = gt_n_map[i,:,:,:].copy()
            p_n_img = ((p_n + 1.) / 2. * 255.).astype(np.uint8)
            g_n_img = ((g_n + 1.) / 2. * 255.).astype(np.uint8)

            ae_map = ang_error_map[i,:,:].copy()
            ae_map = np.clip(ae_map / maxerror, 0,1)
            ae_img = angular_error_convert_color(ae_map)

            mask = np.sum(g_n ** 2, -1)
            p_n_img[np.where(mask == 0)] = 255
            g_n_img[np.where(mask == 0)] = 255
            ae_img[np.where(mask == 0)] = 255
            cv2.imwrite(os.path.join(path,  "%02d_pred.png" % i), p_n_img[:, :, ::-1])
            cv2.imwrite(os.path.join(path,  "%02d_gt.png"% i), g_n_img[:, :, ::-1])
            cv2.imwrite(os.path.join(path,  "%02d_err.png"% i), ae_img[:, :, ::-1])
    else:
        p_n = pred_n_map
        g_n = gt_n_map
        p_n_img = ((p_n + 1.) / 2. * 255.).astype(np.uint8)
        g_n_img = ((g_n + 1.) / 2. * 255.).astype(np.uint8)

        ae_map = ang_error_map
        ae_map = np.clip(ae_map / maxerror, 0, 1)
        ae_img = angular_error_convert_color(ae_map)

        mask = np.sum(g_n ** 2, -1)
        p_n_img[np.where(mask == 0)] = 255
        g_n_img[np.where(mask == 0)] = 255
        ae_img[np.where(mask == 0)] = 255
        cv2.imwrite(os.path.join(path, "single_pred.png"), p_n_img[:, :, ::-1])
        cv2.imwrite(os.path.join(path, "single_gt.png"), g_n_img[:, :, ::-1])
        cv2.imwrite(os.path.join(path, "single_err.png"), ae_img[:, :, ::-1])
    if mae is None:
        return
    ld_file_name = "mae.txt"
    txt_file = open(os.path.join(path, ld_file_name), 'w')
    for ld in mae:
        txt_file.write("%f\n" % ld)
    txt_file.close()


def angular_error_convert_color(ae):
    out = np.zeros((ae.shape[0], ae.shape[1], 3))
    for i in range(ae.shape[0]):
        for j in range(ae.shape[1]):
            out[i,j,:] = scalars_to_rgb(ae[i,j])

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def scalars_to_rgb(f):
    a = (1 - f) / 0.25
    X = np.floor(a)
    Y = np.floor(255 * (a - X))
    if X==0:
        r = 255
        g = Y
        b = 0
    elif X==1:
        r = 255 - Y
        g = 255
        b = 0
    elif X==2:
        r = 0
        g = 255
        b = Y
    elif X == 3:
        r = 0
        g = 255 - Y
        b = 255
    elif X == 4:
        r = 0
        g = 0
        b = 255
    return r,g,b


if __name__ == "__main__":
    test_data_path = "/media/li191/Backup/PhotometricStereoDataset/DiLiGenT/DiLiGenT_train/pmsData/"

    model_path = "./learned_model/6/pscnn.weights.h5"
    model = get_model_from_path(model_path)
    model_path = "./learned_model/8/pscnn.weights.h5"
    model2 = get_model_from_path(model_path)
    model_path = "./learned_model/10/pscnn.weights.h5"
    model3 = get_model_from_path(model_path)

    pred_n_map, gt_n_map, ang_error_map, mae = compute_mean_error_all_diligent(model, w_size=14,data_path=test_data_path)
    pred_n_map2, gt_n_map2, ang_error_map2, mae2 = compute_mean_error_all_diligent(model2, w_size=14,data_path=test_data_path)
    pred_n_map3, gt_n_map3, ang_error_map3, mae3 = compute_mean_error_all_diligent(model3, w_size=14, data_path=test_data_path)

    #  When testing on full inputs, you need to comment line 29 in "pscnn_models.py" to get the correct model.
    # model_path = "./learned_model/96/pscnn.weights.h5"
    # model4 = get_model_from_path(model_path)
    # pred_n_map4, gt_n_map4, ang_error_map4, mae4 = compute_mean_error_all_diligent(model4, w_size=14, data_path=test_data_path)

    #  To write the tseting results into .png files:
    # write_normal_to_png(pred_n_map, gt_n_map, ang_error_map, mae, path='./test_results/inputs-6', maxerror=90)
    # write_normal_to_png(pred_n_map2, gt_n_map2, ang_error_map2, mae2, path='./test_results/inputs-8', maxerror=90)
    # write_normal_to_png(pred_n_map3, gt_n_map3, ang_error_map3, mae3, path='./test_results/inputs-10', maxerror=90)

    # ws = model.get_weights()  # show connection map
    # plt.matshow(ws[0][0, :, :, 0])

    # pred_n_map, gt_n_map, ang_error_map = test_model(model, ob_map, gt_normal, nor2index=v2inx, img_shape=im_shape)
    # plt.matshow(ang_error_map)
    # mae = np.sum(ang_error_map) / np.where(ang_error_map > 1e-5)[0].shape[0]
    # print("MAE for %s: %f" % (os.path.basename(data_path), mae))
    #
    # plt.matshow(pred_n_map[:, :, 0])
    # plt.matshow(pred_n_map[:, :, 1])
    # plt.matshow(pred_n_map[:, :, 2])
    # plt.matshow(gt_n_map[:, :, 0])
    # plt.matshow(gt_n_map[:, :, 1])
    # plt.matshow(gt_n_map[:, :, 2])
