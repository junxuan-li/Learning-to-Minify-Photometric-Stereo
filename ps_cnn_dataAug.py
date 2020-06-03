import keras

from keras.callbacks import TensorBoard, ModelCheckpoint

import densenet121
import os
from resnet import ResNet
from ps_simple_cnn import simple_cnn
from pscnn_models import get_densenet_2d_channel_last_2dense
import my_function
from my_function import mean_sqrt_error, mean_angular_error, weight_crossentropy, focal_weight_crossentropy, LogRegularizer, \
    normalize, MaxL2Regularizer, MaxL2Regularizer2, MaxL2Regularizer3, MaxL2Regularizerk, NonNeg_MaxMinNorm, Connection_map_plot
from custom_layers import Scale
from data_augment import DataGenerator
from shutil import copyfile
from connect_map import ConnectMap


def train_ps_cnn(epochs, datapath, net_type, log_save_path, continue_from_weight, keep_zero_map_rate, update_zero_map_path, update_zero_map_array, lr,lr_s, training_generator, validation_generator, kernel_regu):
    normal_classes = 3
    continue_from_model = None

    kernel_constraint = keras.constraints.NonNeg()
    weight_decay = 0

    loss_funcs = keras.losses.mean_squared_error
    metric_funcs = [mean_angular_error]
    input_shape = (14,14,1)

    if continue_from_model is None:
        if net_type == "DenseNet":
            model = densenet121.DenseNet(nb_dense_block=3, growth_rate=32, nb_filter=64, reduction=0.0,
                                         dropout_rate=0.2, weight_decay=1e-4, classes=normal_classes, weights_path=None,
                                         input_shape=input_shape)
            model.compile(loss=loss_funcs,
                          optimizer=keras.optimizers.Adam(),
                          metrics=metric_funcs)
        elif net_type == "ResNet":
            model = ResNet(input_shape=input_shape, n=3, classes=normal_classes)
            model.compile(loss=loss_funcs,
                          optimizer=keras.optimizers.Adam(lr=lr),
                          metrics=metric_funcs)
        elif net_type == "Simple":
            model = simple_cnn(input_shape,
                               classes=normal_classes,
                               weights_path=continue_from_weight,
                               kernel_regu=kernel_regu,
                               kernel_constraint=kernel_constraint,
                               weight_decay=weight_decay)
            model.compile(loss=loss_funcs,
                          optimizer=keras.optimizers.Adam(lr=lr),
                          metrics=metric_funcs)
        elif net_type == "pscnn":
            model = get_densenet_2d_channel_last_2dense(input_shape[0], input_shape[1],
                                                        classes=normal_classes,
                                                        weights_path=continue_from_weight,
                                                        kernel_regu=kernel_regu,
                                                        kernel_constraint=kernel_constraint,
                                                        weight_decay=weight_decay)
            model.compile(loss=loss_funcs,
                          optimizer=keras.optimizers.Adam(lr=lr),
                          metrics=metric_funcs)
        else:
            print("No model:", net_type)
    else:
        model = keras.models.load_model(continue_from_model, custom_objects={"Scale": Scale,
                                           "mean_angular_error": mean_angular_error,
                                           "mean_sqrt_error": mean_sqrt_error,
                                           "ConnectMap": ConnectMap,
                                           "normalize": normalize})
        model.compile(loss=loss_funcs,
                      optimizer=keras.optimizers.Adam(lr=lr),
                      metrics=metric_funcs)

    if keep_zero_map_rate != 1:
        ws = model.get_weights()
        my_function.update_ZeroMap(ws, keep_zero_map_rate)
        model.set_weights(ws)

    if update_zero_map_path is not None:
        ws = model.get_weights()
        my_function.update_ZeroMap_from_npy(ws, update_zero_map_path)
        model.set_weights(ws)

    if update_zero_map_array is not None:
        ws = model.get_weights()
        ws[1] = update_zero_map_array.reshape((1, 14, 14, 1))
        model.set_weights(ws)

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = net_type + '.{epoch:03d}.weights.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(log_save_path):
        os.makedirs(log_save_path)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=False,
                                 save_weights_only=True)
    tensorboard = TensorBoard(log_dir=log_save_path)
    call_lists = [tensorboard, checkpoint]
    if kernel_regu is not None:
        save_connect_map = Connection_map_plot(log_save_path)
        call_lists.append(save_connect_map)

    if lr_s is not None:
        call_lists.append(lr_s)

    if continue_from_model is None and continue_from_weight is None:
        initial_epoch = 0
    else:
        initial_epoch = int(os.path.basename(continue_from_weight).split('.')[1])

    copyfile(__file__, os.path.join(log_save_path, "note%03d.py" % initial_epoch))

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=len(training_generator),
                        validation_steps=len(validation_generator),
                        epochs=epochs,
                        use_multiprocessing=True,
                        workers=3,
                        max_queue_size=10,
                        verbose=1,
                        initial_epoch=initial_epoch,
                        callbacks=call_lists)
    model.save(os.path.join(log_save_path, net_type + '.%03d.h5' % epochs))
    model.save_weights(os.path.join(log_save_path, net_type + '.%03d.weights.h5' % epochs))


if __name__ == "__main__":
    batch_size = 4096

    kernel_regu = keras.regularizers.l1(1e-3) #MaxL2Regularizerk(1e-3, 1) #None #MaxL2Regularizer3(50) # #
    continue_from_weight = None #"./logs/pscnn/data_aug/polar-14-shaved/learned/16/8/pscnn.090.weights.h5"

    datapath = "./ob_dataset/3_14polar_normalize"
    net_type = "pscnn"

    lr = 1e-3
    lr_s = keras.callbacks.LearningRateScheduler(my_function.lr_schedule)

    ################
    epochs = 80
    log_save_path = './logs/' + net_type + '/data_aug/polar-14-shaved/learned/64/2'

    keep_zero_map_rate = 196 / 196
    update_zero_map_path = "./save_connect_map/learned70.npy"
    update_zero_map_array = None

    training_generator = DataGenerator(datapath, "_train", batch_size, zero_outer=0.3, random_fail=0.05, sample_weight=50)
    validation_generator = DataGenerator("./testing_data/14", "_diligent", batch_size, zero_outer=0, random_fail=0)

    # training_generator = DataGenerator("./testing_data/14/train", "_diligent", batch_size, zero_outer=0.05, random_fail=0)
    # validation_generator = DataGenerator("./testing_data/14/val", "_diligent", batch_size, zero_outer=0, random_fail=0)

    train_ps_cnn(epochs, datapath, net_type, log_save_path, continue_from_weight, keep_zero_map_rate,
                 update_zero_map_path, update_zero_map_array, lr, lr_s, training_generator, validation_generator,
                 kernel_regu)

    # tensorboard --logdir=/media/li191/Backup/Junxuan_tf_code/photometric_stereo/logs/pscnn/data_aug/polar-14-shaved/learned/32
    # tensorboard --logdir=/media/li191/Backup/Junxuan_tf_code/photometric_stereo/logs/pscnn/data_aug/polar-14-shaved/scul8/learned/96/