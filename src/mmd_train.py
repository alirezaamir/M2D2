import numpy as np
import os
from utils import vae_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics.pairwise import polynomial_kernel
import pickle
import logging
import matplotlib.pyplot as plt
from utils.data import dataset_training, get_non_seizure_signal, get_epilepsiae_seizures, get_epilepsiae_test
from utils.prepare_dataset import get_epilepsiae_non_seizure
from training import get_all_filenames
from vae_mmd import build_dataset_pickle as test_dataset
from vae_mmd import plot_mmd
from utils.params import pat_list
import datetime


LOG = logging.getLogger(os.path.basename(__file__))
ch = logging.StreamHandler()
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch.setFormatter(logging.Formatter(log_fmt))
LOG.addHandler(ch)
LOG.setLevel(logging.INFO)

SF = 256
SEG_LENGTH = 1024
EXCLUDED_SIZE = 15
interval_len = 4
SEQ_LEN = 899
STATE_LEN = 300
LATENT_DIM = 32


def main():
    arch = 'vae_sup_chb'
    beta = 1e-05
    lr = 0.0001
    decay = 0.02
    gamma = 0.0

    root = "../output/vae/{}/".format(arch)
    stub = "seg_n_{}/beta_{}/latent_dim_{}/lr_{}/decay_{}/gamma_{}/test_{}/saved_model/"
    build_model = vae_model.build_model
    build_model_args = {
        "input_shape": (SEG_LENGTH, 2,),
        "enc_dimension": LATENT_DIM,
        "beta": beta,
        "gamma": 0,
        "optim": Adam(lr),
        "FS": SF
    }

    model, encoder = build_model(**build_model_args)

    print(encoder.summary())

    subdirname = "../temp/vae_mmd/integrated/{}/{}/vae_point_mmd_v20".format(SEG_LENGTH, arch)
    if not os.path.exists(subdirname):
        os.makedirs(subdirname)

    log_dir = "{}/logs/{}".format(subdirname, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    middle_diff = []
    all_filenames = get_all_filenames(entire_dataset=True)
    # input_dir = "../temp/vae_mmd_data/1024/epilepsiae_seizure"
    for test_id in ["-1"]:  # range(30):  # range(1,25):
        # test_patient = pat_list[test_id]
        test_patient = test_id
        train_data, train_label = dataset_training("train", test_patient, all_filenames, max_len=SEQ_LEN)
        # train_data, train_label = get_epilepsiae_seizures("train", test_patient, input_dir, max_len=SEQ_LEN)
        print("Label {}, Max {}".format(train_label.shape, np.max(train_label)))
        val_data, val_label = dataset_training("valid", test_patient, all_filenames, max_len=SEQ_LEN)
        # val_data, val_label = get_epilepsiae_seizures("valid", test_patient, input_dir, max_len=SEQ_LEN)

        # train_label = np.expand_dims(train_label, -1)
        # val_label = np.expand_dims(val_label, -1)
        # train_label = tf.keras.utils.to_categorical(train_label, num_classes=SEQ_LEN)
        # val_label = tf.keras.utils.to_categorical(val_label, num_classes=SEQ_LEN)

        # Load the specific weights for the model
        load_dirname = root + stub.format(SEG_LENGTH, beta, LATENT_DIM, lr, decay, gamma, test_patient)
        if not os.path.exists(load_dirname):
            print("Model does not exist in {}".format(load_dirname))
            exit()
        model.load_weights(load_dirname)

        vae_mmd_model = vae_model.get_mmd_model(state_len=STATE_LEN, latent_dim=LATENT_DIM, signal_len=SEG_LENGTH,
                                                seq_len=None, trainable_vae=True)

        print(vae_mmd_model.summary())
        for layer_num in range(10):  # 13 for VAE
            weights = encoder.layers[layer_num].get_weights()
            vae_mmd_model.layers[layer_num].set_weights(weights)
        weights = encoder.get_layer('z').get_weights()
        vae_mmd_model.get_layer('z').set_weights(weights)

        print("input shape: {}".format(train_data.shape))
        vae_mmd_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
        # for iter in range(40):
        train_random = np.random.randn(train_data.shape[0], train_data.shape[1], LATENT_DIM)
        val_random = np.random.randn(val_data.shape[0], val_data.shape[1], LATENT_DIM)

        vae_mmd_model.fit(x=[train_data, train_random], y=train_label,
                          validation_data=([val_data, val_random], val_label), batch_size=1, epochs=200,
                          callbacks=[tensorboard_callback])

            # non_seizure_signals = get_epilepsiae_non_seizure(test_patient, STATE_LEN)
            # intermediate_model = tf.keras.models.Model(inputs=vae_mmd_model.inputs,
            #                                            outputs=vae_mmd_model.get_layer('latents').output)
            # z_non_seiz = intermediate_model.predict(x=[non_seizure_signals, train_random[:1, :STATE_LEN, :]])
            # vae_mmd_model.get_layer('MMD').set_weights(weights=[z_non_seiz[0, :, 0, :], z_non_seiz[0, :, 0, :]])

        savedir = '{}/model/test_{}/saved_model/'.format(subdirname, test_patient)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        vae_mmd_model.save(savedir)

        # diffs, _ = inference(test_patient, trained_model=vae_mmd_model, subdirname=subdirname)
        # middle_diff += diffs

    # print(middle_diff)
    # plt.figure()
    # plt.hist(middle_diff)
    # plt.savefig("{}/hist_diff.png".format(subdirname))
    # plt.show()


def inference(test_patient, trained_model, subdirname):
    sessions = get_epilepsiae_test(test_patient)
    middle_diff = []
    not_detected = {}

    if trained_model is None:
        save_path = '{}/model/test_{}/saved_model/'.format(subdirname, test_patient)
        trained_model = tf.keras.models.load_model(save_path)

        vae_mmd_model = vae_model.get_mmd_model(state_len=STATE_LEN, latent_dim=LATENT_DIM, signal_len=SEG_LENGTH,
                                                seq_len=None, trainable_vae=True)

        for layer_num in range(len(vae_mmd_model.layers)):
            weights = trained_model.layers[layer_num].get_weights()
            vae_mmd_model.layers[layer_num].set_weights(weights)
    else:
        vae_mmd_model = trained_model

    for node in sessions.keys():
        # patient_num = int(node[3:5])
        # if test_patient != patient_num:
        #     continue

        LOG.info("{}, session name: {}".format(test_patient, node))
        X = sessions[node]['data']
        LOG.info("session number: {}".format(len(X)))
        y_true = sessions[node]['label']

        if np.sum(y_true) == 0:
            continue

        for section in [-1]:  # range(X.shape[0]//SEQ_LEN):  #
            # y_true_section = y_true[SEQ_LEN*section:SEQ_LEN*(section+1)]
            #
            # if np.sum(y_true_section) == 0:
            #     continue
            #
            # X_section = X[SEQ_LEN*section:SEQ_LEN*(section+1)]

            X_section = X
            y_true_section = y_true
            X_section = np.expand_dims(X_section, 0)

            input_random = np.random.randn(X_section.shape[0], X_section.shape[1], 16)
            mmd_predicted = vae_mmd_model.predict([X_section, input_random])
            print("Predict : {}".format(mmd_predicted.shape))
            mmd_maximum = [np.argmax(mmd_predicted)]
            name = "{}_{}".format(node, section)
            plot_mmd(mmd_predicted[0, :, 0], mmd_maximum, y_true_section, name, subdirname)

            y_non_zero = np.where(y_true_section > 0, 1, 0)
            y_non_zero = np.concatenate((y_non_zero, [0]))
            # For sections which have seizure at the end or start of the section
            y_non_zero = np.concatenate(([0], y_non_zero,))
            y_diff = np.diff(y_non_zero)
            start_points = np.where(y_diff > 0)[0]
            stop_points = np.where(y_diff < 0)[0]

            accepted_points = []
            for start, stop in zip(start_points, stop_points):
                accepted_points += range(start, stop)
            middle_points = (start_points + stop_points) // 2
            LOG.info("start: {}, stop: {}\npoints: {}".format(start_points, stop_points, accepted_points))

            t_diff = np.abs(accepted_points - mmd_maximum[0])
            LOG.info("Time diff : {}".format(np.min(t_diff)))
            if np.max(mmd_predicted) < 0.01:
                not_detected[node] = (np.max(mmd_predicted), np.min(t_diff))
            else:
                middle_diff.append(np.min(t_diff))
    return middle_diff, not_detected


def get_results():
    arch = 'epilepsiae'
    subdirname = "../temp/vae_mmd/integrated/{}/{}/epilepsiae_v17".format(SEG_LENGTH, arch)
    diffs = []
    nc = {}
    for pat_id in range(30):
        pat = pat_list[pat_id]
        diff_pat, not_detected_pat = inference(pat, None, subdirname)
        diffs += diff_pat
        nc.update(not_detected_pat)
    print("Differences: {}\nMedian: {}\nMean: {}".format(diffs, np.median(diffs), np.mean(diffs)))
    print("Not detected patients: {}".format(nc))
    diffs_minute = [x/15.0 for x in diffs]
    plt.figure()
    plt.hist(diffs_minute, bins=150,range=(0, 4 * 60))
    plt.savefig("{}/hist_diff_{}.png".format(subdirname, SEQ_LEN))


# def across_dataset():
#     source_arch = 'vae_unsupervised'
#     source_model = 'not_frozen_v13'
#     subdirname = "../temp/vae_mmd/integrated/{}/across/from_{}/{}".format(SEG_LENGTH, source_arch, source_model)
#     if not os.path.exists(subdirname):
#         os.makedirs(subdirname)
#     diffs = []
#     nc = {}
#     save_path = '../temp/vae_mmd/integrated/{}/{}/{}/model/test_{}/saved_model/'.format(SEG_LENGTH,
#                                                                                         source_arch,
#                                                                                         source_model,
#                                                                                         1)
#     trained_model = tf.keras.models.load_model(save_path)
#     for pat_id in range(30):
#         pat = pat_list[pat_id]
#         diff_pat, not_detected_pat = inference(pat, trained_model, subdirname)
#         diffs += diff_pat
#         nc.update(not_detected_pat)
#     print("Differences: {}\nMedian: {}\nMean: {}".format(diffs, np.median(diffs), np.mean(diffs)))
#     print("Not detected patients: {}".format(nc))
#     plt.figure()
#     plt.hist(diffs, bins=150, range=(0, 3000 / 15))
#     plt.savefig("{}/hist_diff_{}.png".format(subdirname, SEQ_LEN))


if __name__ == "__main__":
    # tf.config.experimental.set_visible_devices([], 'GPU')
    main()
    # get_results()
    # across_dataset()