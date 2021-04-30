import numpy as np
import os
from utils import vae_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from sklearn.metrics.pairwise import polynomial_kernel
import pickle
import logging
import matplotlib.pyplot as plt
from utils.data import dataset_training, get_non_seizure_signal, get_epilepsiae_seizures, get_epilepsiae_test, \
    get_new_conv_w, get_epilepsiae_non_seizure
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
STATE_LEN = 899
LATENT_DIM = 32


def main():
    arch = 'vae_free'
    subdirname = "../temp/vae_mmd/integrated/{}/{}/chb_copyz_v39".format(SEG_LENGTH, arch)
    if not os.path.exists(subdirname):
        os.makedirs(subdirname)

    # log_dir = "{}/logs/{}".format(subdirname, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    middle_diff = []
    all_filenames = get_all_filenames(entire_dataset=False)
    input_dir = "../temp/vae_mmd_data/1024/epilepsiae_seizure"
    for test_id in range(1, 24):  # ["-1"]:  # range(30):  # range(1,24):
        # test_patient = pat_list[test_id]
        test_patient = str(test_id)
        train_data, train_label = dataset_training("train", test_patient, all_filenames, max_len=SEQ_LEN, state_len=STATE_LEN)
        # train_data, train_label = get_epilepsiae_seizures("train", test_patient, input_dir, max_len=SEQ_LEN,
        #                                                   state_len=STATE_LEN)
        # print("Label {}, Max {}".format(train_label.shape, np.max(train_label)))
        val_data, val_label = dataset_training("valid", test_patient, all_filenames, max_len=SEQ_LEN, state_len=STATE_LEN)
        # val_data, val_label = get_epilepsiae_seizures("valid", test_patient, input_dir, max_len=SEQ_LEN,
        #                                               state_len=STATE_LEN)

        # Load the specific weights for the model
        # load_dirname = root + stub.format(SEG_LENGTH, beta, LATENT_DIM, lr, decay, gamma, test_patient)
        # if not os.path.exists(load_dirname):
        #     print("Model does not exist in {}".format(load_dirname))
        #     exit()
        # model.load_weights(load_dirname)

        vae_mmd_model = vae_model.get_mmd_model(state_len=STATE_LEN, latent_dim=LATENT_DIM, signal_len=SEG_LENGTH,
                                                seq_len=None, trainable_vae=True)

        print(vae_mmd_model.summary())

        conv_weight = get_new_conv_w(state_len=STATE_LEN, max_window=6, state_dim=14)
        vae_mmd_model.get_layer('conv_interval').set_weights(conv_weight)

        early_stopping = EarlyStopping(
            monitor="loss", patience=50, restore_best_weights=True)
        history = CSVLogger("{}/{}_training.log".format(subdirname, test_patient))

        print("input shape: {}".format(train_data.shape))
        vae_mmd_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00001), loss='binary_crossentropy')

        vae_mmd_model.fit(x=train_data, y=train_label,
                          validation_data=(val_data, val_label), batch_size=1, epochs=25,
                          callbacks=[early_stopping, history])
            # for layer in vae_mmd_model.layers:
            #     print("name : {}".format(layer.name))
            #     if len(layer.get_weights()) != 0:
            #         print("Max : {}, Min :{}\n".format(np.max(layer.get_weights()[0]), np.min(layer.get_weights()[0])))

        savedir = '{}/model/test_{}/saved_model/'.format(subdirname, test_patient)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        vae_mmd_model.save(savedir)

    #     diffs, _ = inference(int(test_patient), trained_model=vae_mmd_model, subdirname=subdirname)
    #     middle_diff += diffs
    #
    # print(middle_diff)
    # plt.figure()
    # plt.hist(middle_diff)
    # plt.savefig("{}/hist_diff.png".format(subdirname))


def inference(test_patient, trained_model, subdirname):
    sessions = test_dataset(test_patient)
    middle_diff = []
    not_detected = {}

    if trained_model is None:
        save_path = '{}/model/test_{}/saved_model/'.format(subdirname, test_patient)
        trained_model = tf.keras.models.load_model(save_path)
        vae_mmd_model = trained_model
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
            y_true_section = np.concatenate((np.zeros(STATE_LEN), y_true, np.zeros(STATE_LEN)))
            X_section = np.expand_dims(X_section, 0)
            print("X Shape: {}".format(X_section.shape))
            X_edge = get_epilepsiae_non_seizure(test_patient, state_len=STATE_LEN)
            print("X edge : {}".format(X_edge.shape))
            print("Edge Shape: {}".format(X_edge.shape))
            concatenated = np.concatenate((X_edge, X_section, X_edge), axis=1)
            print("Concatenate Shape: {}".format(concatenated.shape))
            X_section = concatenated

            mmd_predicted = vae_mmd_model.predict(X_section)
            print("Predict : {}".format(mmd_predicted.shape))
            mmd_edge_free = mmd_predicted[:, STATE_LEN:-STATE_LEN, :]
            mmd_maximum = [np.argmax(mmd_edge_free) + STATE_LEN]
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
            if 0 < np.max(mmd_predicted) < 0.001:
                not_detected[node] = (np.max(mmd_predicted), np.min(t_diff))
            else:
                middle_diff.append(np.min(t_diff))
    return middle_diff, not_detected


def get_results():
    arch = 'vae_free'
    subdirname = "../temp/vae_mmd/integrated/{}/{}/binary_ce_v32".format(SEG_LENGTH, arch)
    diffs = []
    nc = {}
    for pat_id in range(1, 24):
        # pat = pat_list[pat_id]
        pat = pat_id
        diff_pat, not_detected_pat = inference(pat, None, subdirname)
        diffs += diff_pat
        nc.update(not_detected_pat)
    print("Differences: {}\nMedian: {}\nMean: {}".format(diffs, np.median(diffs), np.mean(diffs)))
    print("Not detected patients: {}".format(nc))
    diffs_minute = [x / 15.0 for x in diffs]
    plt.figure()
    plt.hist(diffs_minute, bins=150, range=(0, 200))
    plt.savefig("{}/hist_diff_{}.png".format(subdirname, SEQ_LEN))


def across_dataset():
    source_arch = 'epilepsiae'
    source_model = 'epilepsiae_v17'
    subdirname = "../temp/vae_mmd/integrated/{}/across/from_{}/{}".format(SEG_LENGTH, source_arch, source_model)
    if not os.path.exists(subdirname):
        os.makedirs(subdirname)
    diffs = []
    nc = {}
    save_path = '../temp/vae_mmd/integrated/{}/{}/{}/model/test_{}/saved_model/'.format(SEG_LENGTH,
                                                                                        source_arch,
                                                                                        source_model,
                                                                                        'pat_8902')
    trained_model = tf.keras.models.load_model(save_path)
    for pat_id in range(1, 25):
        pat = pat_id
        # pat = pat_list[pat_id]
        diff_pat, not_detected_pat = inference(pat, trained_model, subdirname)
        diffs += diff_pat
        nc.update(not_detected_pat)
    print("Differences: {}\nMedian: {}\nMean: {}".format(diffs, np.median(diffs), np.mean(diffs)))
    print("Not detected patients: {}".format(nc))
    diffs_minute = [x / 15.0 for x in diffs]
    plt.figure()
    plt.hist(diffs_minute, bins=150, range=(0, 200))
    plt.savefig("{}/hist_diff_{}.png".format(subdirname, SEQ_LEN))


if __name__ == "__main__":
    tf.config.experimental.set_visible_devices([], 'GPU')
    main()
    # get_results()
    # across_dataset()
