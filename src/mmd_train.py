import pprint

import numpy as np
import os
from utils import vae_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from sklearn.metrics.pairwise import polynomial_kernel
import pickle, time
import logging
import matplotlib.pyplot as plt
from utils.data import dataset_training, get_non_seizure_signal, get_epilepsiae_seizures, get_epilepsiae_test, \
    get_new_conv_w, get_epilepsiae_non_seizure, get_seizure_point_from_label
from utils.data import build_dataset_pickle as test_dataset
from training import get_all_filenames
from vae_mmd import plot_mmd
from utils.params import pat_list
import datetime
import sys

LOG = logging.getLogger(os.path.basename(__file__))
ch = logging.StreamHandler()
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch.setFormatter(logging.Formatter(log_fmt))
LOG.addHandler(ch)
LOG.setLevel(logging.INFO)

SEG_LENGTH = 1024
SEQ_LEN = 899
STATE_LEN = 899
LATENT_DIM = 32


def train_model(test_id):
    arch = '23channel'
    subdirname = "../temp/vae_mmd/integrated/{}/{}/FCN_v10".format(SEG_LENGTH, arch)
    if not os.path.exists(subdirname):
        os.makedirs(subdirname)

    middle_diff = []
    all_filenames = get_all_filenames(entire_dataset=False, full_channel=True)
    # input_dir = "../temp/vae_mmd_data/1024/epilepsiae_seizure"

    # for test_id in range(2,24):  # ["-1"]:  # range(30):  # range(1,24):
        # test_patient = pat_list[test_id]
        # test_patient = test_id
    test_patient = str(test_id)
    train_data, train_label = dataset_training("train", test_patient, all_filenames, max_len=SEQ_LEN, state_len=None)
    # train_data, train_label = get_epilepsiae_seizures("train", test_patient, input_dir, max_len=SEQ_LEN, state_len=40)
    # val_data, val_label = dataset_training("valid", test_patient, all_filenames, max_len=SEQ_LEN, state_len=None)
    # val_data, val_label = get_epilepsiae_seizures("valid", test_patient, input_dir, max_len=SEQ_LEN, state_len=40)
    print("Shape :{}".format(train_data.shape))
    train_data = np.reshape(train_data, newshape=(-1, 1024,23))
    print("Shape :{}".format(train_data.shape))
    train_label = np.reshape(train_label, newshape=(-1,))
    print("Shape :{}".format(train_label.shape))
    # train_data = np.clip(train_data, a_min=-10, a_max=10)
    # val_data = np.clip(val_data, a_min=-10, a_max=10)

    # train_data = np.reshape(train_data, newshape=(-1, 1024, 2))
    # train_label = np.reshape(train_label, newshape=(-1, 1))
    # val_data = np.reshape(val_data, newshape=(-1, 1024, 2))
    # val_label = np.reshape(val_label, newshape=(-1, 1))


    # load the model
    vae_mmd_model = vae_model.get_FCN_model(state_len=STATE_LEN, latent_dim=LATENT_DIM, signal_len=SEG_LENGTH,
                                            seq_len=None, trainable_vae=True)

    print(vae_mmd_model.summary())

    # load the weights in the convolution layer
    # conv_weight = get_new_conv_w(state_len=STATE_LEN, N=8, state_dim=18)
    # vae_mmd_model.get_layer('conv_interval').set_weights(conv_weight)

    history = CSVLogger("{}/{}_training.log".format(subdirname, test_patient))

    vae_mmd_model.compile(optimizer=tf.keras.optimizers.SGD(), loss='binary_crossentropy')

    BCE = tf.keras.losses.BinaryCrossentropy()
    bce_train = []
    bce_val = []

    savedir = '{}/model/test_{}/saved_model/'.format(subdirname, test_patient)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    vae_mmd_model.fit(x=train_data, y=train_label,
                      # validation_data=[val_data, val_label],
                      batch_size=32, epochs=40)

    # diffs = inference(test_patient, trained_model=vae_mmd_model, subdirname=subdirname, dataset='Epilepsiae')
    # middle_diff += diffs
    vae_mmd_model.save(savedir)
    # plt.figure()
    # plt.hist(middle_diff)
    # plt.savefig("{}/hist_diff.png".format(subdirname))


def inference(test_patient:int, trained_model, subdirname:str, dataset='CHB'):
    """
    To evaluate the model on the test dataset
    :param test_patient: int, number of test patient. Ex, 1 -> CHB_01
    :param trained_model: TF model, trained on the training set, if None, the method will take a saved model
    :param subdirname: str, If TF model is not passed, the method takes a saved model from this address
    :param dataset: str, 'CHB' or 'Epilepsiae'
    :return: int[], distances from the seizure for every sessions
    """

    # Load the test dataset
    if dataset == 'CHB':
        sessions = test_dataset(test_patient)
        non_seizure_dataset = get_non_seizure_signal
    else:
        sessions = get_epilepsiae_test(test_patient)
        non_seizure_dataset = get_epilepsiae_non_seizure
    diffs = []

    # Load the trained model
    if trained_model is None:
        save_path = '{}/model/test_{}/saved_model/'.format(subdirname, test_patient)
        trained_model = tf.keras.models.load_model(save_path)
        vae_mmd_model = trained_model
    else:
        vae_mmd_model = trained_model

    # Evaluate for every sessions in the test dataset
    for node in sessions.keys():
        LOG.info("{}, session name: {}".format(test_patient, node))
        X = sessions[node]['data']
        LOG.info("session number: {}".format(len(X)))
        y_true = sessions[node]['label']

        if np.sum(y_true) == 0:
            continue

        # Add non seizure signal as the initialization of the states
        # X_section = np.expand_dims(X, 0)
        # X_edge = non_seizure_dataset(test_patient, state_len=STATE_LEN)
        # concatenated = np.concatenate((X_edge, X_section, X_edge), axis=1)
        # X_section = concatenated
        # X_section = np.clip(X_section, a_min=-10, a_max=10)


        mmd_predicted = vae_mmd_model.predict(X)

        # Remove the non seizure signal to compute the MMD
        # mmd_edge_free = mmd_predicted[:, STATE_LEN:-STATE_LEN, :]
        mmd_edge_free = mmd_predicted
        mmd_maximum = [np.argmax(mmd_edge_free)]
        name = "{}".format(node)
        y_true = np.squeeze(y_true)
        plot_mmd(mmd_edge_free[:, 0], mmd_maximum, y_true, name, subdirname)

        # pprint.pprint(y_true)
        seizure_points = get_seizure_point_from_label(y_true)

        t_diff = np.abs(seizure_points - mmd_maximum[0])
        LOG.info("Time diff : {}".format(np.min(t_diff)))
        diffs.append(np.min(t_diff))
    return diffs


def get_results():
    """
    This method is only for evaluation a saved model
    """
    arch = '23channel'
    subdirname = "../temp/vae_mmd/integrated/{}/{}/FCN_v10".format(SEG_LENGTH, arch)
    diffs = []
    for pat_id in range(1,24):
        pat = pat_id
        diff_pat= inference(pat, None, subdirname, dataset='CHB')
        diffs += diff_pat
    print("Differences: {}\nMedian: {}\nMean: {}".format(diffs, np.median(diffs), np.mean(diffs)))
    diffs_minute = [x / 15.0 for x in diffs]
    plt.figure()
    plt.hist(diffs_minute, bins=150, range=(0, 200))
    plt.savefig("{}/hist_diff_{}.png".format(subdirname, SEQ_LEN))


def across_dataset():
    source_arch = 'vae_free'
    source_model = 'Epilepsiae_s16_max4_v109'
    subdirname = "../temp/vae_mmd/integrated/{}/across/from_{}/{}".format(SEG_LENGTH, source_arch, source_model)
    if not os.path.exists(subdirname):
        os.makedirs(subdirname)
    diffs = []
    nc = {}
    pat_source = "-1"
    save_path = '../temp/vae_mmd/integrated/{}/{}/{}/model/test_{}/saved_model/'.format(SEG_LENGTH,
                                                                                        source_arch,
                                                                                        source_model,
                                                                                        pat_source)
    trained_model = tf.keras.models.load_model(save_path)
    for pat_id in range(1,24):
        pat = pat_id
        # pat = pat_list[pat_id]
        diff_pat = inference(pat, trained_model, subdirname, dataset='CHB')
        diffs += diff_pat
    print("Patient {}\nDifferences: {}\nMedian: {}\nMean: {}".format(pat_source, diffs, np.median(diffs), np.mean(diffs)))
    #     diffs_minute = [x / 15.0 for x in diffs]
    # plt.figure()
    # plt.hist(diffs_minute, bins=150, range=(0, 200))
    # plt.savefig("{}/hist_diff_{}.png".format(subdirname, SEQ_LEN))


if __name__ == "__main__":
    # tf.config.experimental.set_visible_devices([], 'GPU')
    # for latent in [16, 32, 64]:
    # test_pat = int(sys.argv[1])
    # train_model(test_pat)
    get_results()
    # across_dataset()
#