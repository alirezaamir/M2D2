import numpy as np
import os
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from sklearn.metrics.pairwise import polynomial_kernel
import pickle, time
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
import logging
import matplotlib.pyplot as plt
from utils.data import dataset_training, get_non_seizure_signal, get_epilepsiae_seizures, get_epilepsiae_test, \
    get_new_conv_w, get_epilepsiae_non_seizure, get_seizure_point_from_label
from utils.data import build_dataset_pickle as test_dataset
from training import get_all_filenames
from vae_mmd import plot_mmd, get_interval_mmd, get_simplified_mmd
from utils.params import pat_list
import datetime
from scipy.stats import pearsonr

LOG = logging.getLogger(os.path.basename(__file__))
ch = logging.StreamHandler()
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch.setFormatter(logging.Formatter(log_fmt))
LOG.addHandler(ch)
LOG.setLevel(logging.INFO)

SEG_LENGTH = 1024
SEQ_LEN = 899
STATE_LEN = 20
LATENT_DIM = 16


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
        sessions = test_dataset(test_patient, root='../..')
        non_seizure_dataset = get_non_seizure_signal
    else:
        sessions = get_epilepsiae_test(test_patient, root='../..')
        non_seizure_dataset = get_epilepsiae_non_seizure
    orig_diffs = []
    smpl_diffs = []
    similarity = []
    p_value = []

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
        X_section = np.expand_dims(X, 0)
        X_edge = non_seizure_dataset(test_patient, state_len=STATE_LEN, root='../..')
        concatenated = np.concatenate((X_edge, X_section, X_edge), axis=1)
        X_section = concatenated

        intermediate_model = tf.keras.models.Model(inputs=vae_mmd_model.input,
                                                   outputs=[
                                                       vae_mmd_model.get_layer('MMD').input])
        latent = intermediate_model.predict(X_section)
        print("Shape : {}".format(latent.shape))

        latent = np.squeeze(latent, axis=0)
        latent = np.squeeze(latent, axis=1)
        K = polynomial_kernel(latent)
        K_rbf = rbf_kernel(latent)

        original_mmd = get_simplified_mmd(K)
        simplified_mmd = get_simplified_mmd(K_rbf)
        print(original_mmd.shape)

        remove_len = STATE_LEN - 6

        # Remove the non seizure signal to compute the MMD
        seizure_points = get_seizure_point_from_label(y_true)
        original_mmd = original_mmd[remove_len: -remove_len]
        mmd_maximum = [np.argmax(original_mmd)]
        name = "{}_polynomial".format(node)
        plot_mmd(original_mmd, mmd_maximum, y_true, name, subdirname)
        t_diff = np.abs(seizure_points - mmd_maximum[0])
        orig_diffs.append(np.min(t_diff))

        simplified_mmd = simplified_mmd[remove_len: -remove_len]
        mmd_maximum = [np.argmax(simplified_mmd)]
        name = "{}_rbf".format(node)
        plot_mmd(simplified_mmd, mmd_maximum, y_true, name, subdirname)
        t_diff = np.abs(seizure_points - mmd_maximum[0])
        smpl_diffs.append(np.min(t_diff))

        simil, p = pearsonr(original_mmd, simplified_mmd)
        similarity.append(simil)
        p_value.append(p)
        print("Correlation: {} ".format(simil))
        print("Diff in original and simplified: {}".format(np.subtract(smpl_diffs, orig_diffs)))

    return orig_diffs, smpl_diffs


def get_results():
    """
    This method is only for evaluation a saved model
    """
    arch = 'vae_free'
    subdirname = "../../temp/vae_mmd/integrated/{}/{}/z_minus1_v52".format(SEG_LENGTH, arch)
    orig_diffs = []
    smpl_diffs = []
    for pat_id in range(1, 24):
        pat = pat_id
        orig, smpl = inference(pat, None, subdirname, dataset='CHB')
        orig_diffs += orig
        smpl_diffs += smpl

    for diffs in [orig_diffs, smpl_diffs]:
        print("Differences: {}\nMedian: {}\nMean: {}".format(diffs, np.median(diffs), np.mean(diffs)))
        # diffs_minute = [x / 15.0 for x in diffs]
        # plt.figure()
        # plt.hist(diffs_minute, bins=150, range=(0, 200))
        # plt.savefig("{}/hist_diff_{}.png".format(subdirname, SEQ_LEN))


def across_dataset():
    source_arch = 'vae_free'
    source_model = 'no_mmd_v63'
    subdirname = "../temp/vae_mmd/integrated/{}/across/from_{}/{}".format(SEG_LENGTH, source_arch, source_model)
    if not os.path.exists(subdirname):
        os.makedirs(subdirname)
    diffs = []
    nc = {}
    save_path = '../temp/vae_mmd/integrated/{}/{}/{}/model/test_{}/saved_model/'.format(SEG_LENGTH,
                                                                                        source_arch,
                                                                                        source_model,
                                                                                        -1)
    trained_model = tf.keras.models.load_model(save_path)
    for pat_id in range(30):
        # pat = pat_id
        pat = pat_list[pat_id]
        diff_pat = inference(pat, trained_model, subdirname, dataset='Epilepsiae')
        diffs += diff_pat
    print("Differences: {}\nMedian: {}\nMean: {}".format(diffs, np.median(diffs), np.mean(diffs)))
    diffs_minute = [x / 15.0 for x in diffs]
    plt.figure()
    plt.hist(diffs_minute, bins=150, range=(0, 200))
    plt.savefig("{}/hist_diff_{}.png".format(subdirname, SEQ_LEN))


if __name__ == "__main__":
    tf.config.experimental.set_visible_devices([], 'GPU')
    get_results()
    # across_dataset()
