import os
import sys
import tables
import logging
import numpy as np
from scipy import signal, integrate
import vae_model
import autoencoder_model
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow as tf
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils import create_seizure_dataset
import pickle
from tensorflow.keras.callbacks import Callback, EarlyStopping, CSVLogger, LearningRateScheduler

sys.path.append("../")
LOG = logging.getLogger(os.path.basename(__file__))
ch = logging.StreamHandler()
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch.setFormatter(logging.Formatter(log_fmt))
LOG.addHandler(ch)
LOG.setLevel(logging.INFO)

SF = 256
SEG_LENGTH = 2048
EXCLUDED_SIZE = 16
AE_EPOCHS = 120
interval_len = 4


def get_PCA(x):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    return principalComponents


def get_TSNE(x):
    tsne = TSNE(n_components=2)
    x_embedded = tsne.fit_transform(x)
    return x_embedded


def get_interval_mmd(kernel, latent):
    K = kernel(latent)
    mmd = []
    input_length = latent.shape[0]
    N = 2 * interval_len
    for index in range(interval_len, input_length - interval_len):
        M = input_length - N
        start = index-interval_len
        end = index+ interval_len
        Kxx = K[start: end, start: end].sum()
        Kxy = K[start:end, :start].sum() +  K[start:end, end:].sum()
        Kyy = K[:start, :start].sum() + K[end:, end:].sum() + 2*K[end:, :start].sum()
        mmd.append(np.sqrt(
            ((1 / float(N * N)) * Kxx) +
            ((1 / float(M * M)) * Kyy) -
            ((2 / float(N * M)) * Kxy)
        ))

    ws = []
    mmd = np.array(mmd)
    mmd_corr = np.zeros(mmd.size)
    N = input_length - 1
    for ix in range(1, mmd_corr.size):
        w = (N / float(ix * (N - ix)))
        ws.append(w)
        mmd_corr[ix] = mmd[ix] - w * mmd.max()

    # arg_max_mmd = np.argmax(mmd[EXCLUDED_SIZE:-EXCLUDED_SIZE]) + EXCLUDED_SIZE
    arg_max_mmd = np.argmax(mmd_corr)
    return arg_max_mmd, mmd


def get_mmd(kernel, latent, return_mmd= False):
    K = kernel(latent)
    mmd = []
    input_length = latent.shape[0]
    for N in range(1, input_length):
        M = input_length - N
        Kxx = K[:N, :N].sum()
        Kxy = K[:N, N:].sum()
        Kyy = K[N:, N:].sum()
        mmd.append(np.sqrt(
            ((1 / float(N * N)) * Kxx) +
            ((1 / float(M * M)) * Kyy) -
            ((2 / float(N * M)) * Kxy)
        ))

    ws = []
    mmd = np.array(mmd)
    mmd_corr = np.zeros(mmd.size)
    N = input_length - 1
    for ix in range(1, mmd_corr.size):
        w = (N / float(ix * (N - ix)))
        ws.append(w)
        mmd_corr[ix] = mmd[ix] - w * mmd.max()

    arg_max_mmd = np.argmax(mmd_corr[EXCLUDED_SIZE:-EXCLUDED_SIZE]) + EXCLUDED_SIZE
    if return_mmd:
        return arg_max_mmd, mmd_corr
    else:
        return arg_max_mmd


def plot_mmd(mmd, argmax_mmd, y_true, name, dir):

    y_non_zero = np.where(y_true > 0, 1, 0)
    y_diff = np.diff(y_non_zero)
    start_points = np.where(y_diff > 0)[0]
    stop_points = np.where(y_diff < 0)[0]

    plt.figure()
    plt.plot(mmd, label="MMD")
    for seizure_start, seizure_stop in zip(start_points, stop_points):
        plt.axvspan(seizure_start, seizure_stop, color='r', alpha=0.5)
        plt.plot(argmax_mmd, mmd[argmax_mmd], 'go', markersize=12)
    plt.savefig("{}/{}_mmd_corrected.png".format(dir, name))
    plt.close()


def main():
    dirname = "../temp/vae_mmd"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    arch = 'psd'
    beta = 0.1
    latent_dim = 16
    lr = 0.0001
    decay = 0.5
    gamma = 0.0

    root = "../output/vae/{}/".format(arch)
    stub = "seg_n_{}/beta_{}/latent_dim_{}/lr_{}/decay_{}/gamma_{}/saved_model"
    dirname = root + stub.format(SEG_LENGTH, beta, latent_dim, lr, decay, gamma)
    build_model = vae_model.build_model
    build_model_args = {
        "input_shape": (SEG_LENGTH, 2,),
        "enc_dimension": latent_dim,
        "beta": beta,
        "gamma": 0,
        "optim": Adam(lr),
        "FS": SF
    }

    model, _ = build_model(**build_model_args)

    if not os.path.exists(dirname):
        print("Model does not exist in {}".format(dirname))
        exit()
    model.load_weights(dirname)
    intermediate_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[1].output)
    kernel = polynomial_kernel
    kernel_name = "polynomial_seizures"
    dirname = "../temp/vae_mmd/{}".format(kernel_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    subdirname = "{}/{}".format(dirname, SEG_LENGTH)
    if not os.path.exists(subdirname):
        os.makedirs(subdirname)

    middle_diff = []

    sessions = create_seizure_dataset(SEG_LENGTH, SF)
    LOG.info("session number: {}".format(len(sessions.keys())))
    for node in sessions.keys():  # Loop1: cross validation
        test_patient = node
        LOG.info("patient: {}".format(test_patient))
        train_patients = [p for p in sessions.keys() if p != node]
        for epoch in range(1):  # Loop2: epochs
            Z1_data = np.zeros((0, latent_dim))
            Z1_label = np.zeros((0,))
            true_label = np.zeros((0,))
            for patient in train_patients:  # Loop3: train patients
                X = sessions[patient]['data']
                y_true = sessions[patient]['label']
                latent = intermediate_model.predict(X)[2]
                Z1_data = np.concatenate((Z1_data, latent))
                true_label = np.concatenate((true_label, y_true))
                mmd_maximum, mmd = get_interval_mmd(kernel, latent)
                plot_mmd(mmd, mmd_maximum, y_true, patient, subdirname)

                mmd_label = np.zeros(latent.shape[0])
                mmd_label[mmd_maximum - EXCLUDED_SIZE:mmd_maximum + EXCLUDED_SIZE] = 1
                Z1_label = np.concatenate((Z1_label, mmd_label))
            pickle.dump({"X": Z1_data, "y": Z1_label, "label": true_label}, open("z1_{}.pickle".format(latent_dim), "wb"))

            history = CSVLogger(dirname + "/training.log")
        break
        # ae_model.fit(x=Z1_data, y=Z1_data, epochs=AE_EPOCHS, batch_size=64,
        #              verbose=2, callbacks=[history, PrintLogs(AE_EPOCHS)])

        # X_test = sessions[test_patient]['data']
        # latent = intermediate_model.predict(X_test)[2]


    # y_non_zero = np.where(y > 0, 1, 0)
    # y_diff = np.diff(y_non_zero)
    # start_points = np.where(y_diff > 0)[0]
    # stop_points = np.where(y_diff < 0)[0]
    # middle_points = (start_points + stop_points) // 2
    # LOG.info("points: {}, {}".format(start_points, stop_points))

    # plt.figure()
    # plt.plot(mmd_corr, label="MMD")
    # for seizure_start, seizure_stop in zip(start_points, stop_points):
    #     plt.axvspan(seizure_start, seizure_stop, color='r', alpha=0.5)
    # if len(node.attrs.seizures) > 0:
    #     plt.plot(arg_max_mmd, mmd_corr[arg_max_mmd], 'go', markersize=12)
    # plt.savefig("{}/{}_mmd_corrected.png".format(subdirname, node._v_name))
    # plt.close()
    #
    # t_diff = np.abs(middle_points - arg_max_mmd)
    # LOG.info("Time diff : {}".format(t_diff))
    # middle_diff.append(t_diff)
    #
    # plt.figure()
    # plt.hist(middle_diff)
    # plt.savefig("{}/hist_diff.png".format(subdirname))
    # plt.close()


def train_ae():
    data = pickle.load(open("z1.pickle", "rb"))
    X, y, label = data["X"], data["y"], data["label"]

    ae_model_args = {
        "input_shape": (64,),
        "label_shape": (1,),
        "enc_dimension": 64,
        "optim": RMSprop(),
    }
    ae_model = autoencoder_model.build_model(**ae_model_args)
    ae_model.fit(x=X, y=X, epochs=200, batch_size=64)


class PrintLogs(tf.keras.callbacks.Callback):
    def __init__(self, epochs):
        self.epochs = epochs

    def set_params(self, params):
        params['epochs'] = 0

    def on_epoch_begin(self, epoch, logs=None):
        print('Epoch %d/%d: ' % (epoch + 1, self.epochs), end='')


if __name__ == "__main__":
    # tf.config.experimental.set_visible_devices([], 'GPU')
    main()
    # train_ae()
