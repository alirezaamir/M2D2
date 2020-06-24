import os
import sys
import tables
import logging
import numpy as np
from scipy import signal, integrate
import vae_model
import autoencoder_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils import create_seizure_dataset
import pickle

sys.path.append("../")
LOG = logging.getLogger(os.path.basename(__file__))
ch = logging.StreamHandler()
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch.setFormatter(logging.Formatter(log_fmt))
LOG.addHandler(ch)
LOG.setLevel(logging.INFO)

SF = 256
SEG_LENGTH = 512
EXCLUDED_SIZE = 64


def get_PCA(x):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    return principalComponents


def get_TSNE(x):
    tsne = TSNE(n_components=2)
    x_embedded = tsne.fit_transform(x)
    return x_embedded


def get_mmd(kernel, latent):
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
    return arg_max_mmd


def main():
    dirname = "../temp/vae_mmd"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    arch = 'psd'
    beta = 0.001
    latent_dim = 64
    lr = 0.0001
    decay = 0.0
    gamma = 500000.0

    root = "../output/vae/{}/".format(arch)
    stub = "seg_n_{}/beta_{}/latent_dim_{}/lr_{}/decay_{}/gamma_{}"
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

    subdirname = "{}/single_seizure".format(dirname)
    if not os.path.exists(subdirname):
        os.makedirs(subdirname)

    middle_diff = []

    sessions = create_seizure_dataset(SEG_LENGTH, SF)
    LOG.info("session number: {}".format(len(sessions.keys())))
    ae_model_args = {
        "input_shape": (latent_dim,),
        "label_shape": (1,),
        "enc_dimension": 16,
        "optim": Adam(learning_rate=0.00001),
    }
    for node in sessions.keys():  # Loop1: cross validation
        test_patient = node
        train_patients = [p for p in sessions.keys() if p != node]
        ae_model = autoencoder_model.build_model(**ae_model_args)
        for epoch in range(1):  # Loop2: epochs
            Z1_data = np.zeros((0, latent_dim))
            Z1_label = np.zeros((0,))
            for patient in train_patients:  # Loop3: train patients
                LOG.info("patient: {}".format(patient))
                X = sessions[patient]['data']
                latent = intermediate_model.predict(X)[2]
                Z1_data = np.concatenate((Z1_data, latent))

                mmd_maximum = get_mmd(kernel, latent)
                y_label = np.zeros(latent.shape[0])
                y_label[mmd_maximum - EXCLUDED_SIZE:mmd_maximum + EXCLUDED_SIZE] = 1
                Z1_label = np.concatenate((Z1_label, y_label))

            ae_model.fit(x=Z1_data, y= Z1_data, epochs=10, batch_size=64)
        break

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


if __name__ == "__main__":
    tf.config.experimental.set_visible_devices([], 'GPU')
    main()
